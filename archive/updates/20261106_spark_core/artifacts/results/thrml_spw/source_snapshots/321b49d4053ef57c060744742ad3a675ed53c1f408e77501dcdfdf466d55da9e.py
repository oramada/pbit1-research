"""Inference-only CPU-master parameter offloading for LFM3 eager leaf calls.

This does not offload buffers, support training, or cover direct functional
weight access. Use the documented LFM3 eager convolution fallback. A context
is tied to a fixed model weight point; create a new context after replacement.
"""
import torch

class LeafOffload:
    def __init__(self,model,device='cuda',cpu_copy_test=False):
        self.model=model;self.device=torch.device(device);self.cpu_copy_test=cpu_copy_test
        self.parameters={id(p):p for p in model.parameters()}
        if any(p.device.type!='cpu'for p in self.parameters.values()):raise ValueError('CPU-resident masters required')
        self.masters={key:p.detach()for key,p in self.parameters.items()}
        self.refs={key:0 for key in self.parameters};self.stack={};self.handles=[];self.active_bytes=0;self.peak_active_bytes=0;self.calls=[]
    def enter_module(self,name,module,args):
        if torch.is_grad_enabled():raise RuntimeError('LeafOffload requires torch.inference_mode or no_grad')
        entered=[];self.stack.setdefault(id(module),[]).append(entered)
        for p in module.parameters(recurse=False):
            key=id(p)
            if self.refs[key]==0:
                materialized=self.masters[key].to(self.device)
                if self.device.type=='cpu'and self.cpu_copy_test:materialized=materialized.clone()
                p.data=materialized;self.active_bytes+=p.numel()*p.element_size();self.peak_active_bytes=max(self.peak_active_bytes,self.active_bytes)
            self.refs[key]+=1;entered.append(key)
        self.calls.append(name)
    def exit_module(self,module,args,output):
        stack=self.stack.get(id(module),[])
        if not stack:return
        for key in reversed(stack.pop()):
            self.refs[key]-=1
            if self.refs[key]==0:
                p=self.parameters[key];p.data=self.masters[key];self.active_bytes-=p.numel()*p.element_size()
    def __enter__(self):
        if self.model.training:raise RuntimeError('model.eval() required')
        for name,module in self.model.named_modules():
            if any(True for _ in module.parameters(recurse=False)):
                self.handles.append(module.register_forward_pre_hook(lambda m,a,name=name:self.enter_module(name,m,a)))
                self.handles.append(module.register_forward_hook(self.exit_module,always_call=True))
        return self
    def __exit__(self,typ,value,traceback):
        for h in self.handles:h.remove()
        self.handles=[]
        for key,p in self.parameters.items():p.data=self.masters[key];self.refs[key]=0
        self.active_bytes=0;self.stack.clear()
        return False
