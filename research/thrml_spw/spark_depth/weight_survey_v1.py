"""CPU-only all-parameter range/control/noise survey with neuron-axis arrays.

No neural forwards, random sampling, fitting or heldout evaluation. The
vectorized encoding is checked against the accepted original scalar-group codec.
"""
from pathlib import Path
import hashlib,json,math,time
import numpy as np
from .prepare_v1 import OUT,CHECKPOINT,sha


def mapped(flat,group):
    flat=np.asarray(flat,np.float32).reshape(-1);n=len(flat);pad=(-n)%group
    x=np.pad(flat,(0,pad),constant_values=np.nan).reshape(-1,group)if pad else flat.reshape(-1,group)
    low=np.nanmin(x,axis=1);span=np.nanmax(x,axis=1)-low
    unit=np.divide(x.astype(np.float64)-low[:,None],span[:,None],out=np.full(x.shape,.5,np.float64),where=span[:,None]!=0).reshape(-1)[:n]
    edge=1/(1+np.exp(8.));p=np.clip(unit,edge,1-edge);h=.5*(np.log(p)-np.log1p(-p))
    codes=np.rint(np.clip((h+4)*(4095/8),0,4095)).astype(np.uint16)
    fields=codes.astype(np.float32)*np.float32(8/4095)-np.float32(4)
    prob=1/(1+np.exp(-2*fields.astype(np.float64)))
    expanded_low=np.repeat(low,group)[:n].astype(np.float64);expanded_span=np.repeat(span,group)[:n].astype(np.float64)
    mean=expanded_low+expanded_span*prob;variance=expanded_span**2*prob*(1-prob)
    return codes,low,span,mean,variance


def codec_fixture():
    from research.thrml_spw.broad.parameter_program_v1 import encode,expectation
    checked=0
    for n in [17,128,257]:
        for constant in [False,True]:
            w=np.full(n,.125,np.float32)if constant else np.linspace(-2,3,n,dtype=np.float32)
            a=encode(w);c,l,s,m,v=mapped(w,128)
            for x,y in [(c,a['codes']),(l,a['lower']),(s,a['span']),(m,expectation(a,1))]:np.testing.assert_array_equal(x,y)
            checked+=n
    return checked


def main():
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(2);assert not torch.cuda.is_initialized()
    prepared=json.loads((OUT/'prepared.json').read_text());assert prepared['checkpoint']==str(CHECKPOINT)
    inventory=Path('results/thrml_spw/broad_verification_20261105/remote_metadata/additional_parameter_packages_v1/gpt2/package.json')
    old=json.loads(inventory.read_text());assert sha(CHECKPOINT/'model.safetensors')==old['checkpoint_sha256']
    destination=OUT/'weight_survey_v1';destination.mkdir(exist_ok=False)
    fixture=codec_fixture();begin=time.monotonic()
    model=AutoModelForCausalLM.from_pretrained(CHECKPOINT,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='eager').cpu().eval()
    assert not torch.cuda.is_initialized() and all(p.device.type=='cpu'for p in model.parameters())
    parameters=dict(model.named_parameters());assert sum(p.numel()for p in parameters.values())==124439808
    assert list(parameters)==[r['name']for r in old['canonical_parameters']]
    module_lookup=dict(model.named_modules());axis_arrays={};reports=[]
    for record in old['canonical_parameters']:
        name=record['name'];param=parameters[name].detach();index=record['index']
        assert hashlib.sha256(param.view(torch.uint16).numpy().tobytes()).hexdigest()==record['original_BF16_raw_sha256']
        w=param.float().numpy();n=w.size;module=module_lookup[name.rsplit('.',1)[0]];kind=type(module).__name__
        rms=math.sqrt(float(np.mean(w.astype(np.float64)**2)));conditions=[]
        layouts=['storage_flat','transposed_matrix']if w.ndim==2 else['storage_flat']
        for layout in layouts:
            encoded=w.T.copy()if layout=='transposed_matrix'else w
            for group in [16,32,64,128]:
                c,lo,span,mean,var=mapped(encoded.reshape(-1),group)
                mean=mean.reshape(encoded.shape);var=var.reshape(encoded.shape)
                if layout=='transposed_matrix':mean=mean.T;var=var.T
                bias=mean-w.astype(np.float64);noise_total=float(var.sum());signal=float(np.sum(w.astype(np.float64)**2))
                conditions.append({'layout':layout,'group_size':group,'range_groups':len(lo),
                    'range_metadata_bits':64*len(lo),'finite_control_bits':12*n,'logical_cell_state_bits':n,
                    'ideal_packed_control_range_state_bits_per_weight':13+64*len(lo)/n,
                    'constant_range_groups':int(np.count_nonzero(span==0)),
                    'controller_bias_RMSE':math.sqrt(float(np.mean(bias*bias))),
                    'controller_bias_relative_RMSE':math.sqrt(float(np.sum(bias*bias))/max(signal,1e-30)),
                    'predicted_noise_RMSE_T8192':math.sqrt(noise_total/n/8192),
                    'predicted_noise_relative_RMSE_T8192':math.sqrt(noise_total/max(signal,1e-30)/8192),
                    'predicted_noise_relative_RMSE_T262144':math.sqrt(noise_total/max(signal,1e-30)/262144),
                    'max_group_span':float(span.max()),'max_coordinate_noise_std_T8192':float(np.sqrt(var.max()/8192))})
                if group==128:
                    for axis in range(w.ndim):
                        other=tuple(i for i in range(w.ndim)if i!=axis)
                        key=f'p{index:03d}_{layout}_axis{axis}'
                        axis_arrays[key+'_weight_energy']=np.sum(w.astype(np.float64)**2,axis=other)
                        axis_arrays[key+'_noise_variance_coefficient']=np.sum(var,axis=other)
                        axis_arrays[key+'_controller_bias_energy']=np.sum(bias*bias,axis=other)
        report={'index':index,'name':name,'shape':list(w.shape),'elements':n,'module_class':kind,'weight_RMS':rms,
            'max_abs_weight':float(np.max(np.abs(w))),'weight_zero_fraction':float(np.mean(w==0)),
            'conditions':conditions,
            'axes':'Conv1Dweights[inputfeature,outputneuron]; Linearweights[outputneuron,inputfeature]; Embeddingweights[tokenID,feature]; vectors[norm/biascoordinate]. Bothaxes retained for2Dtensors.'}
        reports.append(report);(destination/'progress.json').write_text(json.dumps({'index':index,'name':name,'elapsed_seconds':time.monotonic()-begin})+'\n')
        print(json.dumps({'surveyed':index,'name':name,'variants':len(conditions)}),flush=True)
    assert len(reports)==148 and not torch.cuda.is_initialized()
    arrays=destination/'axis_statistics.npz'
    with arrays.open('xb')as f:np.savez_compressed(f,**axis_arrays)
    result={'status':'all_GPT2_parameter_ranges_controller_bias_predicted_noise_and_axis_statistics_surveyed_CPU_only',
        'source_sha256':sha(__file__),'prepared_sha256':sha(OUT/'prepared.json'),'old_package_sha256':sha(inventory),
        'original_codec_fixture_coordinates':fixture,'canonical_parameters':148,'unique_parameters':124439808,
        'parameters':reports,'axis_arrays':str(arrays),'axis_arrays_sha256':sha(arrays),'elapsed_seconds':time.monotonic()-begin,
        'scope':'Exactfinite12bitcontrolmean and independentBernoulliweightvariancebeforeBF16readoutrounding, acrossminmaxgroups16/32/64/128 andtwo2Dlayouts. T8192/262144variancepredictionsonly,notsamples orneuralquality. BF16sourcepointverifiedbytensorhashes. NoGPUwork,testlabels,parameterfittingoruniversalclaim. Everyvariantcountsonecellperscalar plusallcontrol/rangebits; group16canexceedBF16payload.'}
    with (destination/'results.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items()if k!='parameters'}),flush=True)

if __name__=='__main__':main()
