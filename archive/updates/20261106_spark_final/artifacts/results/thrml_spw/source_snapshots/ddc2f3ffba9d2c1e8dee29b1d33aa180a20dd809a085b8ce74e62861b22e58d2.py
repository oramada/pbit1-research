"""Evaluate sealed precision controls; compare only after parent audits close."""
import argparse,json
import numpy as np
import torch
from results.thrml_spw.cifar_cnn_native.precision_baselines.common import ROOT,OUT,PARENT,TRAIN,NAMES,sha,write,save,guard,parent_controls,parent_closed,output,metric
def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['evaluate','statistics']);args=parser.parse_args();p=json.loads((OUT/'protocol.json').read_text());guard(p);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    if args.action=='evaluate':
        parent_controls(p);assert not (OUT/'evaluation.json').exists();d=np.load(PARENT/'test_data.npz');x=d['inputs'];training=json.loads((TRAIN/'manifest.json').read_text());arrays={}
        for mi,row in enumerate(training['rows']):
            for ci,name in enumerate(NAMES):arrays[f'c{ci}_model{mi}']=np.concatenate([output(x[first:first+128],row,name) for first in range(0,10000,128)]);assert np.isfinite(arrays[f'c{ci}_model{mi}']).all();print(mi,name,flush=True)
        save(OUT/'logits.npz',**arrays);guard(p);write(OUT/'evaluation.json',{'complete':True,'protocol_sha256':sha(OUT/'protocol.json'),'test_data_sha256':sha(PARENT/'test_data.npz'),'parent_controls_sha256':sha(PARENT/'controls.json'),'logits_sha256':sha(OUT/'logits.npz'),'new_native_updates':0});return
    parent_closed(p);assert not (OUT/'statistics.json').exists();d=np.load(PARENT/'test_data.npz');saved=np.load(OUT/'logits.npz');parent=np.load(PARENT/'statistical_arrays.npz');metrics=np.array([[metric(saved[f'c{ci}_model{mi}'],d['labels']) for mi in range(5)] for ci in range(4)]);bm,bs=parent['bootstrap_models'],parent['bootstrap_seeds'];dist={};contrasts=[]
    for bi,t in enumerate(p['samples']):
        for ci,name in enumerate(NAMES):
            delta=parent['values'][bi]-metrics[ci,:,None,:];z=delta[bm[:,:,None],bs].mean((1,2));dist[f'budget{bi}_control{ci}']=z;contrasts.append({'T':t,'budget_index':bi,'control_index':ci,'control':name,'difference':delta.mean((0,1)).tolist(),'descriptive_95_interval':np.quantile(z,[.025,.975],axis=0).T.tolist()})
    save(OUT/'statistical_arrays.npz',control_metrics=metrics,**dist);write(OUT/'statistics.json',{'parent_statistics_sha256':sha(PARENT/'statistics.json'),'parent_statistical_arrays_sha256':sha(PARENT/'statistical_arrays.npz'),'evaluation_sha256':sha(OUT/'evaluation.json'),'logits_sha256':sha(OUT/'logits.npz'),'summary':{name:metrics[ci].mean(0).tolist() for ci,name in enumerate(NAMES)},'contrasts':contrasts,'metric_order':['accuracy_fraction','mean_CE_nats'],'statistical_arrays_sha256':sha(OUT/'statistical_arrays.npz'),'scope':p['scope']})
if __name__=='__main__':main()
