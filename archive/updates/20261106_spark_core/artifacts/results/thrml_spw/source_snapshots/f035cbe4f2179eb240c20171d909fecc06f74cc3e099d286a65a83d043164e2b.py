"""Nominal word-mean nonlinear plug-in, no additional stochastic draws."""
import json
import numpy as np
from results.thrml_spw.fashion_cnn_native.word_comparison.common import OUT,PARENT,TRAIN,sha,write,save,guard,parent_closed,mean_weights
from results.thrml_spw.fashion_cnn_native.cohort.common import numpy_control
def main():
    p=json.loads((OUT/'protocol.json').read_text());guard(p);parent_closed();assert not (OUT/'controls.json').exists();x=np.load(PARENT/'test_data.npz')['inputs'];models=json.loads((TRAIN/'manifest.json').read_text())['rows'];data={};weight_data={}
    for mi,row in enumerate(models):
        model=np.load(row['path']);ww=mean_weights(model);data[f'model{mi}']=np.concatenate([numpy_control(x[first:first+128],model,ww,True) for first in range(0,10000,128)])
        for li,w in enumerate(ww):weight_data[f'model{mi}_layer{li}']=w
    save(OUT/'mean_logits.npz',**data);save(OUT/'mean_weights.npz',**weight_data);guard(p);write(OUT/'controls.json',{'complete':True,'protocol_sha256':sha(OUT/'protocol.json'),'parent_test_data_sha256':sha(PARENT/'test_data.npz'),'mean_logits_sha256':sha(OUT/'mean_logits.npz'),'mean_weights_sha256':sha(OUT/'mean_weights.npz'),'scope':'Nominal finite±4field word mean weights followed by deterministic ACT8 nonlinear network; not E[finite-sample CNN] or actual calibrated device law.'})
if __name__=='__main__':main()
