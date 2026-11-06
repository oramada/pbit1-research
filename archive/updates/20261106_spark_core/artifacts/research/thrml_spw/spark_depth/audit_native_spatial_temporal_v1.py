"""Audit native count readouts and the actual positional-word reference."""
import json
import numpy as np
from .prepare_v1 import OUT,sha
from research.thrml_spw.broad.parameter_program_v1 import word_codes

def main():
    root=OUT/'native_spatial_temporal_v1';result=json.loads((root/'results.json').read_text());protocol=json.loads((root/'protocol.json').read_text());dest=OUT/'native_spatial_temporal_audit_v1';dest.mkdir(exist_ok=False)
    assert len(result['records'])==20
    with np.load(OUT/'native_head_v1/inputs.npz')as f:d={k:f[k].copy()for k in f.files}
    X=d['hidden'].astype(np.float64);W=d['original_weights'].astype(np.float64);eps=1/(1+np.exp(8.));checked=0
    for layout in ['storage_flat','transposed_matrix']:
        code=d[layout+'_codes'];program={'codes':code.reshape(-1),'lower':np.zeros(48,np.float32),'span':np.ones(48,np.float32),'shape':np.array([8,768],np.int64),'group_size':128,'format_version':1}
        reference_words=word_codes(program,4).reshape(8,768)
        p=1/(1+np.exp(-2*(code.astype(np.float32)*np.float32(8/4095)-np.float32(4)).astype(np.float64)))
        independent_words=np.floor(np.clip((p-eps)/(1-2*eps),0,1)*15+.5).astype(np.uint8)
        np.testing.assert_array_equal(reference_words,independent_words)
        lo=d[layout+'_lower'].astype(np.float64);span=d[layout+'_span'].astype(np.float64)
        for r in [r for r in result['records']if r['layout']==layout]:
            path=root/f"{layout}_{r['method']}_seed{r['seed']}.npz";assert sha(path)==r['counts_file_sha256']
            with np.load(path)as f:counts=f['counts'];saved=f['estimate']
            T=r['samples'];assert counts.dtype==np.int32 and np.all((counts>=0)&(counts<=T))
            if r['method']=='K1':
                assert counts.shape==(8,768);unit=counts/T;mean=p;var=p*(1-p)/T
            else:
                assert counts.shape==(4,8,768);unit=np.tensordot(np.array([1,2,4,8],np.int64),counts.astype(np.int64),axes=1)/(15*T);mean=eps+(1-2*eps)*reference_words/15.;var=np.full(W.shape,eps*(1-eps)*85/(225*T))
            reconstructed=lo+span*unit;np.testing.assert_array_equal(reconstructed,saved)
            actual=float(np.mean((X@(reconstructed-W).T)**2));theory=float(np.mean((X@(lo+span*mean-W).T)**2)+np.mean(X**2@(span**2*var).T))
            assert np.isclose(actual,r['output_MSE_about_original'],atol=1e-12,rtol=1e-10);assert np.isclose(theory,r['predicted_output_MSE_about_original'],atol=1e-12,rtol=1e-10)
            assert r['observations']==201326592;checked+=1
    audit={'status':'all20_native_count_readouts_and_reference_words_verified','source_sha256':sha(__file__),'result_sha256':sha(root/'results.json'),'checked_cases':checked,
        'scope':'Equaldeviceobservationsbutdifferentcontrolprecision: K1twelvebitprobabilitycode versusK4fourbitword. InternalstudybaselineusingofficialTHRML, notavendorproductionbaseline; no fullmodelorphysicalcostclaim.'}
    (dest/'results.json').write_text(json.dumps(audit,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5))
    for layout,color in [('storage_flat','tomato'),('transposed_matrix','teal')]:
        y=[result['summary'][layout][method]['mean_output_MSE']for method in ['K1','K4']];ax.plot([1,4],y,'o-',color=color,label=layout)
    ax.set(yscale='log',xticks=[1,4],xticklabels=['K1: 12-bit control\n32,768 samples','K4: 4-bit word\n8,192 samples'],ylabel='Mean output MSE against original weights',title='Native THRML real-head comparison, five seeds\nEqual device observations; different numerical formats')
    ax.legend();fig.tight_layout();fig.savefig(dest/'equal_observation_head.png',dpi=160);fig.savefig(dest/'equal_observation_head.svg')
    print(json.dumps(result['summary'],indent=2))

if __name__=='__main__':main()
