"""Independent public paths, explicit gradients, optimizer, selection and test audit."""
import argparse,csv,hashlib,json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from .calibrated_mnist_path_audit import PublicLayer
from .verify_transformer_control import sha
from .verify_calibrated_mnist import independently_score
OUT=Path('results/thrml_spw/probability_finetune');DATA=Path('data/thrml_spw/positive_mnist');TRAIN=Path('data/thrml_spw/native_training/training_data.npz')


def device(model,index,base):
    z=np.random.default_rng(np.random.SeedSequence([base,model,index])).standard_normal((2,2368))
    return np.exp(.2*z[1]-.02),.2*z[0]


def control(theta):
    c=np.rint((np.clip(theta.astype(float)/2,-4,4)+4)*255/8)
    return (-4+c*8/255).astype(np.float32)


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def independent_gradient(theta,x,y,pre,logits):
    p=1/(1+np.exp(-theta.astype(float)));p1=p[:,2048:].reshape(-1,32,10)
    q=np.exp(logits.astype(float)-logits.max(-1,keepdims=True));q/=q.sum(-1,keepdims=True)
    ce=-np.log(np.take_along_axis(q,y[:,:,None],axis=2)[:,:,0]).mean(1)
    for b in range(len(q)):q[b,np.arange(y.shape[1]),y[b]]-=1
    q/=y.shape[1];hidden=np.maximum(pre.astype(float),0)
    g1=np.einsum('bni,bno->bio',hidden,q).reshape(-1,320)*p[:,2048:]*(1-p[:,2048:])
    delta=np.einsum('bno,bio->bni',q,p1)*(pre>0)
    g0=np.einsum('bni,bno->bio',x.astype(float),delta).reshape(-1,2048)*p[:,:2048]*(1-p[:,:2048])
    return ce,np.concatenate([g0,g1],axis=1)


def initial(model):
    e=np.load(DATA/f'seed{model}.npz');w=[e[f'weight_{i}'] for i in range(2)];bias=[e[f'bias_{i}'] for i in range(2)]
    flat=np.concatenate([a.reshape(-1) for a in w]).astype(float);eps=1/(1+np.exp(8));flat=np.clip(flat,eps,1-eps)
    return (np.log(flat)-np.log1p(-flat)).astype(np.float32),bias,w


def scaled_error(a,b):return float(np.max(np.abs(a.astype(float)-b.astype(float)))/(1+np.max(np.abs(b))))


def training_audit():
    assert jax.default_backend()=='gpu';p=json.loads((OUT/'protocol.json').read_text());report=json.loads((OUT/'TRAINING_REPORT.json').read_text());manifest=json.loads((OUT/'training_manifest.json').read_text())
    assert manifest['config']==p and manifest['result']==report and manifest['source_commit']=='2d54cd8404df85bb65703d7e8b4e84e48c1b72a9'
    for n,h in p['source_sha256'].items():assert sha(n)==h
    assert sha(TRAIN)==p['training_data_sha256'];data=np.load(TRAIN);x,y=data['train_inputs'],data['train_labels'];vx,vy=data['validation_inputs'],data['validation_labels']
    assert len(set(data['train_indices']).intersection(data['validation_indices']))==0
    ids=np.arange(55000);assert len(x)==len(y)==55000 and len(vx)==5000
    vb=np.pad(vx,((0,24),(0,0))).reshape(157,32,64)
    selected=json.loads((OUT/'selected.json').read_text());assert len(selected)==15 and sha(OUT/'selected.json')==report['selection_sha256']
    metrics=read(OUT/'validation_metrics.csv');assert len(metrics)==450
    max_forward=max_grad=max_optimizer=max_loss=0.;native_steps=expected_steps=validation_cases=0;choices=[]
    for model in range(5):
        assert sha(DATA/f'seed{model}.npz')==p['model_sha256'][str(model)]
        init,bias,w=initial(model);reference=np.maximum(vx@w[0]+bias[0],0)@w[1]+bias[1]
        layers=[PublicLayer(np.full((64,32),.5)),PublicLayer(np.full((32,10),.5))]
        for l in layers:assert l.n_free==l.size and not l.clamped
        fns=[l.frequencies(64) for l in layers]
        dynamic=[jax.jit(jax.vmap(lambda k,f,fn=fn:fn(k[None],f)[0])) for fn in fns]
        orders=np.load(OUT/'orders'/f'model{model}.npz')['indices'];assert orders.shape==(5,215,256)
        for epoch in range(1,6):
            order=np.random.default_rng(np.random.SeedSequence([2026113600,model,epoch])).permutation(55000)
            np.testing.assert_array_equal(orders[epoch-1],np.concatenate([order,order[:40]]).reshape(215,256))
        for method in ['expected','native_ideal','native_device']:
            previous=init.copy();moment=np.zeros(2368,np.float32);variance=np.zeros(2368,np.float32);candidates=[]
            for epoch in range(6):
                name=f'model{model}_{method}_epoch{epoch}';tp=OUT/'training'/f'{name}.npz';record=np.load(tp);tm=json.loads(tp.with_suffix('.json').read_text());assert sha(tp)==tm['archive_sha256']
                np.testing.assert_array_equal(record['bias0'],bias[0]);np.testing.assert_array_equal(record['bias1'],bias[1])
                if not epoch:
                    np.testing.assert_array_equal(record['theta_final'],init);assert record['theta_before'].size==0
                else:
                    theta=record['theta_before'];grads=record['gradients'];pre=record['preactivation0'];logits=record['logits'];loss=record['loss'];xx=x[orders[epoch-1]];yy=y[orders[epoch-1]]
                    assert theta.shape==grads.shape==(215,2368) and pre.shape==(215,256,32) and logits.shape==(215,256,10)
                    assert np.all((theta>=-8)&(theta<=8));np.testing.assert_array_equal(theta[0],previous)
                    if method=='expected':
                        pp=1/(1+np.exp(-theta.astype(float)));ep=np.einsum('bni,bio->bno',xx.astype(float),pp[:,:2048].reshape(215,64,32))+bias[0]
                        el=np.einsum('bni,bio->bno',np.maximum(pre,0).astype(float),pp[:,2048:].reshape(215,32,10))+bias[1];expected_steps+=215
                    else:
                        fields=[];keys=[]
                        for batch in range(215):
                            index=(epoch-1)*215+batch;a,d=device(model,index,2026110600) if method=='native_device' else (np.ones(2368),np.zeros(2368))
                            fields.append((a*control(theta[batch]).astype(float)+d).astype(np.float32))
                            key=jax.random.key(1110730)
                            for value in [model,epoch,batch]:key=jax.random.fold_in(key,value)
                            keys.append(jax.random.split(key))
                        fields=np.stack(fields);keys=jnp.stack(keys)
                        q0=np.asarray(dynamic[0](keys[:,0],jnp.asarray(fields[:,:2048])))
                        ep=np.einsum('bni,bio->bno',xx.astype(float),q0.astype(float))+bias[0]
                        q1=np.asarray(dynamic[1](keys[:,1],jnp.asarray(fields[:,2048:])))
                        el=np.einsum('bni,bio->bno',np.maximum(pre,0).astype(float),q1.astype(float))+bias[1];native_steps+=215
                    err=max(scaled_error(pre,ep),scaled_error(logits,el));max_forward=max(max_forward,err);assert err<2e-5,(name,'forward',err)
                    ce,expected=independent_gradient(theta,xx,yy,pre,logits);ge=float(np.max(np.abs(grads-expected)/(2e-6+2e-4*np.abs(expected))));max_grad=max(max_grad,ge);assert ge<1,(name,'STE gradient normalized error',ge)
                    le=float(np.max(np.abs(loss-ce)));max_loss=max(max_loss,le);assert le<2e-5,(name,'loss',le)
                    for batch in range(215):
                        g=grads[batch];step=(epoch-1)*215+batch+1
                        moment=np.float32(.9)*moment+np.float32(.1)*g;variance=np.float32(.999)*variance+np.float32(.001)*g*g
                        bc1=np.float32(1)-np.power(np.float32(.9),np.float32(step));bc2=np.float32(1)-np.power(np.float32(.999),np.float32(step))
                        update=np.clip(theta[batch]-np.float32(.001)*(moment/bc1)/(np.sqrt(variance/bc2)+np.float32(1e-8)),-8,8)
                        target=theta[batch+1] if batch<214 else record['theta_final'];oe=scaled_error(update,target);max_optimizer=max(max_optimizer,oe);assert oe<3e-5,(name,'Adam',batch,oe)
                    np.testing.assert_allclose(moment,record['moment'],atol=3e-6,rtol=3e-5);np.testing.assert_allclose(variance,record['variance'],atol=3e-7,rtol=3e-5)
                previous=record['theta_final'];assert np.all((previous>=-8)&(previous<=8))
                assert tm['training_steps']==(215 if epoch else 0) and tm['native_free_evaluations']==(215*64*2368 if epoch and method!='expected' else 0)
                vp=OUT/'validation'/f'{name}.npz';raw=np.load(vp);vm=json.loads(vp.with_suffix('.json').read_text());assert sha(vp)==vm['archive_sha256'] and sha(tp)==vm['training_archive_sha256']
                actual=[]
                for dev in range(5):
                    a,d=device(model,dev,2026111600);fields=(a*control(previous).astype(float)+d).astype(np.float32)
                    key=jax.random.key(1120730)
                    for value in [model,dev]:key=jax.random.fold_in(key,value)
                    keys=jax.vmap(lambda b:jax.random.split(jax.random.fold_in(key,b)))(jnp.arange(157))
                    q0=np.asarray(fns[0](keys[:,0],jnp.asarray(fields[:2048])));ep=(np.einsum('bni,bio->bno',vb.astype(float),q0.astype(float))+bias[0]).reshape(-1,32)[:5000]
                    pe=scaled_error(raw['preactivation0'][dev],ep)
                    hb=np.pad(np.maximum(raw['preactivation0'][dev],0),((0,24),(0,0))).reshape(157,32,32)
                    q1=np.asarray(fns[1](keys[:,1],jnp.asarray(fields[2048:])));el=(np.einsum('bni,bio->bno',hb.astype(float),q1.astype(float))+bias[1]).reshape(-1,10)[:5000]
                    err=max(pe,scaled_error(raw['logits'][dev],el));assert err<2e-5,(name,'validation path',dev,err);max_forward=max(max_forward,err)
                    row=vm['rows'][dev];independently_score(raw['logits'][dev],vy,reference,row)
                    csvrow=next(r for r in metrics if int(r['model'])==model and r['method']==method and int(r['epoch'])==epoch and int(r['device'])==dev)
                    for k,v in row.items():assert str(v)==csvrow[k]
                    actual.append(float(row['cross_entropy']));validation_cases+=1
                assert abs(np.mean(actual)-vm['mean_native_validation_CE'])<1e-15;candidates.append(vm)
                print(f'Public native/gradient/Adam/validation audit {name}',flush=True)
            chosen=min(candidates,key=lambda c:(c['mean_native_validation_CE'],c['epoch']));sel=selected[f'model{model}_{method}'];assert sel['selected_epoch']==chosen['epoch'] and sel['validation_CE']==chosen['mean_native_validation_CE']
            cp=OUT/'selected'/f'model{model}_{method}.npz';assert sha(cp)==sel['checkpoint_sha256'];a=np.load(cp);np.testing.assert_array_equal(a['theta'],np.load(sel['training_archive'])['theta_final']);assert sha(sel['training_archive'])==sel['training_archive_sha256']
            choices.append({'model':model,'method':method,'epoch':chosen['epoch']})
        jax.clear_caches()
    assert native_steps==10750 and expected_steps==5375 and validation_cases==450
    assert report['native_training_free_evaluations']==native_steps*64*2368 and report['native_validation_free_evaluations']==450*157*64*2368
    audit={'passed':True,'native_training_updates':native_steps,'expected_training_updates':expected_steps,'native_validation_full_sets':validation_cases,'all_STE_gradients_and_Adam_updates_checked':True,'all_checkpoints_validation_selected_before_test':True,'max_scaled_forward_error':max_forward,'max_gradient_error_normalized_to_tolerance':max_grad,'gradient_tolerance':'2e-6+2e-4*abs(reference)','max_scaled_one_step_Adam_error':max_optimizer,'max_training_CE_absolute_error':max_loss,'selected_epochs':choices,'source_sha256':sha(__file__),
        'scope':'All native training and validation paths independently replayed using public THRML frequency factors; every biased expected-linear gradient checked against explicit algebra and every optimizer update checked. This validates the stated surrogate, not unbiased differentiation or generalization.'}
    (OUT/'training_path_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))


def test_audit():
    assert jax.default_backend()=='gpu';p=json.loads((OUT/'test_protocol.json').read_text());report=json.loads((OUT/'TEST_REPORT.json').read_text());manifest=json.loads((OUT/'test_manifest.json').read_text())
    assert manifest['config']==p and manifest['result']==report and manifest['source_commit']=='2d54cd8404df85bb65703d7e8b4e84e48c1b72a9'
    for n,h in p['source_sha256'].items():assert sha(n)==h
    assert sha(OUT/'selected.json')==p['selection_sha256']==report['selection_frozen_before_test'] and sha(OUT/'training_manifest.json')==p['training_manifest_sha256']
    assert sha(DATA/'evaluation_data.npz')==p['evaluation_data_sha256'];d=np.load(DATA/'evaluation_data.npz');x,y=d['test_inputs'],d['test_labels'];xb=np.pad(x,((0,16),(0,0))).reshape(313,32,64)
    rows=read(OUT/'test_metrics.csv');base=read(OUT/'test_baseline_metrics.csv');assert len(rows)==600 and len(base)==80
    assert {(int(r['model']),int(r['device']),r['kind'],r['condition'],int(r['T'])) for r in rows}=={(m,dev,k,c,t) for m in range(5) for dev in range(5) for k in ['initial','expected','native_ideal','native_device'] for c in ['ideal','uncalibrated','cal4096'] for t in [64,256]}
    total=0;max_error=0.;cal_probes=0
    for model in range(5):
        init,bias,w=initial(model);reference=np.maximum(x@w[0]+bias[0],0)@w[1]+bias[1];checkpoints={'initial':init}
        for kind in ['expected','native_ideal','native_device']:checkpoints[kind]=np.load(OUT/'selected'/f'model{model}_{kind}.npz')['theta']
        layers=[PublicLayer(np.full((64,32),.5)),PublicLayer(np.full((32,10),.5))];public=[{t:l.frequencies(t) for t in [64,256]} for l in layers];probe=PublicLayer(np.full(2368,.5)).frequencies(2048)
        for kind,theta in checkpoints.items():
            a=np.load(OUT/'test_baselines'/f'model{model}_{kind}.npz');ref=a['original_FP32'];np.testing.assert_allclose(ref,reference,atol=2e-5)
            pp=(1/(1+np.exp(-theta.astype(float)))).astype(np.float32);ww=[pp[:2048].reshape(64,32),pp[2048:].reshape(32,10)]
            expected={'original_FP32':reference,'expected':np.maximum(x@ww[0]+bias[0],0)@ww[1]+bias[1]}
            for bits in [4,8]:
                qw=[]
                for weight in ww:
                    scale=float(weight.max())/((1<<bits)-1);qw.append((np.rint(weight/scale)*scale).astype(np.float32))
                expected[f'UINT{bits}']=np.maximum(x@qw[0]+bias[0],0)@qw[1]+bias[1]
            for condition,value in expected.items():
                np.testing.assert_allclose(a[condition],value,atol=2e-5)
                row=next(r for r in base if int(r['model'])==model and r['kind']==kind and r['condition']==condition);independently_score(a[condition],y,ref,row)
        for dev in range(5):
            cp=OUT/'test_devices'/f'model{model}_device{dev}.npz';cal=np.load(cp);cm=json.loads(cp.with_suffix('.json').read_text());assert sha(cp)==cm['archive_sha256']
            gain,offset=device(model,dev,2026112600);np.testing.assert_array_equal(cal['gain'],gain);np.testing.assert_array_equal(cal['offset'],offset)
            pf=(-4+np.array([96,159])*8/255).astype(np.float32).astype(float);u=[]
            for pi,field in enumerate(pf):
                fields=(gain*field+offset).astype(np.float32);key=jax.random.key(1130731)
                for value in [model,dev,pi]:key=jax.random.fold_in(key,value)
                q=np.asarray(probe(key[None],jnp.asarray(fields)))[0];counts=np.rint(q*2048).astype(np.int32)
                np.testing.assert_array_equal(cal[f'count{pi}'],counts);u.append(.5*(np.log(counts+.5)-np.log(2048-counts+.5)));cal_probes+=1
            raw=(u[1]-u[0])/(pf[1]-pf[0]);ah=np.clip(raw,.25,4);dh=(u[1]+u[0])/2-raw*(pf[1]+pf[0])/2
            np.testing.assert_allclose(cal['estimated_gain'],ah,atol=2e-14,rtol=1e-13);np.testing.assert_allclose(cal['estimated_offset'],dh,atol=2e-14,rtol=1e-13)
            for kind,theta in checkpoints.items():
                nominal=control(theta);codes=np.rint((np.clip((nominal.astype(float)-dh)/ah,-4,4)+4)*255/8);corrected=(-4+codes*8/255).astype(np.float32)
                fieldsets={'ideal':nominal,'uncalibrated':(gain*nominal.astype(float)+offset).astype(np.float32),'cal4096':(gain*corrected.astype(float)+offset).astype(np.float32)}
                for t in [64,256]:
                    key=jax.random.key(1130730)
                    for value in [model,dev,t]:key=jax.random.fold_in(key,value)
                    keys=jax.vmap(lambda b:jax.random.split(jax.random.fold_in(key,b)))(jnp.arange(313))
                    for condition,fields in fieldsets.items():
                        name=f'model{model}_device{dev}_{kind}_{condition}_T{t}';rn=OUT/'test_runs'/f'{name}.npz';a=np.load(rn);row=json.loads(rn.with_suffix('.json').read_text());assert sha(rn)==row['archive_sha256']
                        assert hashlib.sha256(np.ascontiguousarray(fields).tobytes()).hexdigest()==row['fields_sha256']
                        q0=np.asarray(public[0][t](keys[:,0],jnp.asarray(fields[:2048])));pre=(np.einsum('bni,bio->bno',xb.astype(float),q0.astype(float))+bias[0]).reshape(-1,32)[:10000]
                        hb=np.pad(np.maximum(a['preactivation0'],0),((0,16),(0,0))).reshape(313,32,32)
                        q1=np.asarray(public[1][t](keys[:,1],jnp.asarray(fields[2048:])));ll=(np.einsum('bni,bio->bno',hb.astype(float),q1.astype(float))+bias[1]).reshape(-1,10)[:10000]
                        err=max(scaled_error(a['preactivation0'],pre),scaled_error(a['logits'],ll));assert err<2e-5,(name,err);max_error=max(max_error,err)
                        csvrow=next(r for r in rows if int(r['model'])==model and int(r['device'])==dev and r['kind']==kind and r['condition']==condition and int(r['T'])==t)
                        for k,v in row.items():assert str(v)==csvrow[k]
                        ref=np.load(OUT/'test_baselines'/f'model{model}_{kind}.npz')['original_FP32'];independently_score(a['logits'],y,ref,row)
                        assert row['inference_free_evaluations']==row['logical_weight_readouts']==313*t*2368 and row['binary_products_with_padding']==10016*t*2368 and row['binary_products_without_padding']==10000*t*2368;total+=313*t*2368
            print(f'Public reserved test/calibration/full-score audit model{model} device{dev}',flush=True)
        jax.clear_caches()
    assert cal_probes==50 and report['native_test_free_evaluations']==total and report['native_test_calibration_free_evaluations']==25*4096*2368
    audit={'passed':True,'native_full_test_runs':600,'examples_per_run':10000,'public_probe_windows':50,'max_scaled_forward_error':max_error,'selection_sha256':sha(OUT/'selected.json'),'source_sha256':sha(__file__),
        'scope':'All reserved-device full-test fields, measured counts/fits, classical controls, scores/counters and both layer paths independently reconstructed using public THRML factors. Existing test set, no physical claim.'}
    (OUT/'test_path_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))


def main():
    a=argparse.ArgumentParser();a.add_argument('--phase',choices=['training','test'],required=True);args=a.parse_args()
    if args.phase=='training':training_audit()
    else:test_audit()
    if (OUT/'training_path_audit.json').exists() and (OUT/'test_path_audit.json').exists():
        train=json.loads((OUT/'training_path_audit.json').read_text());test=json.loads((OUT/'test_path_audit.json').read_text());assert train['passed'] and test['passed']
        (OUT/'evidence_audit.json').write_text(json.dumps({'passed':True,'native_training_updates':train['native_training_updates'],'expected_training_updates':train['expected_training_updates'],'native_validation_full_sets':450,'native_full_test_runs':600,'native_test_probe_windows':50,'all_training_gradients_and_optimizer_steps_checked':True,'source_sha256':sha(__file__),'scope':'Combined independent native training/validation/test path audits, explicit biased-STE gradients, optimizer, selection, finite controls and full task scores. Five paired starting models; no universal or hardware performance claim.'},indent=2)+'\n')


if __name__=='__main__':main()
