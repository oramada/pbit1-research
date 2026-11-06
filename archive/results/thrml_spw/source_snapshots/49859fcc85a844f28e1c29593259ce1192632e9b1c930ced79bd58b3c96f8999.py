"""Five paired finite-control probability fine-tuning runs per method; validation only."""
import csv,json,time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from .probability_training_native import TrainingNetwork,METHODS,initial_theta,profile,command,adam
from .controlled_positive_native import ControlledPositiveNetwork
from .calibrated_mnist import forward,scores
from .verify_transformer_control import sha
from .provenance import manifest
OUT=Path('results/thrml_spw/probability_finetune');DATA=Path('data/thrml_spw/positive_mnist');TRAIN=Path('data/thrml_spw/native_training/training_data.npz')


def main():
    if jax.default_backend()!='gpu':raise RuntimeError('Native CUDA training required')
    OUT.mkdir(exist_ok=True)
    for folder in ['training','validation','selected','orders']:(OUT/folder).mkdir(exist_ok=True)
    files=['probability_finetune.py','probability_training_native.py','native_training.py','controlled_positive_native.py','positive_inference.py','native.py','calibration_count_native.py','calibrated_mnist.py']
    protocol={'schema':'spw-probability-finetune-five-paired-v1','methods':list(METHODS),'model_seeds':list(range(5)),'epochs':5,'candidate_epochs':list(range(6)),
        'training_examples':55000,'validation_examples':5000,'train_batch':256,'batches_per_epoch':215,'native_train_samples':64,'validation_samples':64,'validation_profiles':5,'validation_batch':32,
        'optimizer':'Adam .9/.999/eps1e-8 lr.001; theta-only, cliptheta[-8,8] aftereveryupdate; deterministicneuralbiases frozen frompairedFP32checkpoint. Bothoptimizer moments startzero separatelyformethod.',
        'mapping':'p=sigmoid(theta), onefreeSpinNode perlearnedmatrixweight, noaffinerange orendpointclamps. Nativefields use nearest8bit[-4,4] codeforfloat64(theta/2), then F64gain*command+offset emittedF32. Theta finite, so allp lieinopenunitinterval.',
        'training':'expected uses deterministicexpectedweights; native_ideal uses actual THRML outputaccumulation T64; native_device uses samenativeforwardwith gainlognormalmean1/logSD.2,offsetSD.2 redrawn perminibatch butheldthroughall64draws. Thisdistributionaugmentation isnotdevicecharacterization orcalibration. NetworkfirstlinearmeanprecedesReLU.',
        'gradient':'ExplicitbiasedSTE: nativeforwardvalues, backwardthroughunquantizedexpectedlinears andsigmoidonly; backwardusesexpectedsecondmatrix, notactualsamplemeanmatrix. No differentiationthroughGibbs, controlrounding, gainoroffset. No unbiasedgradientclaim.',
        'orders':'default_rng(SeedSequence([2026113600,model,epoch1..5])).permutation(55000), pad40withfirst40; ordersidenticalacrossmethods. Each215batcheshassize256.',
        'profiles':'probability_training_native.profile usesSeedSequence([domainbase,model,index]); bases train2026110600/validation2026111600/test2026112600. independentnormalrows0offset/1gain; staticgainexp(.2*z1-.02),offset.2*z0. Trainindexglobalstep0..1074. Validationindex0..4staticfullset. Testindex0..4reservedfromselection.',
        'keys':'Training1110730foldmodel,epoch1..5,minibatch0..214 thensplitlayers. Validation1120730foldmodel,profile thendatasetbatchandsplitlayers. Samevalidationnoise/profilesacrossmethods/epochs toreducecomparisonnoise; selectioncouldoverfitthatfinitebank, so testkeys/devicesareseparate.',
        'selection':'For eachmethod/modelchooseminimummeanvalidationnativeuncalibratedCE across5fixedprofiles amongepochs0..5; tieearlierepoch. Fixedschedule,nohyperparametergrid.No testarraysorscores used beforeselected.json isfrozen.',
        'prospective_test':'Afterall15runs/selectionfreeze,evaluateinitialclippedcheckpointandthreechosenmethods onall10000existingtestimages,5reservedstaticdeviceprofiles,ideal/uncalibrated/measuredcal4096,T64and256. Primarynative_device minusexpectedatuncalibratedT64. Nativeideal andcalibrated/ideal/T256 controlssecondary. NumericalFP32/UINT4/UINT8 controlsforeachcheckpoint. Oldtestsetisnotfreshdata; no test-drivenchangesallowed.',
        'scope':'Fiveindependentlypretrainedstartingmodels, pairedfine-tuningmethods. Not15independentinitializationsorfrom-scratchtraining. Freepbitprecision usesfloatingoptimizerstates andfiniteprogrammingcodes; softwaretrainingcostisnotphysicaltrainingenergy. No drift or temporal/crossdevicecorrelation here.',
        'training_data_sha256':sha(TRAIN),'model_sha256':{str(m):sha(DATA/f'seed{m}.npz') for m in range(5)},
        'source_sha256':{str(Path('research/thrml_spw')/f):sha(Path('research/thrml_spw')/f) for f in files}}
    pp=OUT/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==protocol
    else:pp.write_text(json.dumps(protocol,indent=2)+'\n')
    archive=Path('results/thrml_spw/source_snapshots');archive.mkdir(exist_ok=True)
    for n,h in protocol['source_sha256'].items():
        path=archive/(h+'.py');data=Path(n).read_bytes()
        if path.exists():assert path.read_bytes()==data
        else:path.write_bytes(data)
    d=np.load(TRAIN);x,y=d['train_inputs'],d['train_labels'];vx,vy=d['validation_inputs'],d['validation_labels']
    assert len(x)==55000 and len(vx)==5000
    vb=jnp.asarray(np.pad(vx,((0,24),(0,0))).reshape(157,32,64));selection={};allrows=[];started=time.monotonic();update=jax.jit(adam)
    for model in range(5):
        e=np.load(DATA/f'seed{model}.npz');weights=[e[f'weight_{i}'] for i in range(2)];bias=[e[f'bias_{i}'] for i in range(2)];init=initial_theta(weights)
        reference=forward(vx,weights,bias)
        templates=[np.full((64,32),.5),np.full((32,10),.5)]
        infer=ControlledPositiveNetwork(templates,bias).dataset(64);training=TrainingNetwork(bias,64)
        orders=[]
        for epoch in range(1,6):
            order=np.random.default_rng(np.random.SeedSequence([2026113600,model,epoch])).permutation(55000);orders.append(np.concatenate([order,order[:40]]).reshape(215,256))
        np.savez_compressed(OUT/'orders'/f'model{model}.npz',indices=np.stack(orders))
        for method in METHODS:
            theta=jnp.asarray(init);mom=jnp.zeros_like(theta);var=jnp.zeros_like(theta);fn=training.objective(method);candidate=[]
            for epoch in range(6):
                name=f'model{model}_{method}_epoch{epoch}';tp=OUT/'training'/f'{name}.npz';tj=tp.with_suffix('.json')
                if tj.exists():
                    tm=json.loads(tj.read_text());assert sha(tp)==tm['archive_sha256'];record=np.load(tp);theta=jnp.asarray(record['theta_final']);mom=jnp.asarray(record['moment']);var=jnp.asarray(record['variance'])
                else:
                    records={'theta_before':[],'gradients':[],'preactivation0':[],'logits':[],'loss':[]}
                    if epoch:
                        for batch,ids in enumerate(orders[epoch-1]):
                            before=np.asarray(theta);index=(epoch-1)*215+batch
                            gain,offset=profile(model,index,'train') if method=='native_device' else (np.ones(2368),np.zeros(2368))
                            fields=(gain*command(before).astype(float)+offset).astype(np.float32)
                            key=jax.random.key(1110730)
                            for value in [model,epoch,batch]:key=jax.random.fold_in(key,value)
                            (loss,(pre,logits)),grad=fn(theta,jnp.asarray(x[ids]),jnp.asarray(y[ids],dtype=jnp.int32),key,jnp.asarray(fields))
                            for k,value in [('theta_before',before),('gradients',grad),('preactivation0',pre),('logits',logits),('loss',loss)]:records[k].append(np.asarray(value))
                            theta,mom,var=update(theta,grad,mom,var,index+1)
                    arrays={k:np.stack(v) if v else np.empty((0,),np.float32) for k,v in records.items()}
                    np.savez_compressed(tp,**arrays,theta_final=np.asarray(theta),moment=np.asarray(mom),variance=np.asarray(var),bias0=bias[0],bias1=bias[1])
                    tm={'model':model,'method':method,'epoch':epoch,'archive_sha256':sha(tp),'training_steps':215 if epoch else 0,
                        'native_free_evaluations':215*64*2368 if epoch and method!='expected' else 0}
                    tj.write_text(json.dumps(tm,indent=2)+'\n')
                vp=OUT/'validation'/f'{name}.npz';vj=vp.with_suffix('.json')
                if vj.exists():
                    vm=json.loads(vj.read_text());assert sha(vp)==vm['archive_sha256'] and vm['training_archive_sha256']==sha(tp)
                else:
                    allpre=[];alllogits=[];vr=[];nominal=command(np.asarray(theta))
                    for device in range(5):
                        gain,offset=profile(model,device,'validation');fields=(gain*nominal.astype(float)+offset).astype(np.float32)
                        key=jax.random.key(1120730)
                        for value in [model,device]:key=jax.random.fold_in(key,value)
                        pre,logits=[np.asarray(v).reshape(-1,v.shape[-1])[:5000] for v in infer(key,vb,jnp.asarray(fields))]
                        allpre.append(pre);alllogits.append(logits);vr.append({'model':model,'method':method,'epoch':epoch,'device':device,**scores(logits,vy,reference)})
                    np.savez_compressed(vp,preactivation0=np.stack(allpre),logits=np.stack(alllogits))
                    vm={'model':model,'method':method,'epoch':epoch,'archive_sha256':sha(vp),'training_archive_sha256':sha(tp),
                        'mean_native_validation_CE':float(np.mean([r['cross_entropy'] for r in vr])),'mean_native_validation_accuracy':float(np.mean([r['accuracy'] for r in vr])),
                        'native_free_evaluations':5*157*64*2368,'rows':vr}
                    vj.write_text(json.dumps(vm,indent=2)+'\n')
                candidate.append(vm);allrows.extend(vm['rows'])
                print(f'{name} valCE={vm["mean_native_validation_CE"]:.5f} valAcc={vm["mean_native_validation_accuracy"]:.5f} elapsed={time.monotonic()-started:.1f}s',flush=True)
            chosen=min(candidate,key=lambda c:(c['mean_native_validation_CE'],c['epoch']));epoch=chosen['epoch'];name=f'model{model}_{method}_epoch{epoch}'
            selected=np.load(OUT/'training'/f'{name}.npz');cp=OUT/'selected'/f'model{model}_{method}.npz'
            np.savez_compressed(cp,theta=selected['theta_final'],bias0=bias[0],bias1=bias[1])
            selection[f'model{model}_{method}']={'model':model,'method':method,'selected_epoch':epoch,'checkpoint_sha256':sha(cp),'training_archive':str(OUT/'training'/f'{name}.npz'),'training_archive_sha256':chosen['training_archive_sha256'],'validation_archive_sha256':chosen['archive_sha256'],'validation_CE':chosen['mean_native_validation_CE']}
        jax.clear_caches()
    assert len(selection)==15 and len(allrows)==450
    (OUT/'selected.json').write_text(json.dumps(selection,indent=2)+'\n')
    with (OUT/'validation_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
    result={'fine_tuning_runs':15,'independent_starting_models':5,'training_updates':15*5*215,'native_training_updates':10*5*215,
        'native_training_free_evaluations':10*5*215*64*2368,'native_validation_free_evaluations':90*5*157*64*2368,
        'selection_sha256':sha(OUT/'selected.json'),'elapsed_seconds':time.monotonic()-started,'scope':protocol['scope']}
    (OUT/'TRAINING_REPORT.json').write_text(json.dumps(result,indent=2)+'\n');manifest(OUT/'training_manifest.json',protocol,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
