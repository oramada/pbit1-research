"""Independent whole-checkpoint package oracle using raw safetensors tensors.

Does not instantiate the producer model or import its encoder. Group reductions
use reduceat rather than producer group loops; all parameter codes are checked.
"""
from pathlib import Path
import hashlib,json,math,time
import numpy as np
import torch
from safetensors.torch import load_file

BASE=Path('results/thrml_spw/broad_verification_20261105')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
    return h.hexdigest()

def checkpoint_key(model_type,name):
    if model_type=='gpt2':
        assert name.startswith('transformer.')
        return name[len('transformer.'):]
    assert model_type=='distilbert'
    return name


def main():
    torch.set_num_threads(4);begin=time.monotonic()
    root=BASE/'additional_parameter_packages_v1';result_path=root/'results.json'
    result=json.loads(result_path.read_text());inventory=json.loads((BASE/'additional_model_inventory/parameter_inventory.json').read_text())
    assets=json.loads((BASE/'additional_assets_v1/manifest.json').read_text())
    assert len(result['packages'])==len(inventory['models'])==2
    reports=[]
    for produced,meta in zip(result['packages'],inventory['models']):
        assert produced['model']==meta['model'] and sha(produced['path'])==produced['sha256']
        p=json.loads(Path(produced['path']).read_text())
        assert p['model']==meta['model'] and p['revision']==meta['revision'] and p['aliases']==meta['aliases']
        checkpoint=Path(assets['checkpoint_locations'][meta['model']])/'model.safetensors'
        assert sha(checkpoint)==p['checkpoint_sha256']
        source=load_file(checkpoint,device='cpu')
        assert len(p['canonical_parameters'])==len(meta['canonical_parameters'])
        mapping={v['name']:checkpoint_key(meta['model_type'],v['name']) for v in meta['canonical_parameters']}
        assert len(set(mapping.values()))==len(mapping)
        extras=set(source)-set(mapping.values())
        if meta['model_type']=='gpt2':
            expected_masks={f'h.{i}.attn.bias' for i in range(12)}
            assert extras==expected_masks
            mask=torch.tril(torch.ones((1024,1024),dtype=torch.float32)).reshape(1,1,1024,1024)
            assert all(torch.equal(source[name],mask) for name in extras)
        else:
            assert extras==set()
        excluded_mask_scalars=sum(source[name].numel() for name in extras)

        scalars=groups=bias=constant=serialized=0
        for actual,expected in zip(p['canonical_parameters'],meta['canonical_parameters']):
            name=expected['name'];assert actual['name']==name
            tensor=source[mapping[name]].to(torch.bfloat16)
            assert list(tensor.shape)==actual['shape']==expected['shape']
            assert tensor.numel()==actual['elements']==expected['elements']
            assert hashlib.sha256(tensor.view(torch.uint16).numpy().tobytes()).hexdigest()==actual['original_BF16_raw_sha256']
            path=Path(actual['program']);assert sha(path)==actual['program_sha256']
            with np.load(path,allow_pickle=False) as a:
                assert set(a.files)=={'codes','lower','span','shape','group_size','format_version'}
                assert int(a['group_size'])==128 and int(a['format_version'])==1
                assert a['shape'].dtype==np.int64 and a['lower'].dtype==a['span'].dtype==np.float32
                assert list(a['shape'])==expected['shape']
                w=tensor.float().numpy().reshape(-1);n=len(w)
                starts=np.arange(0,n,128);lower=np.minimum.reduceat(w,starts);upper=np.maximum.reduceat(w,starts)
                span=(upper-lower).astype(np.float32)
                assert a['codes'].dtype==np.uint16 and a['codes'].shape==(n,)
                np.testing.assert_array_equal(a['lower'],lower);np.testing.assert_array_equal(a['span'],span)
                ll=np.repeat(lower,128)[:n].astype(np.float64);rr=np.repeat(span,128)[:n].astype(np.float64)
                unit=np.divide(w.astype(np.float64)-ll,rr,out=np.full(n,.5),where=rr!=0)
                bound=1/(1+math.exp(8));prob=np.maximum(bound,np.minimum(1-bound,unit))
                h=(np.log(prob)-np.log1p(-prob))*.5
                codes=np.rint(np.minimum(4095,np.maximum(0,(h+4)*(4095/8)))).astype(np.uint16)
                np.testing.assert_array_equal(a['codes'],codes)
                r=actual['allocation'];ng=math.ceil(n/128);padding=math.ceil(n/65536)*65536
                assert r['unique_parameter_scalars']==r['logical_pbits']==n
                assert r['range_groups']==ng and r['range_metadata_bits']==64*ng
                assert r['finite_control_information_bits']==12*n and r['last_group_actual_size']==n-128*(ng-1)
                assert r['logical_retained_device_evaluations']==8192*n
                assert r['scheduled_retained_device_evaluations']==8192*padding and r['scheduled_warmup_device_evaluations']==padding
                assert actual['constant_range_groups']==int((span==0).sum())
                assert r['software_int32_count_carry_bytes']==4*65536 and r['software_retained_count_array_bytes']==4*n
                scalars+=n;groups+=ng;constant+=int((span==0).sum());bias+=n if name.endswith('.bias') else 0
                serialized+=path.stat().st_size
                assert path.stat().st_size==actual['program_serialized_bytes']
            del tensor,w,ll,rr,prob,h,codes
        assert [v['index'] for v in p['canonical_parameters']]==list(range(len(meta['canonical_parameters'])))
        assert scalars==meta['unique_learned_scalars']==p['unique_learned_scalars']
        assert bias==meta['bias_scalars']==p['bias_scalars'] and groups==p['range_groups']
        assert p['range_metadata_bits']==64*groups and p['finite_controller_information_bits']==12*scalars
        assert p['one_logical_pbit_per_scalar'] and p['all_constants_still_assigned_and_charged']
        reports.append({'model':p['model'],'package_sha256':produced['sha256'],'canonical_parameters':len(p['canonical_parameters']),
            'all_parameter_codes_checked':scalars,'bias_scalars_checked':bias,'range_groups_checked':groups,
            'constant_groups_still_charged':constant,'range_metadata_bits':64*groups,'finite_control_information_bits':12*scalars,
            'serialized_program_bytes':serialized,'canonical_to_raw_checkpoint_keys':mapping,
            'unlearned_checkpoint_mask_scalars_excluded':excluded_mask_scalars})
        del source
    audit={'status':'both_complete_parameter_packages_and_every_code_range_BF16point_alias_and_allocation_independently_verified',
        'source_sha256':sha(Path(__file__)),'results_sha256':sha(result_path),'packages':reports,
        'elapsed_seconds':time.monotonic()-begin,
        'scope':'Every control code and group checked against raw pinned safetensors rounded to BF16, complete canonical coverage including biases and norms, metadata/controller costs explicit. No neural forward, task quality, full-model stochastic sampling or physical density proof.'}
    (root/'independent_audit_v2.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit))

if __name__=='__main__':main()
