"""Independent scalar-loop range/command/word/count/readout/resource audit."""
from pathlib import Path
import hashlib,json,math
import numpy as np

OUT=Path('results/thrml_spw/broad_verification_20261105/parameter_counts_h100_v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    result=json.loads((OUT/'results.json').read_text());states_checked=0;values_checked=0
    assert len(result['cases'])==9 and result['backend']=='gpu'
    assert {(c['parameter_index'],c['bits']) for c in result['cases']}=={(i,k) for i in range(3) for k in [1,4,8]}
    for c in result['cases']:
        assert sha(c['path'])==c['sha256'] and c['samples']*c['bits']==1024
        with np.load(c['path'],allow_pickle=False) as a:
            w=a['weight'];flat=w.ravel();n=len(flat);k=c['bits'];t=c['samples'];counts=a['counts']
            assert list(w.shape)==c['shape'] and np.array_equal(a['shape'],w.shape)
            assert int(a['group_size'])==128 and int(a['format_version'])==1
            low,span=[],[];expected_codes=[];unit=[]
            for begin in range(0,n,128):
                values=flat[begin:begin+128]
                l=min(float(v) for v in values);r=float(np.float32(max(float(v) for v in values)-l))
                low.append(l);span.append(r)
                for value in values:
                    p=(float(value)-l)/r if r else .5
                    p=max(1/(1+math.exp(8)),min(1/(1+math.exp(-8)),p));unit.append(p)
                    h=.5*(math.log(p)-math.log1p(-p))
                    expected_codes.append(round(max(0,min(4095,(h+4)*4095/8))))
            np.testing.assert_array_equal(a['lower'],np.asarray(low,np.float32))
            np.testing.assert_array_equal(a['span'],np.asarray(span,np.float32))
            np.testing.assert_array_equal(a['codes'],np.asarray(expected_codes,np.uint16))
            control_fields=np.array([np.float32(np.float32(v)*np.float32(8/4095))-np.float32(4) for v in expected_codes],np.float32)
            p1=np.array([1/(1+math.exp(-2*float(h))) for h in control_fields])
            endpoints=[1/(1+math.exp(8)),1/(1+math.exp(-8))]
            words=np.array([round(max(0,min(1,(p-endpoints[0])/(endpoints[1]-endpoints[0])))*((1<<k)-1)) for p in p1],np.int64) if k>1 else None
            rebuilt=np.zeros((k,n),np.int64);coverage=np.zeros_like(rebuilt)
            for index,record in enumerate(c['raw_tiles']):
                z=a[f'raw{index}_states'];h=a[f'raw{index}_fields'];total=a[f'raw{index}_padded_counts']
                assert z.dtype==np.bool_ and z.shape==(t,64) and h.dtype==np.float32 and h.shape==(64,)
                begin,width,bit=record['begin'],record['width'],record['bit']
                assert begin==64*record['tile'] and width==min(64,n-begin)
                expected=(control_fields[begin:begin+width] if k==1 else
                    np.asarray([4 if (word>>bit)&1 else -4 for word in words[begin:begin+width]],np.float32))
                np.testing.assert_array_equal(h,np.pad(expected,(0,64-width)))
                independent=np.array([sum(bool(v) for v in z[:,j]) for j in range(64)],np.int64)
                np.testing.assert_array_equal(total,independent)
                rebuilt[bit,begin:begin+width]=independent[:width];coverage[bit,begin:begin+width]+=1
                states_checked+=z.size
            assert np.all(coverage==1);np.testing.assert_array_equal(counts,rebuilt)
            y=[];mean=[]
            for i in range(n):
                stochastic=sum(int(rebuilt[b,i])*(1<<b) for b in range(k))/(t*((1<<k)-1))
                expectation=(p1[i] if k==1 else endpoints[0]+(endpoints[1]-endpoints[0])*int(words[i])/((1<<k)-1))
                group=i//128;y.append(low[group]+span[group]*stochastic);mean.append(low[group]+span[group]*expectation)
            np.testing.assert_allclose(a['decoded'].ravel(),y,rtol=1e-14,atol=1e-14)
            np.testing.assert_allclose(a['expectation'].ravel(),mean,rtol=1e-14,atol=1e-14)
            for bits in [2,4,8]:
                expected=[]
                for i in range(n):
                    g=i//128;p=(float(flat[i])-low[g])/span[g] if span[g] else .5
                    q=round(max(0,min(1,p))*((1<<bits)-1))/((1<<bits)-1)
                    expected.append(low[g]+span[g]*q)
                np.testing.assert_allclose(a['int'+str(bits)].ravel(),expected,rtol=1e-14,atol=1e-14)
            r=c['resources'];groups=math.ceil(n/128);padded=math.ceil(n/64)*64
            assert r['logical_pbits']==n*k and r['unique_parameter_scalars']==n
            assert r['logical_retained_device_evaluations']==n*1024
            assert r['scheduled_retained_device_evaluations']==padded*1024 and r['scheduled_warmup_device_evaluations']==k*padded
            assert r['range_groups']==groups and r['last_group_actual_size']==n-128*(groups-1)
            assert r['range_metadata_bits']==64*groups and r['finite_control_information_bits']==(12*n if k==1 else k*n)
            assert r['software_retained_count_array_bytes']==4*k*n and r['software_int32_count_carry_bytes']==256
            assert r['software_decoded_FP64_array_bytes']==8*n and r['software_sampler_state_slots']==64
            values_checked+=n
    cpu=Path('results/thrml_spw/broad_verification_20261105/parameter_counts_cpu_v1')
    reference=json.loads((cpu/'results.json').read_text())
    state_differences=count_differences=0
    for current,old in zip(result['cases'],reference['cases']):
        assert (current['parameter_index'],current['bits'])==(old['parameter_index'],old['bits'])
        assert sha(old['path'])==old['sha256']
        with np.load(current['path'],allow_pickle=False) as a,np.load(old['path'],allow_pickle=False) as b:
            np.testing.assert_array_equal(a['codes'],b['codes'])
            for ordinal in range(len(current['raw_tiles'])):
                np.testing.assert_array_equal(a[f'raw{ordinal}_fields'],b[f'raw{ordinal}_fields'])
                state_differences+=int(np.count_nonzero(a[f'raw{ordinal}_states']!=b[f'raw{ordinal}_states']))
            count_differences+=int(np.count_nonzero(a['counts']!=b['counts']))
    audit={'status':'all_synthetic_program_commands_tail_ranges_public_state_counts_readouts_INTcontrols_and_resources_independently_verified',
        'source_sha256':sha(Path(__file__)),'results_sha256':sha(OUT/'results.json'),'cases':9,
        'public_state_bits_checked':states_checked,'parameter_coordinates_across_formats_checked':values_checked,
        'CPU_GPU_state_bit_differences':state_differences,'CPU_GPU_count_coordinate_differences':count_differences,
        'scope':'Independent range/field/word/readout/resource oracle with raw publicstates on H100 and explicit CPUcomparison. Not a statistical convergence or neural quality result; Complete-model and quality gates remain.'}
    (OUT/'independent_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit))

if __name__=='__main__':main()
