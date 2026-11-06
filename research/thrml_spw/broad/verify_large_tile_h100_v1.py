"""Independent packed-byte popcount and scalar rational readout oracle."""
from pathlib import Path
import hashlib,json,math
import numpy as np

OUT=Path('results/thrml_spw/broad_verification_20261105/large_tile_h100_v1')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    r=json.loads((OUT/'results.json').read_text());assert r['backend']=='gpu'
    assert [v['bits'] for v in r['cases']]==[1,4,8]
    table=np.asarray([bin(i).count('1') for i in range(256)],np.uint8)
    checked=0;coordinates=0
    for case in r['cases']:
        assert sha(case['path'])==case['sha256'];k=case['bits'];t=case['samples'];assert k*t==8192
        assert case['readout']=='exactint64positionalnumerator_then_FP64divide_mul_add'
        with np.load(case['path'],allow_pickle=False) as a:
            w=a['weight'];c=a['codes'];lower=a['lower'];span=a['span'];n=len(w)
            assert n==65535 and c.shape==(n,) and c.dtype==np.uint16 and c.max()<=4095
            assert lower.dtype==span.dtype==np.float32 and lower.shape==span.shape==(512,)
            starts=np.arange(0,n,128);expected_low=np.minimum.reduceat(w,starts)
            np.testing.assert_array_equal(lower,expected_low)
            np.testing.assert_array_equal(span,np.maximum.reduceat(w,starts)-expected_low)
            assert int(a['group_size'])==128 and int(a['format_version'])==1
            h=c.astype(np.float32)*np.float32(8/4095)-np.float32(4)
            p=1/(1+np.exp(-2*h.astype(np.float64)));low=1/(1+math.exp(8));words=np.rint((p-low)/(1-2*low)*((1<<k)-1)).astype(np.int64) if k>1 else None
            counts=np.zeros((k,n),np.int64)
            assert len(case['raw_tiles'])==k
            for ordinal,record in enumerate(case['raw_tiles']):
                assert record=={'tile':0,'begin':0,'width':n,'bit':ordinal}
                z=a[f'raw{ordinal}_states'];actual=a[f'raw{ordinal}_padded_counts'];fields=a[f'raw{ordinal}_fields']
                assert z.dtype==np.bool_ and z.shape==(t,65536)
                expected_h=h if k==1 else np.where((words>>ordinal)&1,4,-4).astype(np.float32)
                np.testing.assert_array_equal(fields,np.pad(expected_h,(0,1)))
                # Byte packing plus independent256entry population-count table
                # checks the complete public trace using a different reduction.
                packed=np.packbits(z,axis=0,bitorder='little')
                independent=table[packed].sum(axis=0,dtype=np.uint64)
                np.testing.assert_array_equal(actual,independent)
                counts[ordinal]=independent[:n];checked+=z.size
                del z,packed
            np.testing.assert_array_equal(a['counts'],counts)
            combined=[]
            for i in range(n):
                numerator=sum(int(counts[b,i])*(2**b) for b in range(k))
                combined.append(float(lower[i//128])+float(span[i//128])*(numerator/(t*((2**k)-1))))
            np.testing.assert_array_equal(a['decoded'],np.asarray(combined,np.float64))
            cost=case['resources']
            assert cost['logical_pbits']==k*n and cost['scheduled_retained_device_evaluations']==65536*8192
            assert cost['logical_retained_device_evaluations']==n*8192 and cost['scheduled_warmup_device_evaluations']==k*65536
            assert cost['sampling_padding_cells_per_bitplane']==1 and cost['last_group_actual_size']==127
            assert cost['range_groups']==512 and cost['range_metadata_bits']==512*64
            assert cost['software_retained_count_array_bytes']==4*k*n and cost['software_int32_count_carry_bytes']==4*65536
            coordinates+=n
    audit={'status':'full65536lane_T8192matchedwork_states_popcounts_rational_readouts_and_resources_independently_verified',
        'source_sha256':sha(Path(__file__)),'results_sha256':sha(OUT/'results.json'),'cases':3,
        'public_state_bits_checked':checked,'parameter_coordinates_across_formats_checked':coordinates,
        'scope':'Entire H100 publicstate traces checked by packed-bytepopcount, complete rational-count readout checkedscalarby scalar. No real model, task quality, physical energy or hardwarelatency result.'}
    (OUT/'independent_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit))

if __name__=='__main__':main()
