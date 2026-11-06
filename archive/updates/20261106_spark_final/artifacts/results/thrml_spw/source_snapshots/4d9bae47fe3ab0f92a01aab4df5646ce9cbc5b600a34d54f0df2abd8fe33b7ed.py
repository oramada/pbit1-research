"""Finite ACT12 control programs; flat128 groups including explicit short tails.

Input point is a BF16-rounded checkpoint parameter exposed as float32. One
logical stochastic cell remains assigned to every scalar, including constants.
This format stores control words and range metadata; it is not one-bit storage.
"""
import numpy as np


def validate(program):
    assert set(program)=={'codes','lower','span','shape','group_size','format_version'}
    c,l,s,shape,size,version=[program[k] for k in ['codes','lower','span','shape','group_size','format_version']]
    assert c.dtype==np.uint16 and c.ndim==1 and c.size>0 and c.max()<=4095
    assert shape.dtype==np.int64 and shape.ndim==1 and np.prod(shape,dtype=np.int64)==c.size
    assert np.all(shape>0) and version==1 and size==128
    assert l.dtype==s.dtype==np.float32 and l.shape==s.shape==((c.size+127)//128,)
    assert np.isfinite(l).all() and np.isfinite(s).all() and np.all(s>=0)
    return program


def encode(weight):
    w=np.asarray(weight)
    if w.dtype!=np.float32 or not w.size or not np.isfinite(w).all():
        raise ValueError('Nonempty finite float32 parameter point required')
    flat=w.reshape(-1);n=flat.size
    lower=np.empty((n+127)//128,np.float32);span=np.empty_like(lower)
    unit=np.empty(n,np.float64)
    for group,begin in enumerate(range(0,n,128)):
        values=flat[begin:begin+128]
        lower[group]=values.min();span[group]=values.max()-lower[group]
        unit[begin:begin+len(values)]=((values.astype(np.float64)-float(lower[group]))/float(span[group]) if span[group] else .5)
    # All cells remain free: finite fields +/-4 realize probabilities strictly
    # between0and1. Constant groups still count/sample their assigned cells.
    low=1/(1+np.exp(8.0));high=1-low
    p=np.clip(unit,low,high)
    h=.5*(np.log(p)-np.log1p(-p))
    codes=np.rint(np.clip((h+4)*(4095/8),0,4095)).astype(np.uint16)
    return validate({'codes':codes,'lower':lower,'span':span,
        'shape':np.asarray(w.shape,np.int64),'group_size':128,'format_version':1})


def fields(codes):
    c=np.asarray(codes)
    if c.dtype!=np.uint16 or not c.size or c.max()>4095:raise ValueError('ACT12 uint16codes required')
    return c.astype(np.float32)*np.float32(8/4095)-np.float32(4)


def probabilities(codes):
    h=fields(codes).astype(np.float64)
    return 1/(1+np.exp(-2*h))


def word_codes(program,bits):
    validate(program)
    if bits not in (4,8):raise ValueError('Internal positional baselines requireK4orK8')
    low=1/(1+np.exp(8.0));high=1-low
    return np.rint(np.clip((probabilities(program['codes'])-low)/(high-low),0,1)*((1<<bits)-1)).astype(np.uint8)


def decode_probability(program,p):
    validate(program);p=np.asarray(p,dtype=np.float64)
    if p.shape!=program['codes'].shape or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('One finite probability/readout in[0,1] per scalar required')
    l=np.repeat(program['lower'],128)[:len(p)].astype(np.float64)
    s=np.repeat(program['span'],128)[:len(p)].astype(np.float64)
    return (l+s*p).reshape(tuple(program['shape']))


def expectation(program,bits=1):
    if bits==1:p=probabilities(program['codes'])
    else:
        words=word_codes(program,bits);low=1/(1+np.exp(8.0))
        p=low+(1-2*low)*words.astype(np.float64)/((1<<bits)-1)
    return decode_probability(program,p)


def integer_control(weight,bits):
    if bits not in (2,4,8):raise ValueError('Integercontrolwidthmustbe2,4,or8')
    p=encode(weight);flat=np.asarray(weight).reshape(-1);l=np.repeat(p['lower'],128)[:flat.size]
    s=np.repeat(p['span'],128)[:flat.size]
    unit=np.divide(flat.astype(np.float64)-l,s,out=np.full(flat.shape,.5),where=s!=0)
    q=np.rint(np.clip(unit,0,1)*((1<<bits)-1))/((1<<bits)-1)
    return decode_probability(p,q)


def resources(program,bits,samples,tile_size):
    validate(program)
    if bits not in (1,4,8) or type(samples)is not int or samples<1 or type(tile_size)is not int or tile_size<1:
        raise ValueError('Valid format,samplesandtilewidthrequired')
    n=program['codes'].size;groups=len(program['lower']);padded=((n+tile_size-1)//tile_size)*tile_size
    return {'unique_parameter_scalars':n,'logical_pbits':bits*n,'range_groups':groups,
        'last_group_actual_size':n-128*(groups-1),'range_metadata_bits':64*groups,
        'finite_control_information_bits':12*n if bits==1 else bits*n,
        'packed_format_storage_bits_excluding_headers':(12*n if bits==1 else bits*n)+64*groups,
        'actual_single_control_array_bytes':int(program['codes'].nbytes),
        'logical_retained_device_evaluations':bits*n*samples,'scheduled_retained_device_evaluations':bits*padded*samples,
        'scheduled_warmup_device_evaluations':bits*padded,'sampling_padding_cells_per_bitplane':padded-n,
        'software_sampler_state_slots':tile_size,'software_int32_count_carry_bytes':4*tile_size,
        'software_retained_count_array_bytes':4*bits*n,'software_decoded_FP64_array_bytes':8*n,
        'regime':'R1 single model weight reconstruction draw reused across evaluation; no R2or physicalstorageclaim'}
