"""Lossless original-NPY-byte archival, including headers and floating bit patterns."""
import io,json,struct,hashlib,zlib
import numpy as np
MAGIC=b'SPWNPY1\n'
def sha_bytes(data):return hashlib.sha256(data).hexdigest()
def header_info(raw):
    stream=io.BytesIO(raw);version=np.lib.format.read_magic(stream)
    if version==(1,0):shape,order,dtype=np.lib.format.read_array_header_1_0(stream)
    elif version==(2,0):shape,order,dtype=np.lib.format.read_array_header_2_0(stream)
    else:raise ValueError('Unsupported NPY version')
    offset=stream.tell()
    if dtype.kind!='f' or dtype.itemsize not in [4,8]:raise ValueError('Only dense F32/F64 evidence arrays supported')
    if offset+int(np.prod(shape))*dtype.itemsize!=len(raw):raise ValueError('Unexpected NPY length or trailing data')
    return offset,dtype.itemsize
def encode(raw,original_path,mode,mtime_ns):
    offset,itemsize=header_info(raw);metadata={'format':'spw-original-npy-byteplanes-v1','original_path':original_path,'original_bytes':len(raw),'original_sha256':sha_bytes(raw),'header_hex':raw[:offset].hex(),'itemsize':itemsize,'mode':int(mode),'mtime_ns':int(mtime_ns)};meta=json.dumps(metadata,sort_keys=True,separators=(',',':')).encode();planes=np.frombuffer(raw,np.uint8,offset=offset).reshape(-1,itemsize).T.copy().tobytes();return MAGIC+struct.pack('<Q',len(meta))+meta+zlib.compress(planes,6),metadata
def decode(blob):
    if blob[:8]!=MAGIC or len(blob)<16:raise ValueError('Invalid archive magic')
    length=struct.unpack('<Q',blob[8:16])[0]
    if length>65536 or len(blob)<16+length:raise ValueError('Invalid archive metadata length')
    m=json.loads(blob[16:16+length]);header=bytes.fromhex(m['header_hex']);size=m['itemsize'];n=m['original_bytes']-len(header)
    if m['format']!='spw-original-npy-byteplanes-v1' or size not in [4,8] or not 0<=n<=2**31 or n%size:raise ValueError('Invalid archive dimensions')
    dec=zlib.decompressobj();planes=dec.decompress(blob[16+length:],n+1)
    if len(planes)!=n or not dec.eof or dec.unused_data or dec.unconsumed_tail:raise ValueError('Invalid compressed payload')
    raw=header+np.frombuffer(planes,np.uint8).reshape(size,-1).T.copy().tobytes()
    if sha_bytes(raw)!=m['original_sha256']:raise ValueError('Restored NPY checksum mismatch')
    header_info(raw)
    return raw,m
