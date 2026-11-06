"""Independent byte-wise archive decoder; no codec import or float conversion."""
import json,struct,zlib,hashlib
from pathlib import Path
def independently_restore(archive):
    data=Path(archive).read_bytes()
    if data[:8]!=b'SPWNPY1\n' or len(data)<16:raise ValueError('Invalid magic')
    length=struct.unpack('<Q',data[8:16])[0]
    if length>65536 or len(data)<16+length:raise ValueError('Invalid metadata')
    m=json.loads(data[16:16+length]);header=bytes.fromhex(m['header_hex']);width=m['itemsize'];n=m['original_bytes']-len(header)
    if width not in (4,8) or not 0<=n<=2**31 or n%width:raise ValueError('Invalid payload width')
    decoder=zlib.decompressobj();planes=decoder.decompress(data[16+length:],n+1)
    if len(planes)!=n or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:raise ValueError('Incomplete or oversized payload')
    payload=bytearray(n);words=n//width
    for byte in range(width):payload[byte::width]=planes[byte*words:(byte+1)*words]
    raw=header+payload
    if hashlib.sha256(raw).hexdigest()!=m['original_sha256']:raise ValueError('Original-byte hash mismatch')
    return bytes(raw),m
