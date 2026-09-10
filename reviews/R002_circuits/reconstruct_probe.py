#!/usr/bin/env python3
"""Reconstruct R001/R002's exact X64-disabled fixed probe with NumPy.

Threefry 2x32 and legacy split semantics follow JAX 0.4.33:
https://github.com/jax-ml/jax/blob/jax-v0.4.33/jax/_src/prng.py
https://github.com/jax-ml/jax/blob/jax-v0.4.33/jax/_src/random.py
Upstream implementation is Apache-2.0 licensed. This restricted implementation
supports the even-length uint32 counter and randint(0,2) case used by the probe.
The generated NPZ must match the previously recorded complete-file SHA-256.
"""
import hashlib
from pathlib import Path
import numpy as np

EXPECTED = "d49ecf23fd932fba25b1068f8c90820265a05fd90daadc372089520889090b2f"

def threefry(key, count):
    count=np.asarray(count,dtype=np.uint32)
    assert count.ndim==1 and count.size%2==0
    key=np.asarray(key,dtype=np.uint32)
    k=[key[0],key[1],key[0]^key[1]^np.uint32(0x1BD11BDA)]
    a,b=np.split(count.copy(),2)
    rotations=((13,15,26,6),(17,29,16,24))
    with np.errstate(over="ignore"):
        a+=k[0];b+=k[1]
        for s in range(1,6):
            for r in rotations[(s-1)%2]:
                a+=b
                b=((b<<np.uint32(r)) | (b>>np.uint32(32-r))) ^ a
            a+=k[s%3]
            b+=k[(s+1)%3]
            b+=np.uint32(s)
    return np.concatenate([a,b])

def split(key):
    return threefry(key,np.arange(4,dtype=np.uint32)).reshape(2,2)

def reconstruct(path):
    root=np.array([0,1000023],dtype=np.uint32)
    _,sample_key=split(root)
    _,lower_key=split(sample_key)
    # JAX int32 randint modulo 2: its high-word multiplier 2**32 mod 2 is zero.
    shape=(32,16,16,8)
    inputs=(threefry(lower_key,np.arange(np.prod(shape),dtype=np.uint32)) & 1).reshape(shape).astype(np.float32)
    y,x=np.indices((16,16))
    target=np.zeros(shape,np.float32)
    target[...,0]=((x//2+y//2)%2)
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,inputs=inputs,target=target)
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual==EXPECTED,(actual,EXPECTED)
    return {"path":str(path),"bytes":path.stat().st_size,"sha256":actual,
            "reconstructed":True,"matches_original_archive_bytes":True}

if __name__=="__main__":
    import json
    path=Path(__file__).resolve().parents[2]/"artifacts"/"reconstructed_fixed_evaluation_set.npz"
    print(json.dumps(reconstruct(path),indent=2))
