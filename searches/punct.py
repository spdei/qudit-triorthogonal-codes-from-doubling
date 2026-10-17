"""Puncturing the triorthogonal space of a code at several coordinates (library for multi_output.py)."""
import sys,os; sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
import json, numpy as np, itertools, sys
from qcodes import rref, nullspace, rank
J=json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','codes','codes.json')))
def unpuncture(d):
    p=d['p']; y=np.array(d['logical_row']); A=np.array(d['stabilizer_rows']).reshape(-1,len(y))
    w=np.array(d['weights']) if d.get('weights') else np.ones(len(y),int)
    lam=int((w*y**3).sum())%p
    H=np.vstack([y,A])%p; col=np.zeros((H.shape[0],1),int); col[0,0]=1
    return p,np.hstack([H,col]),np.concatenate([w,[(-lam)%p]])
def canon(v,p):
    nz=np.nonzero(v)[0]
    if len(nz)==0: return None,None
    s=pow(int(v[nz[0]]),p-2,p); return tuple((v*s)%p),s
def dZ_at_least(H,p,S,target):
    """True if puncturing columns S gives d_Z >= target (target in {2,3})."""
    r,n=H.shape
    B=H[:,S].T%p
    Ann=nullspace(B,p,r) if len(S) else np.eye(r,dtype=int)
    Ann=np.array(Ann).reshape(-1,r)
    rest=[l for l in range(n) if l not in S]
    Q=(Ann@H[:,rest])%p
    groups={}
    for j,l in enumerate(rest):
        q=Q[:,j]
        if not q.any(): return False          # weight-1 logical
        if target>=3:
            key,s=canon(q,p); hn=tuple((H[:,l]*s)%p)
            if key in groups and groups[key]!=hn: return False   # weight-2 logical
            groups.setdefault(key,hn)
    return True
def valid(H,p,S):
    r,n=H.shape; rest=[l for l in range(n) if l not in S]
    return rank(H[:,S].T%p,p)==len(S) and rank(H[:,rest]%p,p)==r
def greedy(H,p,target,trials=300,seed=0):
    rng=np.random.default_rng(seed); r,n=H.shape; best=[]
    for t in range(trials):
        order=rng.permutation(n); S=[]
        for l in order:
            T=S+[int(l)]
            if len(T)>r-1: break
            if valid(H,p,T) and dZ_at_least(H,p,T,target): S=T
        if len(S)>len(best): best=S
    return best
