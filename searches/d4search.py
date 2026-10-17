"""Exhaustive search for weighted qutrit P-codes with dim S = 4, n columns and distance >= d (used by
exhaustive_claims.py). Usage: d4search.py n d. Backtracks over column sets that contain a basis and avoid
the points on affine lines through the origin; the charges solve a linear system. Last line: done, found N."""
import numpy as np, itertools, sys
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from qcodes import nullspace
p=3; r=4; n=int(sys.argv[1]); dtarget=int(sys.argv[2]) if len(sys.argv)>2 else 4
V=[np.array(v) for v in itertools.product(range(3),repeat=r)]
idx={tuple(v):i for i,v in enumerate(V)}; N=len(V)
neg=[idx[tuple((-v)%3)] for v in V]
# bad[u][v]: bitmask of w forbidden given u,v chosen (w in span(u,v) and not on affine line), plus -u,-v, 0
bad=[[0]*N for _ in range(N)]
for u in range(1,N):
    for v in range(1,N):
        if u==v: continue
        m=0
        for a in range(3):
            for b in range(3):
                w=idx[tuple((a*V[u]+b*V[v])%3)]
                if (a+b)%3!=1: m|=1<<w
        bad[u][v]=m
basis=[idx[tuple(e)] for e in np.eye(r,dtype=int)]
forb0=1|0
for u in basis: forb0|=1<<neg[u]
for u,v in itertools.combinations(basis,2): forb0|=bad[u][v]
for u in basis: forb0|=1<<u
iu=np.triu_indices(r)
found=0
def check(cols):
    global found
    C=np.array([V[c] for c in cols])  # n x r
    # linear in weights: sum w c = 0 (r eqs), sum w c c^T = 0 (10 eqs)
    M=np.vstack([C.T,np.array([np.outer(c,c)[iu] for c in C]).T])%3
    K=nullspace(M,3,len(cols))
    if len(K)==0: return
    for co in itertools.product(range(3),repeat=len(K)):
        w=(np.array(co)@K)%3
        if (w!=0).all() and w.sum()%3:
            S=C.T%3
            # verify distance
            from itertools import combinations, product
            dd=None
            for wt in range(1,8):
                cc=np.array(list(product(range(1,3),repeat=wt)));cc=cc[cc.sum(1)%3!=0]
                hit=False
                for s in combinations(range(len(cols)),wt):
                    if (~((cc@S[:,list(s)].T)%3).any(1)).any(): hit=True;break
                if hit: dd=wt;break
            if dd>=dtarget:
                found+=1
                print("FOUND n",len(cols),"d",dd,"w",w.tolist(),"cols",C.tolist(),flush=True)
                if found>=3: sys.exit()
            return
def bt(chosen,forb,start,k):
    if k==0: check(chosen); return
    for w in range(start,N):
        if (forb>>w)&1: continue
        nf=forb|(1<<w)|(1<<neg[w])
        for u in chosen: nf|=bad[u][w]
        bt(chosen+[w],nf,w+1,k-1)
bt(basis,forb0,1,n-4)
print("done, found",found)
