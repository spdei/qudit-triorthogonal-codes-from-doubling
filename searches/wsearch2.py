"""Weighted qutrit P-codes (charges +-1) with dim S = r and n <= maxn, by enumerating multisets of projective
points (used by exhaustive_claims.py). Usage: wsearch2.py r maxn. Prints each improvement and, on the last
line, a dict {d: shortest n found} over d >= 3."""
import numpy as np, itertools, sys
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from qcodes import rank
p=3; r=int(sys.argv[1]); maxn=int(sys.argv[2])
pts=[]
for v in itertools.product(range(p),repeat=r):
    v=np.array(v)
    if v.any() and v[np.nonzero(v)[0][0]]==1: pts.append(v)
pts=np.array(pts); P=len(pts)
iu=np.triu_indices(r); Q=np.array([np.outer(v,v)[iu] for v in pts])
mults=np.array(list(itertools.product(range(p),repeat=P)),dtype=np.int8)
q=(mults.astype(np.int64)@Q)%p; ok=~q.any(1)
supp=(mults>0).sum(1); mass=mults.sum(1)%p
cand=np.where(ok & (mass!=0) & (supp<=maxn))[0]; cand=cand[np.argsort(supp[cand])]
def dD(S,n,cap):
    for w in range(1,cap):
        co=np.array(list(itertools.product(range(1,p),repeat=w)));co=co[co.sum(1)%p!=0]
        for s in itertools.combinations(range(n),w):
            if (~((co@S[:,list(s)].T)%p).any(1)).any(): return w
    return cap
best={}
for idx in cand:
    m=mults[idx]; sp=np.nonzero(m)[0]; n=len(sp)
    if rank(pts[sp],p)<r: continue
    for a in itertools.product([1,2],repeat=n-1):
        a=np.array((1,)+a); cols=(a[:,None]*pts[sp])%p
        if ((m[sp]@cols)%p).any(): continue
        d=dD(cols.T%p,n,7)
        if all(best.get(dd,99)>n for dd in range(3,d+1)):
            for dd in range(3,d+1): best[dd]=min(best.get(dd,99),n)
            print("n",n,"d",d,"w",m[sp].tolist(),"cols",cols.tolist(),flush=True)
print(best)
