#!/usr/bin/env python3
"""Search of all cyclic P-codes of length n <= N over F_p (n != 0 mod p), by the spectral criterion of the
paper: S is a P-code iff 0 is not in Sigma or 2 Sigma. Only maximal supports are tested, since enlarging
S can only raise the distance, and supports equivalent under multipliers u (gcd(u, n) = 1) are tested
once. For each n prints: n, number of inequivalent maximal supports, best distance (maxw + 1 means
> maxw), and a best support. Asserts that the best distance is at least that of the cyclic P-codes in the
paper's table of P-codes (capped at maxw + 1). The ranges searched in the paper:
    python3 dsearch.py 3 50 8;  python3 dsearch.py 5 39 7;  python3 dsearch.py 7 30 6   # long; see README"""
import numpy as np, itertools, sys
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from qcodes import cosets, cyclic_basis
def dP_cyclic(S_basis,n,p,maxw,cyclic=True):
    """d_P: min weight of v perp S with sum v != 0 (or None if > maxw). Cyclic: may assume 0 in supp(v)."""
    G=S_basis
    for w in range(1,maxw+1):
        coeff=np.array(list(itertools.product(range(1,p),repeat=w)),dtype=np.int64)
        good=coeff.sum(1)%p!=0; coeff=coeff[good]
        rng=itertools.combinations(range(1,n),w-1) if cyclic else itertools.combinations(range(n),w)
        for rest in rng:
            s=[0]+list(rest) if cyclic else list(rest)
            A=(coeff@G[:,s].T)%p if len(G) else np.zeros((len(coeff),1),int)
            if (~A.any(1)).any(): return w
    return None  # > maxw
def maximal_D_supports(n,p):
    cs=[c for c in cosets(n,p) if c[0]!=0]
    res=[]
    def ok(Sset):
        for a in Sset:
            if (-a)%n in Sset: return False   # 0 in 2S (incl a+a=0 impossible for odd n... n may be even)
        return True
    def dfs(i,cur):
        ext=False
        for j in range(i,len(cs)):
            new=cur|set(cs[j])
            if ok(new): dfs(j+1,new); ext=True
        # maximality check
        for c in cs:
            if not set(c)<=cur and ok(cur|set(c)): return
        if cur: res.append(frozenset(cur))
    dfs(0,set())
    return list(set(res))
if __name__=="__main__":
    p=int(sys.argv[1]); N=int(sys.argv[2]); maxw=int(sys.argv[3])
    TABLE={3:{11:5,13:4,22:7,23:8},5:{8:3,11:5,19:7},7:{6:3,9:4,18:6}}[p]   # cyclic codes of the table
    fails=0
    for n in range(3,N+1):
        if n%p==0: continue
        sups=maximal_D_supports(n,p)
        seen=set(); bestn=0
        for S in sups:
            key=min(tuple(sorted((u*s)%n for s in S)) for u in range(1,n) if np.gcd(u,n)==1)
            if key in seen: continue
            seen.add(key)
            B=cyclic_basis(n,p,sorted(S))
            d=dP_cyclic(B,n,p,maxw)
            d=d if d else maxw+1
            if d>bestn: bestn=d; bS=sorted(S)
        bad=n in TABLE and bestn<min(TABLE[n],maxw+1); fails+=bad
        print(n,len(seen),bestn,bS if bestn>=3 else "","MISMATCH" if bad else "",flush=True)
    print("all table entries reached" if not fails else f"{fails} MISMATCH(ES)"); sys.exit(1 if fails else 0)
