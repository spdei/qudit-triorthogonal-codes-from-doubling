"""Exact Z-distance of a punctured code (library for multi_output.py)."""
import sys,os; sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
import numpy as np, itertools
from qcodes import nullspace, rank, rref
from punct import unpuncture, J
def solve_coords(B,x,p):
    """coordinates u with B u = x (B: r x k, independent columns) mod p"""
    r,k=B.shape; M=np.hstack([B%p,x.reshape(-1,1)%p]); R,piv=rref(M,p)
    return R[:k,k]%p
def combos(Q,cols,p,w):
    out={}
    for Jset in itertools.combinations(cols,w):
        for a in itertools.product(range(1,p),repeat=w):
            v=tuple((Q[:,Jset]@np.array(a))%p)
            out.setdefault(v,[]).append((Jset,a))
    return out
def dZ(H,p,S,harmless=None,maxw=5):
    """Exact Z-distance (or maxw+1 if none <= maxw) of the code punctured at S.
    A Z-logical is v on the rest with sum v_l h_l = x in span(h_S), x != 0, and the coordinates u of x
    (in the basis h_S) not in span(harmless)."""
    r,n=H.shape; rest=[l for l in range(n) if l not in S]; B=H[:,S]%p
    Ann=np.array(nullspace(B.T,p,r)).reshape(-1,r); Q=(Ann@H)%p
    hb=None
    if harmless is not None and len(harmless):
        hb=np.array(harmless).reshape(-1,len(S))%p
    def is_logical(supp,coef):
        x=(H[:,list(supp)]@np.array(coef))%p
        if not x.any(): return False
        if hb is None: return True
        u=solve_coords(B,x,p)
        return rank(np.vstack([hb,u]),p)>rank(hb,p)
    zero=tuple([0]*Q.shape[0])
    tabs={w:combos(Q,rest,p,w) for w in range(1,min(3,maxw)+1)}
    for w in range(1,maxw+1):
        # split w = w1 + w2, w1>=w2, w1<=3
        for w1 in range(min(3,w),0,-1):
            w2=w-w1
            if w2>w1 or w2>3: continue
            if w2==0:
                for supp,a in tabs[w1].get(zero,[]):
                    if is_logical(supp,a): return w
                continue
            T1,T2=tabs[w1],tabs[w2]
            for key,lst in T1.items():
                neg=tuple((-np.array(key))%p)
                if neg not in T2: continue
                for s1,a1 in lst:
                    for s2,a2 in T2[neg]:
                        if set(s1)&set(s2): continue
                        if w1==w2 and s1>s2: continue
                        if is_logical(s1+s2,a1+a2): return w
    return maxw+1
