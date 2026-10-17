"""Unweighted qutrit P-codes with d >= 3 (used by exhaustive_claims.py unweighted10). Usage: small3.py p dim_S.
Enumerates multisets of projective points (multiplicity < p) with vanishing first and second moments and
length != 0 mod p, and prints the shortest configurations found (up to three lengths)."""
import numpy as np, itertools, sys
p=int(sys.argv[1]); r=int(sys.argv[2])
# projective points
pts=[]
for v in itertools.product(range(p),repeat=r):
    v=np.array(v)
    if not v.any(): continue
    first=v[np.nonzero(v)[0][0]]
    if first==1: pts.append(v)
pts=np.array(pts); P=len(pts); print("points",P)
iu=np.triu_indices(r)
Q=np.array([np.outer(v,v)[iu] for v in pts])  # P x r(r+1)/2
best=None
# enumerate multiplicities 0..p-1 (vector of length P) iteratively by chunks
mults=np.array(list(itertools.product(range(p),repeat=P)),dtype=np.int8) if p**P<5e6 else None
assert mults is not None
q=(mults.astype(np.int64)@Q)%p
okq=~q.any(1)
n1=mults.sum(1)
cand=np.where(okq & (n1%p!=0))[0]
cand=cand[np.argsort(n1[cand])]
reps=[np.array(c) for c in itertools.product(range(1,p),repeat=1)]
found={}
for idx in cand:
    m=mults[idx]; supp=np.nonzero(m)[0]
    # need rank r of chosen points
    if np.linalg.matrix_rank(pts[supp].astype(float))<r: continue  # rank over F_p <= rank over R, so this skips only rank-deficient sets
    # choose scalars a_P in F_p^* for each supported point: sum m_P a_P v_P = 0 and quad sum m a^2 vv^T=0
    for a in itertools.product(range(1,p),repeat=len(supp)):
        a=np.array(a)
        lin=(m[supp]*a)@pts[supp]%p
        quad=((m[supp]*a*a)@Q[supp])%p
        if not lin.any() and not quad.any():
            nn=int(n1[idx])
            if nn not in found:
                found[nn]=(m.copy(),supp,a)
                print("n1=",nn,"mults",m[supp],"pts",pts[supp].tolist(),"scal",a.tolist(),flush=True)
            break
    if len(found)>=3: break
