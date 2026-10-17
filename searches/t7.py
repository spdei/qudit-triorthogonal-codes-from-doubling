"""All 7-point configurations in PG(2,7) containing a fixed frame, up to PGL(3,7), searched for weighted
P-codes with distance >= d (used by exhaustive_claims.py p7conic). Usage: t7.py n d."""
import numpy as np, itertools, sys, time
from wp import *
p=7; r=3; rng=np.random.default_rng(0)
P=proj_points(p,r); idx={tuple(v):i for i,v in enumerate(P)}
frame=[idx[(0,0,1)],idx[(0,1,0)],idx[(1,0,0)],idx[(1,1,1)]]
rest=[i for i in range(len(P)) if i not in frame]
n=int(sys.argv[1]); d=int(sys.argv[2]); t0=time.time(); cnt=0
for extra in itertools.combinations(rest,n-4):
    V=P[frame+list(extra)]
    res=codes_from_config(V,p,d,rng)
    cnt+=1
    if res:
        S,w,dist=res[0]; print("FOUND n",n,"d",dist,"w",w.tolist(),"S",S.tolist(),flush=True); break
print("configs",cnt,"time",round(time.time()-t0))
