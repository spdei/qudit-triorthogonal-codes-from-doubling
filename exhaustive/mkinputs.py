#!/usr/bin/env python3
"""Write known codes in the systematic form A=(I_r|P), h=(0|q) used by triorth.c,
for validating the -d check. Format: 'p r m d name' / q row / r rows of P."""
import json, numpy as np, itertools
import sys, os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.join(HERE,'..'))
from qcodes import rref, construction_A, concatenate, no_logical_below
J=json.load(open(os.path.join(HERE,'..','codes','codes.json')))
def systematic(y,A,p):
    A=np.array(A)%p; y=np.array(y)%p; R,piv=rref(A,p); r=len(piv)
    order=list(piv)+[c for c in range(A.shape[1]) if c not in piv]
    R=R[:r][:,order]; y=y[order]
    y=(y-sum(y[i]*R[i] for i in range(r)))%p
    assert not R[:,:r].tolist()!=np.eye(r,dtype=int).tolist()
    return r,R[:,r:],y[r:]
def dist(y,A,p,maxd=8):
    H=np.vstack([y,A])%p
    for w in range(1,maxd+1):
        if not no_logical_below(H,p,w+1): return w
    return None
out=[]
for nm,d in [("D15_3",3),("T19_3",3),("D28_3",4),("D37_3",5),("T11_5",3),("D33_5",5),("D20_7",4)]:
    c=J[nm]; y=c['logical_row']; A=np.array(c['stabilizer_rows']).reshape(-1,len(y))
    out.append((nm,c['p'],d,*systematic(y,A,c['p'])))
# [[16,1,4]]_5 as the concatenation of the [[4,1,2]]_5 Reed-Solomon code with itself
H4=construction_A(4,5,[1])
try:
    H16=concatenate(H4,H4,5)
    H16=np.array(H16[0] if isinstance(H16,tuple) else H16)%5
    out.append(("C16_5",5,4,*systematic(H16[0],H16[1:],5)))
except Exception as e: print("concat failed:",e)
with open(os.path.join(HERE,"known_codes.txt"),"w") as f:
    for nm,p,d,r,P,q in out:
        f.write(f"{p} {r} {P.shape[1]} {d} {nm}\n"+" ".join(map(str,q))+"\n")
        for row in P: f.write(" ".join(map(str,row))+"\n")
        print(nm,"p=",p,"r=",r,"m=",P.shape[1],"d(claimed)=",d)
