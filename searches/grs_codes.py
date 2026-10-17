#!/usr/bin/env python3
"""Weighted P-codes on a rational normal curve: the points (1, t, ..., t^(r-1)), t in F_p, with n = 2r + 1
<= p + 1, carry a weighted [[2r+1,1,r+1]]^P_p code, a generalized Reed-Solomon code. Constructs one for
(p, n) = (5, 5), (7, 7) and (11, 11) with the linear-algebra reduction (wp.codes_from_config), and asserts
the weighted P-code conditions and the distance r + 1 by enumerating S^perp.
    python3 grs_codes.py      # about a minute"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..'))
from wp import codes_from_config
from qcodes import check_pcode, nullspace
fails = 0
for p, n in ((5, 5), (7, 7), (11, 11)):
    r = (n - 1) // 2; V = np.array([[pow(t, j, p) for j in range(r)] for t in range(p)][:n], dtype=np.int64)
    res = codes_from_config(V, p, r + 1, np.random.default_rng(0))
    if not res: print(f"p={p} n={n}: no code found  MISMATCH"); fails += 1; continue
    S, w, _ = res[0]; check_pcode(S, p, w)
    G = nullspace(S, p, n); co = np.array(np.meshgrid(*[range(p)] * len(G))).reshape(len(G), -1).T
    V2 = (co @ G) % p; ok = V2.sum(1) % p != 0; d = int((V2[ok] != 0).sum(1).min())
    good = d == r + 1; fails += not good
    print(f"p={p}: weighted [[{n},1,{d}]]^P_{p}, charges {w.tolist()}  {'OK' if good else 'MISMATCH'}", flush=True)
sys.exit(1 if fails else 0)
