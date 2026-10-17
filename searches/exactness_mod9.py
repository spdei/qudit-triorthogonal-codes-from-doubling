#!/usr/bin/env python3
"""Which qutrit codes implement their logical gate exactly, with no Pauli correction.
F(x) = sum_l c_l x_l^3 mod 9 is well defined on residues mod 3, and on the stabilizer space of a
triorthogonal matrix it is additive with values in 3Z/9Z; no correction is needed iff F vanishes on the
stabilizer rows. F(logical row) mod 9 is the exact logical power of U. Asserts the statements of the
paper: [[15,1,3]]_3 (U) and [[28,1,4]]_3 (U^-1 = U^8) are exact, as are the seeds [[8,1,2]]_3 and
[[19,1,3]]_3; [[37,1,5]]_3, [[72,1,6]]_3 and [[118,1,8]]_3 need a correction.
    python3 exactness_mod9.py      # seconds"""
import json, os, sys, numpy as np
J = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'codes', 'codes.json')))
EXPECTED = {'T8_3': True, 'T19_3': True, 'D15_3': True, 'D28_3': True,
            'D37_3': False, 'D72_3': False, 'D118_3': False}     # exact?
POWER = {'D15_3': 1, 'D28_3': 8}                                   # logical gate U^power
fails = 0
for name in EXPECTED:
    d = J[name]; y = np.array(d['logical_row']); A = np.array(d['stabilizer_rows']).reshape(-1, len(y))
    w = np.array(d['weights']) if d.get('weights') else np.ones(len(y), int)
    F = lambda x: int((w * ((x % 3) ** 3)).sum()) % 9
    fs = sorted(set(F(s) for s in A)); exact = all(v == 0 for v in fs)
    ok = exact == EXPECTED[name] and (name not in POWER or F(y) == POWER[name]) and all(v % 3 == 0 for v in fs)
    fails += not ok
    print(f"{d['parameters']:>14}  exact={'yes' if exact else 'no '}  logical gate U^{F(y)}  "
          f"F(stabilizers)={fs}  {'OK' if ok else 'MISMATCH'}")
sys.exit(1 if fails else 0)
