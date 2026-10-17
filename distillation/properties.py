#!/usr/bin/env python3
"""Properties of the seeds and doubled codes (the code-properties table of the paper), computed exactly.

For each code: r (number of stabilizer rows), the logical exponent lambda (mod p, as +-1), the number A_d
of minimum-weight Z-logicals (counted as vectors), the leading output error A_d/(p-1)^d eps^d, the
threshold eps* (smallest eps with eps_out >= eps), and the acceptance probability at eps = 0.01.
All are computed from weight enumerators via the MacWilliams identity (codes_lib.py). Each value is
asserted against the paper, to the digits printed there. [[118,1,8]]_3 is out of reach this way; its
A_8 = 245410 is checked by searches/Ad_doubled.py.
    python3 properties.py            # about 1 minute; needs gcc"""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from codes_lib import Code, threshold
J = json.load(open(os.path.join(HERE, '..', 'codes', 'codes.json')))
J.update(json.load(open(os.path.join(HERE, '..', 'codes', 'C40_3.json'))))

# name: (r, lambda, A_d, leading coefficient, eps*, P_acc(0.01)), as printed in the paper
EXPECTED = {
    'T8_3':  (2, -1, 8, '2', '0.211', '0.92'),
    'T19_3': (5, -1, 42, '5.25', '0.226', '0.83'),
    'D15_3': (4, 1, 50, '6.25', '0.230', '0.86'),
    'D28_3': (7, -1, 230, '14.4', '0.212', '0.75'),
    'D37_3': (10, 1, 2002, '62.6', '0.230', '0.69'),
    'D72_3': (18, -1, 5060, '79.1', '0.217', '0.48'),
    'C40_3': (10, 1, 260, '16.25', '0.223', '0.67'),
    'T11_5': (3, -1, 92, '1.44', '0.346', '0.90'),
    'D33_5': (9, -1, 2552, '2.49', '0.334', '0.72'),
    'T6_7':  (1, -1, 90, '2.5', '0.232', '0.94'),
    'D20_7': (5, -1, 1194, '0.921', '0.376', '0.82'),
}

def close(x, s):
    """x agrees with the decimal string s to the digits shown."""
    dec = len(s.split('.')[1]) if '.' in s else 0
    return abs(x - float(s)) <= 0.5 * 10 ** -dec + 1e-12

fails = 0
for nm, (r_, lam_, Ad_, coef_, thr_, acc_) in EXPECTED.items():
    c = J[nm]; p = c['p']; y = np.array(c['logical_row']); A = np.array(c['stabilizer_rows']).reshape(-1, len(y))
    w = np.ones(len(y), dtype=np.int64) if c.get('weights') is None else np.array(c['weights'])
    lam = int((w * y ** 3).sum() % p); lam = lam - p if lam > p // 2 else lam
    C = Code(nm, A, y[None, :], p); d, Ad = C.A_d(); coef = Ad / (p - 1) ** d
    thr = threshold(lambda e: C.eps_stats(e)[0], hi=(p - 1) / p - 1e-9); acc = C.eps_stats(0.01)[1]
    ok = (len(A) == r_ and lam == lam_ and d == c['distance'] and Ad == Ad_ and close(coef, coef_)
          and close(thr, thr_) and close(acc, acc_))
    fails += not ok
    print(f"{c['parameters']:>14}  r={len(A):2d}  lambda={lam:+d}  A_d={Ad:5d}  eps_out={coef:.4g} eps^{d}  "
          f"eps*={thr:.4f}  P_acc(0.01)={acc:.4f}  {'OK' if ok else 'MISMATCH'}", flush=True)
print('all values agree with the paper' if not fails else f'{fails} MISMATCH(ES)')
sys.exit(1 if fails else 0)
