#!/usr/bin/env python3
"""Number A_d of minimum-weight Z-logicals of a doubled code, from its two inputs.

By the proof of the doubling theorem, a logical of the doubled code of weight d = min(d_P, d_T+2) is
  (a) if d = d_P: (z_A, z_B, 0) with v = z_A - z_B a minimum-weight logical of the P-code, z_A and z_B
      of disjoint supports covering supp(v), and sum(z_B) = 0; or
  (b) if d = d_T + 2: (a e_i, b e_j, z_C) with z_C a minimum-weight logical of the seed, b fixed by
      the linking row, and a e_i - b e_j in S^perp (i = j, a = b always qualifies).
Hence A_d = sum_v #{T subset of supp v : sum_T v = 0}  +  A_{d_T}(seed) * M,
with M = #{(a, i, j) : a != 0, a e_i - e_j in S^perp} (independent of b by scaling). Asserts that M = n_P
(no vector a e_i - e_j with i != j lies in S^perp) for every code, that the formula reproduces the A_d
computed directly by distillation/properties.py, and that A_8 = 129030 + 23 * 5060 = 245410 for
[[118,1,8]]_3, with leading output error A_8 / 2^8 = 959 eps^8 to three digits.
    python3 Ad_doubled.py      # about a minute"""
import sys, os, json, itertools, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from qcodes import nullspace
J = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'codes', 'codes.json')))
def span(B, p):
    B = np.array(B) % p
    for c in itertools.product(range(p), repeat=len(B)):
        yield (np.array(c) @ B) % p
def Ad_doubled(seed, A_seed, d_T, pcode):
    P = J[pcode]; p = P['p']; S = np.array(P['stabilizer_rows']) % p; n = S.shape[1]
    Sperp = np.array(nullspace(S, p, n)) % p
    words = list(span(Sperp, p))
    def ok(v): return not ((S @ v) % p).any()
    d_P = min(int(np.count_nonzero(v)) for v in words if v.sum() % p)
    d = min(d_P, d_T + 2); case_a = case_b = 0
    if d == d_P:
        for v in words:
            if np.count_nonzero(v) == d and v.sum() % p:
                supp = np.flatnonzero(v)
                case_a += sum(1 for r in range(d + 1) for T in itertools.combinations(supp, r)
                              if sum(int(v[t]) for t in T) % p == 0)
    M = 0
    if d == d_T + 2:
        for i in range(n):
            for j in range(n):
                for a in range(1, p):
                    u = np.zeros(n, int); u[i] = (u[i] + a) % p; u[j] = (u[j] - 1) % p
                    if ok(u): M += 1          # u = 0 (i = j, a = 1) is allowed
        case_b = A_seed * M
    return d, d_P, case_a, case_b, case_a + case_b, (M if d == d_T + 2 else n)
# (doubled code, seed, A_{d_T} of the seed, d_T, P-code, A_d: computed directly by properties.py, or for
#  [[118,1,8]]_3 the value in the paper)
CASES = [("D15_3", "T1_3", 2, 1, "W7_3", 50), ("D28_3", "T8_3", 8, 2, "P10_3", 230),
         ("D37_3", "D15_3", 50, 3, "P11_3", 2002), ("D72_3", "D28_3", 230, 4, "P22_3", 5060),
         ("D33_5", "T11_5", 92, 3, "P11_5", 2552), ("D20_7", "T6_7", 90, 2, "W7_7", 1194),
         ("D118_3", "D72_3", 5060, 6, "P23_3", 245410)]
if __name__ == "__main__":
    fails = 0
    for name, seed, A_seed, d_T, pc, known in CASES:
        d, d_P, a, b, tot, M = Ad_doubled(seed, A_seed, d_T, pc)
        nP = len(J[pc]['stabilizer_rows'][0])
        ok = tot == known and M == nP and (J[name]['distance'] == d)
        if name == 'D118_3': ok &= a == 129030 and b == 23 * 5060 and round(tot / 2 ** 8) == 959
        fails += not ok
        print(f"{J[name]['parameters']:>16}: d={d} (d_P={d_P}, d_T+2={d_T+2})  (a)={a}  (b)={b}  A_d={tot}  "
              f"M={M} (n_P={nP})  {'OK' if ok else 'MISMATCH'}", flush=True)
    sys.exit(1 if fails else 0)
