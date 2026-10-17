#!/usr/bin/env python3
"""Why the exact output error of the Prakash-Saha codes is smaller than their published estimate.
For m = 2, the [[14,4,2]]_3 code (stabilizer rows H0, logical rows F, as printed in their paper):
by brute force over all errors of weight <= 2, each output has 32 weight-2 logical errors, whereas the
published estimate counts 2k^2 + 6 = 38 (those with nonzero coordinate sum). The exact
delta_out/delta^2 of codes_lib.Code agrees with the brute-force count, 32/6.
    python3 ps_m2_check.py      # seconds"""
import os, sys, itertools, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from codes_lib import Code
H = np.array([[1,1,0,0,0,0,0,0,0,0,0,2,2,2],
              [0,0,1,1,0,0,0,0,0,0,0,2,2,2],
              [0,0,0,0,1,1,0,0,0,0,0,2,2,2],
              [0,0,0,0,0,0,1,1,0,0,0,2,2,2],
              [1,2,1,2,1,2,1,2,0,1,2,0,1,2],
              [0,0,0,0,0,0,0,0,1,1,1,2,2,2]])
F, H0 = H[:4], H[4:]; n = 14; k = 4
per_output = [0] * k; low = 0; nonzero_sum = 0
for w in (1, 2):
    for supp in itertools.combinations(range(n), w):
        for vals in itertools.product((1, 2), repeat=w):
            v = np.zeros(n, int); v[list(supp)] = vals
            if ((H0 @ v) % 3).any(): continue                     # detected
            hit = [(F[a] @ v) % 3 != 0 for a in range(k)]
            for a in range(k): per_output[a] += int(hit[a])
            low += int(w == 1 and any(hit)); nonzero_sum += int(v.sum() % 3 != 0)
d = 1e-4; ratio = Code('PS m=2', H0, F, 3).step(d)[0] / d ** 2
print(f"weight-2 logical errors per output: {per_output}; weight-1: {low}; "
      f"with nonzero coordinate sum: {nonzero_sum} (published 2k^2+6 = {2 * k * k + 6})")
print(f"exact delta_out/delta^2 at delta = 1e-4 (worst output): {ratio:.4f}; brute force 32/6 = {32 / 6:.4f}")
ok = per_output == [32] * k and low == 0 and nonzero_sum == 2 * k * k + 6 and abs(ratio - 32 / 6) < 0.01
print('OK' if ok else 'MISMATCH'); sys.exit(0 if ok else 1)
