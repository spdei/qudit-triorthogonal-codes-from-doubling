#!/usr/bin/env python3
"""Randomized (not exhaustive) searches for weighted P-codes beyond the arc bound n <= p+1.

For each case (p, dim S, n, d) below, draws point configurations of n points spanning PG(dim S - 1, p),
either uniformly ('rand') or on the common zero set of k random quadrics ('quad1', 'quad2'), and asks
wp.codes_from_config for a weighted P-code with distance >= d on them (the linear-algebra reduction of the
paper). The cases cover d >= 5 with n <= 10 for p = 5, 7 and d >= 6 with n <= 13 for p = 5; smaller dim S
is impossible since d <= dim S + 1, and larger since n >= 2 dim S + 1. The searches use fixed seeds and
trial counts, so they are reproducible; asserts that none finds a code. As a positive control, the same
search must find the weighted [[7,1,4]]^P_7 code (p = 7, dim S = 3, n = 7, d >= 4) on a single quadric.
    python3 random_searches.py            # TRIALS configurations per case and mode; see README for time
    TRIALS=200 python3 random_searches.py # quick smoke test"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wp import proj_points, qmat, codes_from_config, rank
TRIALS = int(os.environ.get('TRIALS', 5000))
CASES = [(5, 4, 9, 5), (5, 4, 10, 5), (7, 4, 9, 5), (7, 4, 10, 5),
         (5, 5, 11, 6), (5, 5, 12, 6), (5, 5, 13, 6), (5, 6, 13, 6)]

def configs(p, r, n, mode, rng):
    P = proj_points(p, r); Q = qmat(P, p).T; tried = 0; attempts = 0
    while tried < TRIALS and attempts < 50 * TRIALS:
        attempts += 1
        if mode == 'rand': sel = rng.choice(len(P), n, replace=False)
        else:
            F = rng.integers(0, p, (Q.shape[1], int(mode[-1])))
            on = np.where(~((Q @ F) % p).any(1))[0]
            if len(on) < n: continue
            sel = rng.choice(on, n, replace=False)
        V = P[sel]
        if rank(V, p) < r: continue
        tried += 1; yield V

rng = np.random.default_rng(0)
control = next((c for _, V in zip(range(TRIALS), configs(7, 3, 7, 'quad1', rng)) for c in codes_from_config(V, 7, 4, rng)), None)
print(f"positive control (p=7, dim S=3, n=7, d>=4): {'found' if control else 'NOT found'}", flush=True)
found = 0 if control else 1
for case in CASES:
    p, r, n, d = case
    for seed, mode in enumerate(('rand', 'quad1', 'quad2')):
        rng = np.random.default_rng(seed); cnt = 0
        for V in configs(p, r, n, mode, rng):
            cnt += 1; res = codes_from_config(V, p, d, rng)
            if res:
                S, w, dist = res[0]; found += 1
                print(f"FOUND p={p} n={n} d={dist} w={w.tolist()} S={S.tolist()}", flush=True); break
        print(f"p={p} dim S={r} n={n} d>={d} mode={mode}: {cnt} configurations", flush=True)
print('none found' if not found else f'{found} FOUND (contradicts the paper)')
sys.exit(1 if found else 0)
