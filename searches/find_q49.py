#!/usr/bin/env python3
"""The qubit [[49,1,5]] code used in the qubit comparison: doubling of the 15-qubit code with a [[17,1,5]]
input S17 (self-orthogonal, even weights, length 17; its dual has no odd-weight vector of weight 1 or 3).
A random search (seed 0, 4000 trials) finds 30 such inputs; S17 is the one whose dual has the fewest
odd-weight vectors of weight 5. Asserts that the search reproduces the stored S17, that the stored
[[49,1,5]] matrix is the doubled matrix with rows (1,1,1), (s,s,0), (0,1,1), (0,0,t), that it is
triorthogonal (all pairwise and triple overlaps even, logical row of odd weight), and that it has
distance 5 with 1411 minimum-weight logicals.
    python3 find_q49.py      # about a minute"""
import numpy as np, itertools, sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'distillation'))
from qcodes import nullspace, rref
from codes_lib import span_wd, dual_wd
rng = np.random.default_rng(0); n = 17
odd_low = [np.eye(n, dtype=int)[list(c)].sum(0) for w in (1, 3) for c in itertools.combinations(range(n), w)]
def rank2(M): return len(rref(np.array(M) % 2, 2)[1]) if len(M) else 0
def random_S():
    S = []
    while len(S) < 8:
        v = rng.integers(0, 2, n)
        if v.sum() % 2: continue
        if any((v @ s) % 2 for s in S): continue
        if rank2(S + [v]) == len(S) + 1: S.append(v)
    return np.array(S)
best = None; found = 0
for t in range(4000):
    S = random_S()
    if any(not ((S @ v) % 2).any() for v in odd_low): continue      # odd dual vector of weight 1 or 3
    found += 1
    C = np.array(nullspace(S, 2, n)) % 2
    words = np.array([(np.array(c) @ C) % 2 for c in itertools.product(range(2), repeat=len(C))])
    A5 = int((words.sum(1) == 5).sum())
    if best is None or A5 < best[0]: best = (A5, S)
print(f"found {found} of 4000; fewest odd-weight-5 dual vectors: {best[0]}")

q = json.load(open(os.path.join(HERE, '..', 'codes', 'Q49_2.json')))['Q49_2']
S17 = np.array(q['S17']); y = np.array(q['logical_row']); A = np.array(q['stabilizer_rows']); H = np.vstack([y, A])
ok = found == 30 and np.array_equal(best[1], S17)
z17, z15 = np.zeros(17, int), np.zeros(15, int); o17 = np.ones(17, int); y15 = y[34:]
ok &= (y == 1).all() and all(np.array_equal(A[i], np.concatenate([S17[i], S17[i], z15])) for i in range(8))
ok &= np.array_equal(A[8], np.concatenate([z17, o17, y15])) and not A[9:, :34].any()
tri = all((H[a] * H[b]).sum() % 2 == 0 for a, b in itertools.combinations(range(len(H)), 2)) and \
      all((H[a] * H[b] * H[c]).sum() % 2 == 0 for a, b, c in itertools.combinations(range(len(H)), 3)) and \
      y.sum() % 2 == 1
L = [int(a - b) for a, b in zip(dual_wd(span_wd(A, 2, 49), 49, 2), dual_wd(span_wd(H, 2, 49), 49, 2))]
d = next(w for w, x in enumerate(L) if x)
print(f"stored S17 reproduced and doubling structure: {ok}; triorthogonal: {tri}; d = {d}, A_d = {L[d]}")
ok &= tri and d == 5 and L[5] == 1411 == q['A_d']
print('OK' if ok else 'MISMATCH'); sys.exit(0 if ok else 1)
