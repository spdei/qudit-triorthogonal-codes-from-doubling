#!/usr/bin/env python3
"""Independent check of codes/codes.json (numpy only; shares no code with build_all_codes.py).
For every code: rows independent; triorthogonality with the stored coefficients, or the (weighted)
P-code conditions; the stored witness is a nontrivial Z-logical of the claimed weight (upper bound).
Also checks the recipe by which the paper describes the unprinted [[72,1,6]]_3 and [[118,1,8]]_3."""
import json, itertools, os, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def rank_mod_p(M, p):
    M = np.array(M, dtype=np.int64) % p; r = 0
    for c in range(M.shape[1]):
        piv = next((i for i in range(r, len(M)) if M[i, c]), None)
        if piv is None: continue
        M[[r, piv]] = M[[piv, r]]; M[r] = M[r] * pow(int(M[r, c]), p - 2, p) % p
        for i in range(len(M)):
            if i != r and M[i, c]: M[i] = (M[i] - M[i, c] * M[r]) % p
        r += 1
    return r

codes = json.load(open('codes/codes.json')); ok = 0
for name, c in codes.items():
    p, n = c['p'], c['n']
    L = np.array(c['logical_row']); A = np.array(c['stabilizer_rows']).reshape(-1, n)
    H = np.vstack([L, A]); w = np.ones(n, dtype=np.int64) if c['weights'] is None else np.array(c['weights'])
    assert rank_mod_p(H, p) == len(H), name
    if c['kind'] == 'P':
        assert (L == 1).all() and w.sum() % p and not ((A * w) @ A.T % p).any() and not ((A * w).sum(1) % p).any(), name
    else:
        for a, b, d in itertools.combinations_with_replacement(range(len(H)), 3):
            t = int((w * H[a] * H[b] * H[d]).sum() % p)
            assert (t != 0) if a == b == d == 0 else (t == 0), (name, a, b, d)
    z = np.array(c['distance_witness'])
    assert not (A @ z % p).any() and (L @ z) % p and (z != 0).sum() == c['distance'], name
    ok += 1
    print(f"{name:7s} {c['parameters']:18s} OK   lower bound: {c['distance_lower_bound']}")
print(f'{ok}/{len(codes)} codes verified')

# The paper describes D72_3 and D118_3 by a recipe instead of printing them; check the recipe.
def _recipe(T, S, kap, p=3):
    y = T[0]; n1 = S.shape[1]; n2 = T.shape[1]; one = np.ones(n1, dtype=np.int64); z = np.zeros(n1, dtype=np.int64)
    rows = [np.concatenate([(-kap * one) % p, kap * one, y])]
    rows += [np.concatenate([s, (-s) % p, np.zeros(n2, dtype=np.int64)]) for s in S]
    rows += [np.concatenate([z, kap * one, y])]
    rows += [np.concatenate([z, z, t]) for t in T[1:]]
    return np.array(rows) % p
_M = lambda k: np.vstack([codes[k]['logical_row'], codes[k]['stabilizer_rows']])
H72 = _recipe(_M('D28_3'), np.array(codes['P22_3']['stabilizer_rows']), 1)
H118 = _recipe(H72, np.array(codes['P23_3']['stabilizer_rows']), 2)
assert (H72 == _M('D72_3')).all() and (H118 == _M('D118_3')).all()
print('recipes for [[72,1,6]]_3 and [[118,1,8]]_3 reproduce the stored matrices')
