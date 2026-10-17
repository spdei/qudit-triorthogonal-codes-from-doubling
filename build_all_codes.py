#!/usr/bin/env python3
"""Build, check and export every code of "Short qudit triorthogonal codes from doubling".

For each code this script
  * constructs the generator matrix (row 0 = logical row, remaining rows = stabilizer rows;
    for P-codes the rows are the stabilizers S and the logical row is the all-ones vector),
  * checks triorthogonality (with weights) or the P-code conditions,
  * certifies the distance: an explicit Z-logical of weight d (upper bound, checked directly)
    and a lower bound (exhaustive search, exact enumeration, or the doubling theorem),
  * writes codes/<name>.txt and codes/codes.json.
Any failed check raises an AssertionError (nonzero exit).

Usage:  python3 build_all_codes.py [--out DIR] [--budget N]
  --out DIR   write to DIR instead of codes/ (run_all.sh uses this to compare with the stored files)
  --budget N  maximum number of supports in the exhaustive low-weight searches (default 400000)
Intermediate results are cached in DIR/cache.json, so the script can be re-run incrementally."""
import json, os, sys, time, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from qcodes import *

BUDGET = int(sys.argv[sys.argv.index('--budget') + 1]) if '--budget' in sys.argv else 400000
OUT = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else 'codes'; os.makedirs(OUT, exist_ok=True)
CACHE = os.path.join(OUT, 'cache.json')
cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
def save_cache(): json.dump(cache, open(CACHE, 'w'))

def M(s): return np.array([[int(c) for c in r.split()] for r in s.strip().split('\n')], dtype=np.int64)

# ------------------------------------------------------------------ explicit ingredients
S10_3 = M("""1 0 0 0 2 1 0 1 2 2
0 1 0 0 2 2 1 0 1 2
0 0 1 0 2 2 2 1 0 1
0 0 0 1 2 1 2 2 1 0""")
RM8_3 = M("""1 1 1 1 1 1 1 1
1 2 0 1 2 0 1 2
0 0 1 1 1 2 2 2""")
H19_3 = M("""0 0 0 0 0 2 2 1 1 2 0 2 0 1 0 1 2 2 1
1 0 0 0 0 1 0 1 0 2 2 1 2 2 1 2 2 0 1
0 1 0 0 0 1 2 2 0 1 2 0 1 1 2 0 0 0 2
0 0 1 0 0 2 0 0 0 2 1 1 0 2 1 0 2 2 1
0 0 0 1 0 1 0 2 2 1 2 2 1 2 2 0 1 0 1
0 0 0 0 1 1 2 2 1 0 1 0 2 0 2 1 1 2 2""")

codes = {}   # name -> dict

def add(name, p, H, w=None, kind='tri', construction='', d_claim=None, lower=None, lower_how='',
        witness=None, note=''):
    codes[name] = dict(name=name, p=p, H=np.array(H) % p, w=None if w is None else np.array(w) % p,
                       kind=kind, construction=construction, d=d_claim, lower=lower,
                       lower_how=lower_how, witness=witness, note=note)

def exhaustive_lower(name, H, p, d):
    """Return True if exhaustive search (within budget) proves no logical of weight < d."""
    key = f'{name}:nolog<{d}'
    if key in cache: return cache[key]
    from math import comb
    nsupp = sum(comb(H.shape[1], t) for t in range(1, d))
    if nsupp > BUDGET: return None
    t0 = time.time(); r = no_logical_below(H, p, d); cache[key] = r; save_cache()
    print(f'   exhaustive weight<{d} search for {name}: {r} ({time.time()-t0:.0f}s)', flush=True)
    return r

def pcode(name, p, S, construction, d_claim, exact=True, omega=None):
    S = np.array(S) % p; check_pcode(S, p, omega); n = S.shape[1]
    H = np.vstack([np.ones(n, dtype=np.int64), S])
    key = f'{name}:exact'
    if exact:
        if key not in cache:
            t0 = time.time(); d, wit = pcode_distance_exact(S, p)
            cache[key] = [d, wit.tolist()]; save_cache()
            print(f'   exact P-code distance {name}: {d} ({time.time()-t0:.0f}s)', flush=True)
        d, wit = cache[key]; assert d == d_claim, (name, d)
        add(name, p, H, w=omega, kind='P', construction=construction, d_claim=d, lower=d,
            lower_how='exact enumeration of S^perp', witness=np.array(wit))
    else:
        r = exhaustive_lower(name, H, p, d_claim)
        assert r, name
        wkey = f'{name}:isd'
        if wkey not in cache:
            v = isd_witness(H, p, d_claim, iters=300); cache[wkey] = v.tolist(); save_cache()
        v = np.array(cache[wkey])
        add(name, p, H, kind='P', construction=construction, d_claim=d_claim, lower=d_claim,
            lower_how='exhaustive search (no logical of smaller weight)', witness=v,
            note=('' if (v!=0).sum() == d_claim else
                  f'best upper-bound witness found has weight {(v!=0).sum()}; distance >= {d_claim}'))
        assert is_logical(v, H, p)

def seed(name, p, H, construction, d_claim, companion_ok=False):
    H = np.array(H) % p; check_triorthogonal(H, p)
    r = exhaustive_lower(name, H, p, d_claim)
    if r: how = 'exhaustive search (no logical of smaller weight)'
    else:
        assert r is None and companion_ok, name
        how = 'certified in arXiv:2610.08012 (exhaustive search here exceeds budget)'
    wkey = f'{name}:isd'
    if wkey not in cache:
        t0 = time.time(); v = isd_witness(H, p, d_claim, iters=2000); cache[wkey] = v.tolist(); save_cache()
        print(f'   witness search {name}: weight {(v!=0).sum()} ({time.time()-t0:.0f}s)', flush=True)
    v = np.array(cache[wkey]); assert is_logical(v, H, p) and (v != 0).sum() == d_claim, (name, (v!=0).sum())
    add(name, p, H, construction=construction, d_claim=d_claim, lower=d_claim, lower_how=how, witness=v)

def doubled(name, Pname, Tname, construction):
    P, T = codes[Pname], codes[Tname]; p = P['p']
    H, w, kappa, eta = double(P['H'][1:], T['H'], p, T['w'], P['w'])
    check_triorthogonal(H, p, w)
    if p == 3: H, w = remove_sign_weights(H, w, p); check_triorthogonal(H, p, w)
    n1 = P['H'].shape[1]; n2 = T['H'].shape[1]; d = min(P['d'], T['d'] + 2)
    # explicit witness (as in the proof of the doubling theorem)
    if P['d'] <= T['d'] + 2:
        v = np.concatenate([P['witness'], np.zeros(n1 + n2, dtype=np.int64)])
        if p == 3 and P['w'] is not None:          # follow the column negations of remove_sign_weights
            neg = np.concatenate([P['w'] == 2, P['w'] == 2, np.zeros(n2, bool)]); v[neg] = (-v[neg]) % p
    else:
        zT = T['witness']; yz = int(T['H'][0] @ zT % p)      # scale so that kappa*a + y.z = 0 with a = 1
        zT = zT * ((-kappa) * pow(yz, p - 2, p)) % p
        e = np.zeros(n1, dtype=np.int64); e[0] = 1
        v = np.concatenate([e, e, zT])
        if p == 3 and P['w'] is not None:
            neg = np.concatenate([P['w'] == 2, P['w'] == 2, np.zeros(n2, bool)]); v[neg] = (-v[neg]) % p
    assert is_logical(v, H, p) and (v != 0).sum() == d
    how = f'doubling theorem: min(d_P, d_T + 2) = min({P["d"]}, {T["d"]}+2)'
    r = exhaustive_lower(name, H, p, d)
    if r: how += '; also confirmed by exhaustive search'
    add(name, p, H, w=None if (w == 1).all() else w, construction=construction + f' (kappa={kappa}, eta={eta})',
        d_claim=d, lower=d, lower_how=how, witness=v)

# ------------------------------------------------------------------ the codes
print('P-codes'); 
pcode('P10_3', 3, S10_3, 'explicit (shortest unweighted P-code with d>=3, exhaustive search)', 4)
W7 = np.array([[0,0,1],[0,1,0],[0,1,1],[1,1,1],[1,1,2],[1,2,1],[1,2,2]]).T
pcode('W7_3', 3, W7, 'weighted P-code, weights (2,2,1,2,1,1,2); shortest weighted P-code with d>=3 (exhaustive search)',
      3, omega=np.array([2,2,1,2,1,1,2]))
pcode('P11_3', 3, cyclic_basis(11, 3, [1,3,4,5,9]), 'cyclic/QR, support {1,3,4,5,9}', 5)
pcode('P13_3', 3, cyclic_basis(13, 3, [2,4,5,6,10,12]), 'cyclic, support {2,4,5,6,10,12}', 4)
pcode('P22_3', 3, cyclic_basis(22, 3, [1,2,3,5,6,8,9,10,15,18]), 'cyclic, support {1,2,3,5,6,8,9,10,15,18}', 7)
pcode('P23_3', 3, cyclic_basis(23, 3, [5,7,10,11,14,15,17,19,20,21,22]), 'cyclic/QR, support {5,7,10,11,14,15,17,19,20,21,22}', 8)
pcode('P8_5', 5, cyclic_basis(8, 5, [1,2,5]), 'cyclic, support {1,2,5}', 3)
pcode('W5_5', 5, np.array([[3,4],[2,0],[0,1],[3,1],[2,2]]).T,
      'weighted P-code (generalized Reed-Solomon, 5 points of PG(1,5)), weights (2,2,3,3,3)', 3,
      omega=np.array([2,2,3,3,3]))
pcode('P11_5', 5, cyclic_basis(11, 5, [1,3,4,5,9]), 'cyclic/QR, support {1,3,4,5,9}', 5)
pcode('P19_5', 5, cyclic_basis(19, 5, [1,4,5,6,7,9,11,16,17]), 'cyclic, support {1,4,5,6,7,9,11,16,17}', 7)
pcode('W7_7', 7, np.array([[0,0,2,1,1,1,4],[0,4,0,1,2,3,6],[2,0,0,1,3,2,3]]),
      'weighted P-code (generalized Reed-Solomon: 7 points on the conic 3x0x1+3x0x2+x1x2=0), weights (2,2,2,4,4,1,2); '
      'shortest weighted P-code with d>=4 for p=7', 4, omega=np.array([2,2,2,4,4,1,2]))
pcode('P6_7', 7, cyclic_basis(6, 7, [4,5]), 'cyclic, support {4,5}', 3)
pcode('P9_7', 7, cyclic_basis(9, 7, [1,4,6,7]), 'cyclic, support {1,4,6,7}', 4)
pcode('P18_7', 7, cyclic_basis(18, 7, [2,5,8,11,12,14,15,17]), 'cyclic, support {2,5,8,11,12,14,15,17}', 6, exact=False)

print('seeds')
seed('T8_3', 3, RM8_3, 'punctured Reed-Muller code, explicit matrix', 2)
seed('T19_3', 3, H19_3, 'cyclic construction B, n=20, explicit matrix of arXiv:2610.08012', 3)
seed('T1_3', 3, np.array([[1]]), 'trivial seed', 1)
seed('T11_5', 5, construction_B(12, 5, [2,3,9,10]), 'cyclic construction B, n=12, support {2,3,9,10}', 3)
seed('T6_7', 7, construction_A(6, 7, [1]), 'cyclic construction A, n=6, support {1}', 2)

print('doubled codes')
doubled('D15_3', 'W7_3', 'T1_3', 'doubling of the trivial seed with the weighted P-code W7_3; weights -1 removed by negating columns')
doubled('D28_3', 'P10_3', 'T8_3', 'doubling of T8_3 with P10_3')
doubled('D37_3', 'P11_3', 'D15_3', 'doubling of D15_3 with P11_3')
doubled('D72_3', 'P22_3', 'D28_3', 'doubling of D28_3 with P22_3')
doubled('D118_3', 'P23_3', 'D72_3', 'doubling of D72_3 with P23_3')
doubled('D33_5', 'P11_5', 'T11_5', 'doubling of T11_5 with P11_5')
doubled('D20_7', 'W7_7', 'T6_7', 'doubling of T6_7 with the weighted P-code W7_7 (weighted)')


# ------------------------------------------------------------------ export
def params(c):
    n = c['H'].shape[1]; return f"[[{n},1,{c['d']}]]_{c['p']}" + ('^P' if c['kind'] == 'P' else '')

js = {}
for name, c in codes.items():
    H = c['H']; n = H.shape[1]
    js[name] = dict(parameters=params(c), p=c['p'], n=int(n), kind=c['kind'],
                    construction=c['construction'], distance=c['d'], distance_lower_bound=c['lower_how'],
                    distance_witness=c['witness'].tolist(), weights=None if c['w'] is None else c['w'].tolist(),
                    logical_row=H[0].tolist(), stabilizer_rows=H[1:].tolist(), note=c['note'])
    with open(os.path.join(OUT, name + '.txt'), 'w') as f:
        f.write(f"# {name}: {params(c)}\n# construction: {c['construction']}\n")
        f.write(f"# distance {c['d']}: upper bound = explicit Z-logical below; lower bound = {c['lower_how']}\n")
        if c['note']: f.write(f"# note: {c['note']}\n")
        if c['kind'] == 'P':
            f.write("# P-code: rows below are the stabilizers S; the logical row is the all-ones vector\n")
            rows = H[1:]
        else:
            f.write("# triorthogonal matrix: first row = logical row, then '-', then stabilizer rows\n")
            f.write("# weights: " + ('all 1' if c['w'] is None else ' '.join(map(str, c['w']))) + "\n")
            rows = H
        for i, r in enumerate(rows):
            f.write(' '.join(map(str, r)) + '\n')
            if i == 0 and c['kind'] != 'P': f.write('-\n')
        f.write('# witness: ' + ' '.join(map(str, c['witness'])) + '\n')
json.dump(js, open(os.path.join(OUT, 'codes.json'), 'w'), indent=1)

print('done:', len(codes), 'codes written to', OUT)
