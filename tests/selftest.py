#!/usr/bin/env python3
"""Self-tests of the shared code, independent of the paper's results.

Each routine is compared with a deliberately naive reimplementation on small inputs (random ones with fixed
seeds, and small known codes):
  1. linear algebra over F_p (rref, rank, nullspace)
  2. cyclic codes: shift invariance, dimension, and the spectral support of the dual
  3. the triorthogonality and P-code checks against a direct evaluation of the trilinear form
  4. the doubling construction: triorthogonal, same lambda, distance min(d_P, d_T + 2)
  5. the distance routines (low-weight search, P-code distance, witness search, punctured distance)
  6. weight distributions (numpy and wd3.c) and the MacWilliams identity against full enumeration
  7. distillation statistics against a sum over all error patterns
  8. the sequence optimizer against exhaustive enumeration of all short sequences
    python3 tests/selftest.py      # about a minute; needs gcc"""
import os, sys, itertools, numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
for d in ('', 'distillation', 'searches'): sys.path.insert(0, os.path.join(ROOT, d))
from qcodes import (rref, rank, nullspace, cosets, cyclic_basis, check_triorthogonal, check_pcode, double,
                    remove_sign_weights, construction_A, construction_B, concatenate, is_logical,
                    no_logical_below, pcode_distance_exact, isd_witness)
from codes_lib import span_wd, dual_wd, Code, _wd3
from dz import dZ
from rounds import Fn, optimize

rng = np.random.default_rng(12345); fails = 0
def check(label, ok):
    global fails; fails += not ok; print(f"  {'ok  ' if ok else 'FAIL'} {label}", flush=True)

# ---------------------------------------------------------------- naive references
def span(G, p):
    """All vectors of the row space of G (small dimensions only)."""
    G = np.array(G, dtype=np.int64) % p
    if len(G) == 0: return np.zeros((1, G.shape[1]), dtype=np.int64)
    return np.array([(np.array(c) @ G) % p for c in itertools.product(range(p), repeat=len(G))])

def naive_rank(G, p):
    """log_p of the number of distinct vectors in the row space."""
    return round(np.log(len({tuple(v) for v in span(G, p)})) / np.log(p))

def naive_dual(G, p, n):
    """All x in F_p^n orthogonal to every row of G (n small)."""
    X = np.array(list(itertools.product(range(p), repeat=n)), dtype=np.int64)
    G = np.array(G, dtype=np.int64).reshape(-1, n)
    return X[~((X @ G.T) % p).any(1)] if len(G) else X

def tau_all_zero(H, p, w):
    """Direct evaluation of tau on every triple of rows; returns (ok as triorthogonal, lambda)."""
    m = len(H)
    for a, b, c in itertools.product(range(m), repeat=3):
        t = int(sum(int(w[l]) * int(H[a][l]) * int(H[b][l]) * int(H[c][l]) for l in range(H.shape[1])) % p)
        if (a, b, c) == (0, 0, 0):
            if t == 0: return False, 0
            lam = t
        elif t: return False, None
    return True, lam

def support_distance(A, H, p, maxw):
    """Smallest |T| such that some vector supported in T lies in A^perp but not in H^perp (H contains A):
    ker(A_T) != ker(H_T). Linear algebra per support, no coefficient enumeration. None if > maxw."""
    A = np.array(A, dtype=np.int64).reshape(-1, H.shape[1]); n = H.shape[1]
    for w in range(1, maxw + 1):
        for T in itertools.combinations(range(n), w):
            T = list(T); kA = len(nullspace(A[:, T], p, w)) if len(A) else w
            if kA != len(nullspace(H[:, T], p, w)): return w
    return None

def rand_full_rank(k, n, p):
    while True:
        G = rng.integers(0, p, (k, n))
        if naive_rank(G, p) == k: return G

# ---------------------------------------------------------------- 1. linear algebra
print('1. linear algebra over F_p')
ok_r = ok_n = ok_f = True
for p in (2, 3, 5, 7):
    for _ in range(15):
        k, n = rng.integers(1, 4), rng.integers(2, 6)
        M = rng.integers(0, p, (k, n)); r = rank(M, p)
        ok_r &= r == naive_rank(M, p) == rank(M.T, p)
        N = nullspace(M, p, n)
        ok_n &= len(N) == n - r and not ((N @ M.T) % p).any()
        ok_n &= {tuple(v) for v in span(N, p)} == {tuple(v) for v in naive_dual(M, p, n)}
        R, piv = rref(M, p); ok_f &= np.array_equal(rref(R, p)[0], R) and rank(R, p) == r
check('rank equals the naive rank and the rank of the transpose', ok_r)
check('nullspace is exactly the dual, of dimension n - rank', ok_n)
check('rref is idempotent and preserves the rank', ok_f)

# ---------------------------------------------------------------- 2. cyclic codes
print('2. cyclic codes')
ok_s = ok_d = ok_p = True
for p, n in ((3, 8), (3, 11), (3, 13), (5, 8), (5, 11), (7, 6), (7, 9), (2, 7)):
    cs = cosets(n, p)
    for _ in range(4):
        sup = sorted(set().union(*[cs[i] for i in rng.choice(len(cs), rng.integers(1, len(cs)), replace=False)]))
        B = cyclic_basis(n, p, sup)
        ok_d &= len(B) == len(sup) == rank(B, p)
        ok_s &= rank(np.vstack([B, np.roll(B, 1, axis=1)]), p) == len(B)          # shift invariant
        comp = sorted(set(range(n)) - {(-s) % n for s in sup})                     # dual: Z_n minus (-Sigma)
        D = cyclic_basis(n, p, comp) if comp else np.zeros((0, n), dtype=np.int64)
        ok_d &= not ((D @ B.T) % p).any() and len(D) == n - len(B)
        if p > 2 and n % p:                                                         # P-code criterion
            crit = 0 not in sup and all((a + b) % n for a in sup for b in sup)
            is_p = not ((B @ B.T) % p).any() and not (B.sum(1) % p).any()
            ok_p &= crit == is_p
check('dimension equals |Sigma|, and the code is invariant under cyclic shifts', ok_s and ok_d)
check('the dual has spectral support Z_n minus (-Sigma)', ok_d)
check('P-code iff 0 is not in Sigma or Sigma + Sigma', ok_p)

# ---------------------------------------------------------------- 3. triorthogonality and P-code checks
print('3. triorthogonality and P-code checks')
H8 = np.array([[1, 1, 1, 1, 1, 1, 1, 1], [1, 2, 0, 1, 2, 0, 1, 2], [0, 0, 1, 1, 1, 2, 2, 2]])
known = [(H8, 3, None), (construction_B(12, 5, [2, 3, 9, 10]), 5, None), (construction_A(6, 7, [1]), 7, None),
         (construction_A(4, 5, [1]), 5, None)]
ok_k = ok_rand = True
for H, p, w in known:
    w = np.ones(H.shape[1], dtype=np.int64) if w is None else w
    lam = check_triorthogonal(H, p, w); ok, lam2 = tau_all_zero(H % p, p, w); ok_k &= ok and lam == lam2
for p in (3, 5, 7):
    for _ in range(40):
        H = rand_full_rank(3, 6, p); w = rng.integers(1, p, 6); ok, _ = tau_all_zero(H, p, w)
        try: check_triorthogonal(H, p, w); got = True
        except AssertionError: got = False
        ok_rand &= got == ok
check('known triorthogonal codes are accepted, with the same lambda as a direct evaluation', ok_k)
check('random matrices: accepted exactly when the direct evaluation says triorthogonal', ok_rand)
ok_pc = True
for p in (3, 5, 7):
    for _ in range(40):
        S = rand_full_rank(2, 5, p); om = rng.integers(1, p, 5)
        direct = om.sum() % p != 0 and all(int((om * a * b).sum()) % p == 0 for a in S for b in S) \
            and all(int((om * a).sum()) % p == 0 for a in S)
        try: check_pcode(S, p, om); got = True
        except AssertionError: got = False
        ok_pc &= got == direct
check('weighted P-code check agrees with the definition on random inputs', ok_pc)

# ---------------------------------------------------------------- 4. doubling
print('4. doubling')
W7 = (np.array([[0,0,1],[0,1,0],[0,1,1],[1,1,1],[1,1,2],[1,2,1],[1,2,2]]).T, np.array([2,2,1,2,1,1,2]), 3, 3)
W5 = (np.array([[3,4],[2,0],[0,1],[3,1],[2,2]]).T, np.array([2,2,3,3,3]), 5, 3)
P6 = (cyclic_basis(6, 7, [4, 5]), None, 7, 3)
import json
J = json.load(open(os.path.join(ROOT, 'codes', 'codes.json')))
H19 = np.array([J['T19_3']['logical_row']] + J['T19_3']['stabilizer_rows'])      # logical row with entries 2
seeds = {3: [(np.array([[1]]), 1), (H8, 2), (H19, 3), (np.array([[2]]), 1)],
         5: [(np.array([[1]]), 1), (construction_A(4, 5, [1]), 2), (construction_B(12, 5, [2, 3, 9, 10]), 3)],
         7: [(np.array([[1]]), 1), (construction_A(6, 7, [1]), 2), (np.array([[3]]), 1)]}
N_of = lambda om, S: S.shape[1] if om is None else int(om.sum())
ok_t = ok_l = ok_dd = ok_sign = True
for S, om, p, dP in (W7, W5, P6):
    for T, dT in seeds[p]:
        H, w, kappa, eta = double(S, T, p, None, om)
        ok, lam = tau_all_zero(H, p, w); ok_t &= ok
        ok_l &= lam == int((T[0] ** 3).sum() % p)
        ok_t &= (eta * N_of(om, S) * kappa ** 3 + lam) % p == 0
        d = support_distance(H[1:], H, p, 5); ok_dd &= d == min(dP, dT + 2)
        if p == 3:
            H2, w2 = remove_sign_weights(H, w, p); ok2, lam2 = tau_all_zero(H2, p, w2)
            ok_sign &= ok2 and lam2 == lam and (w2 == 1).all() and support_distance(H2[1:], H2, p, 5) == d
check('doubled matrices are triorthogonal, and eta N kappa^3 + lambda_T = 0', ok_t)
check('the logical eigenvalue of the seed is preserved', ok_l)
check('distance equals min(d_P, d_T + 2) (independent support computation)', ok_dd)
check('p = 3: negating the columns with coefficient -1 keeps triorthogonality and distance', ok_sign)
H16, w16 = concatenate(construction_A(4, 5, [1]), construction_A(4, 5, [1]), 5)
ok16, _ = tau_all_zero(H16, 5, w16)
check('concatenation [[4,1,2]]_5 with itself: triorthogonal [[16,1,4]]_5',
      ok16 and support_distance(H16[1:], H16, 5, 4) == 4)

# ---------------------------------------------------------------- 5. distance routines
print('5. distance routines')
ok_nl = ok_pd = ok_isd = ok_dz = True
cases = [(H8, 3), (construction_B(12, 5, [2, 3, 9, 10]), 5), (construction_A(6, 7, [1]), 7)]
for S, om, p, dP in (W7, W5, P6):
    H, w, _, _ = double(S, np.array([[1]]), p, None, om); cases.append((H, p))
for H, p in cases:
    d = support_distance(H[1:], H, p, 5)
    ok_nl &= no_logical_below(H, p, d) and not no_logical_below(H, p, d + 1)
    v = isd_witness(H, p, d, iters=200); ok_isd &= v is not None and is_logical(v, H, p) and (v != 0).sum() == d
for S, om, p, dP in ((W7[0], None, 3, None), (cyclic_basis(11, 3, [1, 3, 4, 5, 9]), None, 3, 5),
                      (cyclic_basis(8, 5, [1, 2, 5]), None, 5, 3), (P6[0], None, 7, 3)):
    n = S.shape[1]; dd, wit = pcode_distance_exact(S, p)
    ref = support_distance(S, np.vstack([np.ones(n, dtype=np.int64), S]), p, 6)
    ok_pd &= dd == ref and (dP is None or dd == dP) and not ((S @ wit) % p).any() and wit.sum() % p and (wit != 0).sum() == dd
check('no_logical_below agrees with the support computation', ok_nl)
check('isd_witness finds a logical of minimum weight', ok_isd)
check('pcode_distance_exact agrees with the support computation', ok_pd)
from punct import unpuncture, valid
for nm in ('D15_3', 'T11_5'):
    p, H, c = unpuncture(J[nm]); r, n = H.shape; tried = 0
    while tried < 6:
        S = sorted(rng.choice(n, 2, replace=False).tolist())
        if not valid(H, p, S): continue
        tried += 1; rest = [l for l in range(n) if l not in S]
        Ann = nullspace(H[:, S].T, p, r); Ap = (Ann @ H[:, rest]) % p; Hp = H[:, rest] % p
        ok_dz &= dZ(H, p, S, maxw=4) == (support_distance(Ap, Hp, p, 4) or 5)
check('punctured distance dZ agrees with the support computation', ok_dz)

# ---------------------------------------------------------------- 6. weight distributions
print('6. weight distributions and MacWilliams')
ok_w = ok_c = ok_m = True
for p in (2, 3, 5, 7):
    for _ in range(6):
        k, n = int(rng.integers(1, 4)), int(rng.integers(3, 7)); G = rand_full_rank(k, n, p)
        ref = np.bincount((span(G, p) != 0).sum(1), minlength=n + 1)
        ok_w &= list(span_wd(G, p, n)) == ref.tolist()
        dual = np.bincount((naive_dual(G, p, n) != 0).sum(1), minlength=n + 1)
        ok_m &= list(dual_wd(span_wd(G, p, n), n, p)) == dual.tolist()
for k, n in ((4, 20), (8, 60), (13, 100), (14, 128)):
    G = rng.integers(0, 3, (k, n)); a = _wd3(G)
    b = span_wd(G, 3, n) if k <= 12 else None
    if b is None:                       # numpy reference for k > 12, chunked
        b = np.zeros(n + 1, dtype=np.int64); co = np.array(list(itertools.product(range(3), repeat=k - 6)))
        for c6 in itertools.product(range(3), repeat=6):
            V = (np.hstack([co, np.tile(c6, (len(co), 1))]) @ G) % 3
            b += np.bincount((V != 0).sum(1), minlength=n + 1)
    ok_c &= [int(x) for x in a] == [int(x) for x in b]
check('numpy weight distribution agrees with full enumeration (p = 2, 3, 5, 7)', ok_w)
check('wd3.c agrees with numpy, n up to 128, k up to 14', ok_c)
check('MacWilliams transform agrees with enumeration of the dual', ok_m)

# ---------------------------------------------------------------- 7. distillation statistics
print('7. distillation statistics')
def naive_stats(H0, F, p, eps):
    n = H0.shape[1]; acc = bad = 0.0; worst = np.zeros(len(F))
    for e in itertools.product(range(p), repeat=n):
        e = np.array(e); wt = int((e != 0).sum()); pr = (1 - eps) ** (n - wt) * (eps / (p - 1)) ** wt
        if ((H0 @ e) % p).any(): continue
        acc += pr; worst += pr * (((F @ e) % p) != 0)
    return worst.max() / acc, acc
ok_s = True
for name, H0, F, p in (('[[8,1,2]]_3', H8[1:], H8[:1], 3), ('[[6,1,2]]_7', construction_A(6, 7, [1])[1:], construction_A(6, 7, [1])[:1], 7),
                       ('[[7,1,3]] qubit Steane, logical all-ones', np.array([[1,0,1,0,1,0,1],[0,1,1,0,0,1,1],[0,0,0,1,1,1,1]]),
                        np.ones((1, 7), dtype=np.int64), 2)):
    C = Code(name, H0, F, p)
    for eps in (0.01, 0.2):
        a, b = C.eps_stats(eps), naive_stats(H0, F, p, eps)
        ok_s &= abs(a[0] - b[0]) < 1e-12 and abs(a[1] - b[1]) < 1e-12
check('output error and acceptance agree with a sum over all error patterns', ok_s)
d = 0.03; C = Code('x', H8[1:], H8[:1], 3); out, acc = C.step(d); o2, a2 = C.eps_stats(2 * d / 3)
check('step(delta) converts between depolarizing strength and infidelity', abs(out - 1.5 * o2) < 1e-15 and acc == a2)

# ---------------------------------------------------------------- 8. sequence optimizer
print('8. sequence optimizer')
def toy(n, k, a, b):          # delta_out = a delta^2, acceptance 1 - b delta
    return Fn(f'toy{n}', n, k, lambda d: (a * d * d, 1 - b * d))
codes = [toy(8, 1, 2, 8), toy(14, 4, 6, 14), toy(20, 7, 10, 20), toy(15, 1, 0.5, 15)]
ok_o = True
for d0 in (0.02, 0.05):
    res = optimize(codes, d0=d0, R=3, xs=range(2, 12))
    for x in range(2, 12):
        best = None
        for R in range(1, 4):
            for seq in itertools.product(codes, repeat=R):
                d, c, good = d0, 1.0, True
                for code in seq:
                    dout, acc = code.step(d)
                    if not (0 < dout < d) or acc <= 0: good = False; break
                    c *= code.n / (code.k * acc); d = dout
                if good and d <= 10.0 ** -x and (best is None or c < best - 1e-9): best = c
        got = res[x][1] if res[x] else None
        ok_o &= (best is None and got is None) or (best is not None and got is not None and abs(got - best) <= 1e-9 * best)
check('optimal costs agree with exhaustive enumeration of all sequences of up to 3 rounds', ok_o)

print('all self-tests passed' if not fails else f'{fails} SELF-TEST(S) FAILED')
sys.exit(1 if fails else 0)
