"""Shared library: linear algebra over F_p, cyclic codes, triorthogonality and P-code checks,
the doubling construction, concatenation, and distance certificates. Requires numpy, sympy."""
import numpy as np, itertools, sympy
from functools import lru_cache

# ---------------- GF(p) linear algebra ----------------
def rref(M, p):
    M = np.array(M, dtype=np.int64) % p
    rows, cols = M.shape; r = 0; piv = []
    for c in range(cols):
        nz = [i for i in range(r, rows) if M[i, c]]
        if not nz: continue
        i = nz[0]; M[[r, i]] = M[[i, r]]
        M[r] = (M[r] * pow(int(M[r, c]), p - 2, p)) % p
        for j in range(rows):
            if j != r and M[j, c]: M[j] = (M[j] - M[j, c] * M[r]) % p
        piv.append(c); r += 1
        if r == rows: break
    return M[:r], piv

def rank(M, p):
    M = np.atleast_2d(np.array(M))
    return 0 if M.size == 0 else len(rref(M, p)[1])

def nullspace(M, p, n):
    M = np.array(M, dtype=np.int64).reshape(-1, n)
    if len(M) == 0: return np.eye(n, dtype=np.int64)
    R, piv = rref(M, p); free = [c for c in range(n) if c not in piv]; B = []
    for f in free:
        v = np.zeros(n, dtype=np.int64); v[f] = 1
        for i, c in enumerate(piv): v[c] = (-R[i, f]) % p
        B.append(v)
    return np.array(B, dtype=np.int64).reshape(len(B), n)

# ---------------- cyclic codes ----------------
def cosets(n, p):
    seen = set(); out = []
    for s in range(n):
        if s in seen: continue
        c = []; t = s
        while t not in c: c.append(t); t = (t * p) % n
        seen |= set(c); out.append(tuple(sorted(c)))
    return out

@lru_cache(None)
def _minpoly(n, p):
    """alpha = root of the first irreducible factor of Phi_n over F_p returned by SymPy."""
    x = sympy.symbols('x')
    f = sympy.Poly(sympy.cyclotomic_poly(n, x), x, modulus=p).factor_list()[1][0][0]
    return tuple(int(c) % p for c in f.all_coeffs()[::-1])

def _mulmod(a, b, f, p):
    m = len(f) - 1; r = np.convolve(a, b) % p
    for i in range(len(r) - 1, m - 1, -1):
        if r[i]: r[i - m:i + 1] = (r[i - m:i + 1] - r[i] * np.array(f)) % p
    return r[:m]

@lru_cache(None)
def _powers(n, p):
    f = np.array(_minpoly(n, p)); m = len(f) - 1
    P = np.zeros((n, m), dtype=np.int64); cur = np.zeros(m, dtype=np.int64); cur[0] = 1
    if m == 1:
        a = (-f[0]) % p
        for l in range(n): P[l] = cur; cur = (cur * a) % p
    else:
        xx = np.zeros(m, dtype=np.int64); xx[1] = 1
        for l in range(n): P[l] = cur; cur = _mulmod(cur, xx, f, p)
    return P

def cyclic_basis(n, p, support):
    """Basis of the cyclic code of length n over F_p with spectral support `support`."""
    S = set(support); P = _powers(n, p); eqs = []
    for c in cosets(n, p):
        assert (c[0] in S) == all(t in S for t in c), "support must be a union of cyclotomic cosets"
        if c[0] in S: continue
        eqs.append(np.array([P[(c[0] * l) % n] for l in range(n)]).T)
    if not eqs: return np.eye(n, dtype=np.int64)
    B = nullspace(np.vstack(eqs) % p, p, n)
    assert len(B) == len(S)
    return B

# ---------------- triorthogonality ----------------
def check_triorthogonal(H, p, w=None):
    """Triorthogonality with coefficients w (one logical row, row 0). Returns lambda or raises."""
    H = np.array(H) % p; m, n = H.shape
    w = np.ones(n, dtype=np.int64) if w is None else np.array(w)
    assert rank(H, p) == m, "rows not independent"
    lam = None
    for a in range(m):
        for b in range(a, m):
            ab = w * H[a] * H[b] % p
            for c in range(b, m):
                t = int((ab * H[c]).sum() % p)
                if a == b == c == 0:
                    assert t != 0, "logical cube vanishes"; lam = t
                else:
                    assert t == 0, f"triple ({a},{b},{c}) = {t}"
    return lam

def check_pcode(S, p, omega=None):
    """(Weighted) P-code: S isotropic for diag(omega), S perp_omega 1, sum(omega) != 0."""
    S = np.array(S) % p; n = S.shape[1]
    om = np.ones(n, dtype=np.int64) if omega is None else np.array(omega) % p
    assert om.sum() % p != 0 and not (((S * om) @ S.T) % p).any() and not ((S * om).sum(1) % p).any()
    assert rank(S, p) == len(S)

# ---------------- constructions ----------------
def construction_A(n, p, support):
    C = cyclic_basis(n, p, support)
    return np.vstack([np.ones(n, dtype=np.int64), C]) % p

def construction_B(n, p, support):
    W = cyclic_basis(n, p, support); R, piv = rref(W, p); assert piv[0] == 0
    return np.vstack([R[0, 1:], R[1:, 1:]]) % p

def cube_roots(a, p): return [x for x in range(1, p) if pow(x, 3, p) == a % p]

def double(S, T, p, wT=None, omega=None):
    """Doubling. S: stabilizers of a (weighted) P-code (r x n1) with charges omega.
    T: seed matrix, row 0 logical, coefficients wT. Returns (H, coefficients, kappa, eta), where
    eta is the coefficient on both copies of the P-code and eta * N * kappa^3 + lambda_T = 0."""
    S = np.array(S) % p; T = np.array(T) % p; n1 = S.shape[1]; n2 = T.shape[1]
    wT = np.ones(n2, dtype=np.int64) if wT is None else np.array(wT)
    om = np.ones(n1, dtype=np.int64) if omega is None else np.array(omega) % p
    N = int(om.sum() % p)
    y, T0 = T[0], T[1:]; lamT = int((wT * y ** 3).sum() % p)
    for eta in range(1, p):                    # smallest eta for which eta*N*kappa^3 = -lambda_T is solvable
        ks = cube_roots(-lamT * pow(eta * N, p - 2, p), p)
        if ks: kappa = ks[0]; break
    one = np.ones(n1, dtype=np.int64); z1 = np.zeros(n1, dtype=np.int64); z2 = np.zeros(n2, dtype=np.int64)
    rows = [np.concatenate([(-kappa * one) % p, kappa * one, y])]
    rows += [np.concatenate([s, (-s) % p, z2]) for s in S]
    rows += [np.concatenate([z1, kappa * one, y])]
    rows += [np.concatenate([z1, z1, t]) for t in T0]
    w = np.concatenate([eta * om, eta * om, wT]) % p
    return np.array(rows) % p, w, kappa, eta

def remove_sign_weights(H, w, p):
    """Weights -1 can be removed by negating the column, since (-x)^3 = -x^3."""
    H = np.array(H) % p; w = np.array(w) % p; m = (w == p - 1)
    H[:, m] = (-H[:, m]) % p; w = w.copy(); w[m] = 1
    return H, w

def concatenate(Hout, Hin, p, wout=None, win=None):
    Hout = np.array(Hout) % p; Hin = np.array(Hin) % p
    no, ni = Hout.shape[1], Hin.shape[1]
    wout = np.ones(no, dtype=np.int64) if wout is None else np.array(wout)
    win = np.ones(ni, dtype=np.int64) if win is None else np.array(win)
    h = Hin[0]; rows = [np.kron(x, h) for x in Hout]
    for i in range(no):
        e = np.zeros(no, dtype=np.int64); e[i] = 1
        rows += [np.kron(e, a) for a in Hin[1:]]
    return np.array(rows) % p, np.kron(wout, win) % p

# ---------------- distances ----------------
def is_logical(v, H, p):
    """v is a nontrivial Z-logical of Q(H): orthogonal to stabilizer rows, not to the logical row."""
    v = np.array(v) % p
    return not ((H[1:] @ v) % p).any() and (H[0] @ v) % p != 0

def no_logical_below(H, p, w, start=1):
    """Exhaustively check there is no Z-logical of weight < w. Returns True/False."""
    H = np.array(H) % p; n = H.shape[1]; A = H[1:]
    for wt in range(start, w):
        coeff = np.array(list(itertools.product(range(1, p), repeat=wt)), dtype=np.int64)
        for supp in itertools.combinations(range(n), wt):
            s = list(supp)
            a0 = (coeff @ A[:, s].T) % p; a1 = (coeff @ H[0, s]) % p
            if np.any(~a0.any(1) & (a1 != 0)): return False
    return True

def search_count(n, p, w):
    from math import comb
    return sum(comb(n, t) * (p - 1) ** t for t in range(1, w))

def pcode_distance_exact(S, p, chunk=200000):
    """min wt v in S^perp with nonzero sum, by full enumeration of S^perp."""
    S = np.array(S); n = S.shape[1]; G = nullspace(S, p, n); k = len(G); best = n + 1; wit = None
    for st in range(0, p ** k, chunk):
        idx = np.arange(st, min(p ** k, st + chunk)); co = np.zeros((len(idx), k), np.int64); t = idx.copy()
        for j in range(k): co[:, j] = t % p; t //= p
        V = (co @ G) % p; ok = V.sum(1) % p != 0
        if ok.any():
            wts = (V[ok] != 0).sum(1); i = wts.argmin()
            if wts[i] < best: best = int(wts[i]); wit = V[ok][i]
    return best, wit

def isd_witness(H, p, target, iters=2000, seed=0):
    """Randomized information-set search (Lee-Brickell, up to pairs of rows) for a Z-logical
    of weight <= target. Used only to exhibit upper-bound witnesses."""
    rng = np.random.default_rng(seed); H = np.array(H) % p; n = H.shape[1]
    G = nullspace(H[1:], p, n); best = None
    def consider(r, perm):
        nonlocal best
        if best is not None and (r != 0).sum() >= (best != 0).sum(): return
        v = np.zeros(n, dtype=np.int64); v[perm] = r
        if (H[0] @ v) % p: best = v
    for _ in range(iters):
        perm = rng.permutation(n); R, _ = rref(G[:, perm], p)
        for r in R: consider(r, perm)
        for a, b in itertools.combinations(range(len(R)), 2):
            for t in range(1, p): consider((R[a] + t * R[b]) % p, perm)
        if best is not None and (best != 0).sum() <= target: break
    return best
