"""Weighted P-codes from projective point configurations (linear-algebra reduction).
Columns c_l = a_l v_l, weights w_l.  With beta_l = w_l a_l^2 and b_l = 1/a_l:
  sum beta_l v_l v_l^T = 0            (second moments)
  sum beta_l b_l v_l = 0              (first moments)
  for every dependency u among <= d-1 points: sum u_l b_l = 0     (distance >= d)
  b, beta nowhere zero; mass sum beta_l b_l^2 != 0."""
import numpy as np, itertools
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from qcodes import rref, rank, nullspace

def proj_points(p, r):
    out = []
    for v in itertools.product(range(p), repeat=r):
        v = np.array(v)
        if v.any() and v[np.nonzero(v)[0][0]] == 1: out.append(v)
    return np.array(out, dtype=np.int64)

def qmat(V, p):
    r = V.shape[1]; iu = np.triu_indices(r)
    return np.array([np.outer(v, v)[iu] for v in V]).T % p      # (r(r+1)/2) x n

def nowhere_zero_in(K, p, rng, tries=400):
    """yield nowhere-zero vectors of rowspace(K) (exhaustive if small, else random)."""
    k = len(K)
    if k == 0: return
    if p ** k <= 20000:
        for co in itertools.product(range(p), repeat=k):
            if not any(co): continue
            # projective: first nonzero coefficient = 1
            if co[next(i for i, c in enumerate(co) if c)] != 1: continue
            v = (np.array(co) @ K) % p
            if (v != 0).all(): yield v
    else:
        for _ in range(tries):
            v = (rng.integers(0, p, k) @ K) % p
            if (v != 0).all(): yield v

def dependency_rows(V, p, dmax):
    """constraints sum u_l b_l = 0 for dependencies u among subsets of size <= dmax-1 (size>=3)."""
    n = len(V); rows = []
    for t in range(3, dmax):
        for T in itertools.combinations(range(n), t):
            T = list(T); K = nullspace(V[T].T % p, p, t)
            for u in K:
                if (u != 0).all():          # only circuits on exactly T (smaller ones counted already)
                    row = np.zeros(n, dtype=np.int64); row[T] = u; rows.append(row)
    return rows

def pcode_distance(S, p, cap):
    """min weight v perp S with sum v != 0 (capped search)."""
    n = S.shape[1]
    for w in range(1, cap):
        co = np.array(list(itertools.product(range(1, p), repeat=w))); co = co[co.sum(1) % p != 0]
        for s in itertools.combinations(range(n), w):
            if (~((co @ S[:, list(s)].T) % p).any(1)).any(): return w
    return cap

def codes_from_config(V, p, d, rng, maxout=1):
    """Return list of (S, w, dist) weighted P-codes with distance >= d built on point set V."""
    n, r = V.shape; out = []
    KQ = nullspace(qmat(V, p), p, n)
    if len(KQ) == 0: return out
    dep = dependency_rows(V, p, d)
    for beta in nowhere_zero_in(KQ, p, rng):
        A = [(beta[l] * V[l]) % p for l in range(n)]
        M = np.array(A).T % p
        if dep: M = np.vstack([M, np.array(dep)])
        Kb = nullspace(M % p, p, n)
        for b in nowhere_zero_in(Kb, p, rng):
            if (beta * b * b).sum() % p == 0: continue
            a = np.array([pow(int(x), p - 2, p) for x in b])
            C = (a[:, None] * V) % p; S = C.T % p
            w = (beta * b * b) % p
            if rank(S, p) < r: continue
            dist = pcode_distance(S, p, d + 1)
            if dist >= d:
                out.append((S, w, dist))
                if len(out) >= maxout: return out
    return out
