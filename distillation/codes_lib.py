"""Exact single-round distillation statistics from weight enumerators.

For a code with stabilizer rows H0 and logical rows F, the undetected Z errors are A^perp with
A = span(H0), and an error is harmless for output a iff it lies in (A + <f_a>)^perp. The weight
distributions of these duals follow from those of the small spaces by the MacWilliams identity.
Weight distributions over F_3 are enumerated by wd3.c (compiled on first use); other p use numpy."""
import os, sys, subprocess, numpy as np
from math import comb
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..')
sys.path.insert(0, ROOT)
from qcodes import rref, nullspace
EXE = os.path.join(HERE, 'wd3')

def _wd3(G):
    if not os.path.exists(EXE):
        subprocess.run(['gcc', '-O3', '-o', EXE, os.path.join(HERE, 'wd3.c')], check=True)
    k, n = G.shape
    inp = f"{n} {k}\n" + "\n".join(" ".join(map(str, r)) for r in G)
    out = subprocess.run([EXE], input=inp, capture_output=True, text=True, check=True).stdout
    return np.array(list(map(int, out.split())), dtype=object)

def span_wd(G, p, n):
    """Weight distribution (length n+1, Python ints) of the span of the rows of G over F_p."""
    G = np.array(G, dtype=np.int64).reshape(-1, n) % p; k = len(G)
    if p == 3 and k > 12 and n <= 128: return _wd3(G)
    out = np.zeros(n + 1, dtype=np.int64); tot = p ** k; ch = 200000
    for st in range(0, tot, ch):
        idx = np.arange(st, min(tot, st + ch)); co = np.zeros((len(idx), k), np.int64); t = idx.copy()
        for j in range(k): co[:, j] = t % p; t //= p
        out += np.bincount((((co @ G) % p) != 0).sum(1), minlength=n + 1)
    return np.array([int(x) for x in out], dtype=object)

def dual_wd(A, n, q):
    """MacWilliams identity, exact: weight distribution of the dual of a code with distribution A."""
    size = int(sum(A)); B = [0] * (n + 1)
    for w, a in enumerate(A):
        a = int(a)
        if a:
            for i in range(n - w + 1):
                ci = comb(n - w, i) * (q - 1) ** i
                for j in range(w + 1): B[i + j] += a * ci * comb(w, j) * (-1) ** j
    assert all(b % size == 0 for b in B)
    return np.array([b // size for b in B], dtype=object)

class Code:
    """Per-output error and acceptance probability under twirled depolarizing noise of strength delta:
    each input carries a nontrivial Z^j error with total probability eps = (p-1)/p * delta, uniform over
    the p-1 values. step(delta) returns (delta_out of the worst output, acceptance probability)."""
    def __init__(s, name, H0, F, p):
        s.name = name; s.p = p; F = np.array(F).reshape(-1, np.array(F).shape[-1]); s.n = F.shape[1]
        H0 = np.array(H0).reshape(-1, s.n); s.k = len(F)
        s.Bacc = dual_wd(span_wd(H0, p, s.n), s.n, p)                               # A^perp
        s.Bok = [dual_wd(span_wd(np.vstack([H0, f]), p, s.n), s.n, p) for f in F]   # (A + <f_a>)^perp
    def eps_stats(s, eps):
        """(worst-output logical error probability, acceptance) at infidelity eps."""
        p = s.p; P = [(1 - eps) ** (s.n - w) * (eps / (p - 1)) ** w for w in range(s.n + 1)]
        acc = float(sum(float(b) * P[w] for w, b in enumerate(s.Bacc)))
        bad = max(float(sum(float(a - b) * P[w] for w, (a, b) in enumerate(zip(s.Bacc, B)))) for B in s.Bok)
        return bad / acc, acc
    def step(s, delta):
        p = s.p; out, acc = s.eps_stats((p - 1) / p * delta)
        return out * p / (p - 1), acc
    def A_d(s):
        """(d, number of minimum-weight logicals of the worst output)."""
        bad = [max(int(s.Bacc[w] - B[w]) for B in s.Bok) for w in range(s.n + 1)]
        d = next(w for w, b in enumerate(bad) if b); return d, bad[d]

def threshold(f, lo=1e-4, hi=None, grid=4000):
    """Smallest x in (lo, hi) with f(x) >= x, by a grid scan followed by bisection; None if none."""
    xs = np.linspace(lo, hi, grid)
    for a, b in zip(xs[:-1], xs[1:]):
        if f(a) < a and f(b) >= b:
            for _ in range(60):
                m = (a + b) / 2
                if f(m) < m: a = m
                else: b = m
            return (a + b) / 2
    return None

def ps_family(m):
    """Prakash-Saha [[6m+2,3m-2,2]]_3 code: their [[9m,3m-1]] space punctured at coordinates 3j, j < 3m-2."""
    n = 9 * m; w = np.array([i % 3 for i in range(n)]); rows = [w]
    for a in range(1, 3 * m):
        v = np.zeros(n, int); v[3 * (a - 1):3 * a] = 1; v[n - 3:] = 2; rows.append(v)
    W = np.array(rows) % 3; k = 3 * m - 2; punct = [3 * j for j in range(k)]
    order = punct + [c for c in range(n) if c not in punct]
    R, piv = rref(W[:, order], 3)
    assert piv[:k] == list(range(k))
    return Code(f"[[{n-k},{k},2]]_3", R[k:, k:], R[:k, k:], 3)
