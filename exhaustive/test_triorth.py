#!/usr/bin/env python3
"""
Tests for triorth.c (compile first: gcc -O2 -march=native -o triorth triorth.c).

  python3 test_triorth.py                 # brute-force cross-checks (~3 min)
  python3 test_triorth.py --long          # also p = 7, (r, m) = (2, 4) with general coefficients (long)
  python3 test_triorth.py verify LOG P    # re-verify every code printed in LOG

Cross-checks: with (g) switched off (-R), the double-lex representatives
returned by the C search must cover exactly the equivalence classes found by
naive enumeration of all (w, c, q, P) with lambda = 1.  Equivalence: column
permutations, row permutations, and for p = 7 scaling of rows/columns of
[q; P] by cube roots of unity {1, 2, 4}.

Verification: every printed code is rebuilt as H = [h; A] with weights w and
checked directly against the definition of triorthogonality, and d_Z is computed by brute force.
"""
import itertools, os, re, subprocess, sys, time
import numpy as np

BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "triorth")

# ---------------------------------------------------------------- parsing
def parse_log(text):
    lines = text.split("\n")
    out = []
    for i, l in enumerate(lines):
        if l.strip().startswith("CODE"):
            r = int(re.search(r"r=(\d+)", l).group(1))
            get = lambda s: [int(c) for c in s.split(":")[1].replace(" ", "")]
            w = np.array(get(lines[i + 1]))
            h = np.array(get(lines[i + 2]))
            A = np.array([get(lines[i + 3 + j]) for j in range(r)])
            out.append((r, w, h, A))
    return out

# ---------------------------------------------------------------- direct checks
def def5_violations(p, w, h, A):
    H = np.vstack([h, A]) % p
    bad = 0
    for a, b, c in itertools.combinations_with_replacement(range(len(H)), 3):
        t = int((w * H[a] * H[b] * H[c]).sum() % p)
        if (a == b == c == 0) != (t != 0):
            bad += 1
    return bad

def distance_Z(p, h, A, wmax=4):
    n = len(h)
    for s in range(1, wmax + 1):
        for S in itertools.combinations(range(n), s):
            sub, hs = A[:, S], h[list(S)]
            for vals in itertools.product(range(1, p), repeat=s - 1):
                v = np.array((1,) + vals)
                if not (sub @ v % p).any() and (hs @ v) % p:
                    return s
    return None  # > wmax

def verify_log(path, p):
    codes = parse_log(open(path).read())
    bad = [def5_violations(p, w, h, A) for _, w, h, A in codes]
    ds = [distance_Z(p, h, A) for _, w, h, A in codes]
    print(f"{path}: {len(codes)} codes, triorthogonality violations {set(bad) or '{}'}, "
          f"d_Z values {sorted(set(ds), key=lambda x: (x is None, x))} (None = >4)")
    return all(b == 0 for b in bad) and all(d is None or d >= 3 for d in ds)

# ---------------------------------------------------------------- brute force
def brute(p, r, m, Ws, Cs):
    """All (w, c, q, P) satisfying (a)-(f) with lambda = 1, w in Ws, c in Cs."""
    Qall = np.array(list(itertools.product(range(p), repeat=m)))
    out = []
    for W in Ws:
        W = np.array(W)
        for C in Cs:
            rows = [[np.array(v) for v in itertools.product(range(p), repeat=m)
                     if (C[i] + (W * np.array(v) ** 3).sum()) % p == 0] for i in range(r)]
            for P in itertools.product(*rows):
                P = np.array(P)
                if any((W * P[i] ** 2 * P[j]).sum() % p for i in range(r) for j in range(r) if i != j):
                    continue
                if any((W * P[i] * P[j] * P[k]).sum() % p for i, j, k in itertools.combinations(range(r), 3)):
                    continue
                ok = (Qall ** 3 @ W) % p == 1
                for i in range(r):
                    ok &= ((Qall ** 2) @ (W * P[i])) % p == 0
                    for j in range(r):
                        ok &= (Qall @ (W * P[i] * P[j])) % p == 0
                for q in Qall[ok]:
                    out.append((W, tuple(C), q, P))
    return out

def canon(p, W, C, q, P):
    r, m = P.shape
    zs = [1, 2, 4] if p == 7 else [1]
    best = None
    for s in itertools.permutations(range(m)):
        ww = tuple(int(W[l]) for l in s)
        for cz in itertools.product(zs, repeat=m):
            qq = tuple(int(q[s[l]]) * cz[l] % p for l in range(m))
            for rz in itertools.product(zs, repeat=r):
                PP = tuple(sorted((int(C[i]),) + tuple(int(P[i][s[l]]) * cz[l] * rz[i] % p for l in range(m))
                                  for i in range(r)))
                key = (ww, qq, PP)
                if best is None or key < best:
                    best = key
    return best

def search_relaxed(p, r, m, unit_only):
    txt = subprocess.run([BIN, str(p), str(r + m), "-R", "-v", "-rm", str(r), str(m)],
                         capture_output=True, text=True).stdout
    res = set()
    for rr, w, h, A in parse_log(txt):
        C, W = w[:r], w[r:]
        if unit_only and (w != 1).any():
            continue
        res.add(canon(p, W, C, h[r:], A[:, r:]))
    return res

def cross_checks(long=False):
    ok = True
    cases = [(3, 2, 3, True), (3, 2, 4, True), (3, 2, 5, True), (3, 3, 4, True), (3, 3, 5, True),
             (5, 2, 3, True), (5, 2, 4, True), (5, 3, 3, True),
             (7, 2, 3, True), (7, 2, 4, True), (7, 2, 3, False)] + ([(7, 2, 4, False)] if long else [])
    for p, r, m, unit in cases:
        t = time.time()
        if unit:
            Ws, Cs = [(1,) * m], [(1,) * r]
        else:
            Ws = list(itertools.product([1, 2, 3], repeat=m))
            Cs = list(itertools.product([1, 2, 3], repeat=r))
        B = {canon(p, *x) for x in brute(p, r, m, Ws, Cs)}
        S = search_relaxed(p, r, m, unit)
        eq = B == S
        ok &= eq
        print(f"p={p} r={r} m={m} {'unit' if unit else 'weighted'}: brute-force classes {len(B)}, "
              f"search classes {len(S)}, equal {eq}  [{time.time() - t:.0f}s]", flush=True)
    return ok

if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "verify":
        sys.exit(0 if verify_log(sys.argv[2], int(sys.argv[3])) else 1)
    sys.exit(0 if cross_checks('--long' in sys.argv) else 1)
