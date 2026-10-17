#!/usr/bin/env python3
r"""
Exhaustive search for qutrit (p = 3) triorthogonal codes with distance d >= 3.

Triorthogonality as in Prakash, "Cyclic triorthogonal codes in prime dimension": only the
cubic conditions tau(h_a, h_b, h_c) = sum_l h_al h_bl h_cl = 0 (unless
a = b = c is a logical row), no pairwise conditions.  Unit weights are
assumed; for p = 3 this is no loss of generality, since cubing is a bijection of F_3.

Reduced problem (the normal form of the paper).  A shortest [[n,1,>=3]]_3 code may be
taken with
    A = [ I_r | P ]   (stabilizer rows),     h = ( 0 | q )   (logical row),
where P is r x m over F_3, q in F_3^m, n = r + m, and
  (a) sum_l P_il = 2                              for every i,
  (b) sum_{l in supp p_i} P_jl = 0                for i != j,
  (c) sum_l P_il P_jl P_kl = 0                    for distinct i, j, k,
  (d) sum_l q_l != 0,
  (e) sum_{l in supp q} P_il = 0                  for every i,
  (f) sum_l q_l P_il P_jl = 0                     for all i, j (incl. i = j),
  (g) the columns of P have weight >= 2 and are pairwise non-parallel
      (equivalently, A is projective, i.e. d(A^perp) >= 3).
Moreover r + 1 <= m <= (3^r - 1)/2 - r.

Symmetry breaking: q is normalised to sum(q) = 1 (q -> -q is a symmetry);
the matrix [q; P] is required to be "double-lex": its columns are
lexicographically non-decreasing (q is the most significant entry) and the
rows of P are lexicographically non-decreasing.  Every solution is
equivalent (under column permutations, permutations of the rows of P, and
q -> -q) to one of this form.

Usage:
    python3 qutrit_triorth_search.py                # self-tests + n <= 14 + n = 15
    python3 qutrit_triorth_search.py --nmax 14 --no-15
Requires: numpy, Python >= 3.8.
"""
import argparse
import itertools
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Basic F_3 helpers
# ---------------------------------------------------------------------------

def all_vectors(m):
    """All vectors of F_3^m in lexicographic order (column 0 most significant)."""
    if m == 0:
        return np.zeros((1, 0), dtype=np.int64)
    return np.array(list(itertools.product(range(3), repeat=m)), dtype=np.int64)


def rref_mod3(M):
    """Reduced row echelon form over F_3. Returns (R, pivot_columns)."""
    M = np.array(M, dtype=np.int64) % 3
    rows, cols = M.shape
    piv, r = [], 0
    for c in range(cols):
        if r == rows:
            break
        nz = [i for i in range(r, rows) if M[i, c]]
        if not nz:
            continue
        M[[r, nz[0]]] = M[[nz[0], r]]
        M[r] = (M[r] * (1 if M[r, c] == 1 else 2)) % 3
        for i in range(rows):
            if i != r and M[i, c]:
                M[i] = (M[i] - M[i, c] * M[r]) % 3
        piv.append(c)
        r += 1
    return M[:r], piv


def rank_mod3(M):
    return len(rref_mod3(M)[1])

# ---------------------------------------------------------------------------
# Direct checks on a code given by a logical row h and stabilizer rows A
# ---------------------------------------------------------------------------

def definition5_violations(h, A):
    """List of index triples (0 = logical row) violating triorthogonality (c = 1)."""
    H = np.vstack([h, A]) % 3
    bad = []
    for a, b, c in itertools.combinations_with_replacement(range(len(H)), 3):
        t = int((H[a] * H[b] * H[c]).sum() % 3)
        if a == b == c == 0:
            if t == 0:
                bad.append((a, b, c))
        elif t != 0:
            bad.append((a, b, c))
    return bad


def distance_Z(h, A, wmax=6):
    """min wt(v), v in A^perp \\ H^perp, by brute force up to weight wmax."""
    A = np.array(A) % 3
    h = np.array(h) % 3
    n = len(h)
    for w in range(1, wmax + 1):
        for S in itertools.combinations(range(n), w):
            sub_A, sub_h = A[:, S], h[list(S)]
            for vals in itertools.product((1, 2), repeat=w):
                v = np.array(vals)
                if not (sub_A @ v % 3).any() and (sub_h @ v) % 3:
                    return w
    return None


def distance_X(h, A):
    """min wt(u), u in H \\ A, by enumerating H (fine for small dim)."""
    H = np.vstack([h, A]) % 3
    best = None
    for c in itertools.product(range(3), repeat=len(H)):
        if c[0] == 0:
            continue
        w = int(np.count_nonzero(np.array(c) @ H % 3))
        best = w if best is None else min(best, w)
    return best


def code_from_reduced(q, P):
    """h = (0|q), A = [I|P]."""
    P = np.array(P) % 3
    q = np.array(q) % 3
    r = P.shape[0]
    A = np.hstack([np.eye(r, dtype=np.int64), P])
    h = np.concatenate([np.zeros(r, dtype=np.int64), q])
    return h, A


def reduced_conditions(q, P):
    """Check (a)-(f) of the reduced problem; returns dict of booleans."""
    P = np.array(P) % 3
    q = np.array(q) % 3
    r = P.shape[0]
    R = range(r)
    return {
        "a": all(P[i].sum() % 3 == 2 for i in R),
        "b": all((P[i] ** 2 * P[j]).sum() % 3 == 0 for i in R for j in R if i != j),
        "c": all((P[i] * P[j] * P[k]).sum() % 3 == 0
                 for i, j, k in itertools.combinations(R, 3)),
        "d": q.sum() % 3 != 0,
        "e": all((q ** 2 * P[i]).sum() % 3 == 0 for i in R),
        "f": all((q * P[i] * P[j]).sum() % 3 == 0 for i in R for j in R),
    }


def is_projective_P(P):
    """(g): columns of P have weight >= 2 and are pairwise non-parallel."""
    P = np.array(P) % 3
    cols = [tuple(int(x) for x in P[:, l]) for l in range(P.shape[1])]
    if any(sum(1 for x in c if x) < 2 for c in cols):
        return False
    canon = set()
    for c in cols:
        neg = tuple((-x) % 3 for x in c)
        key = min(c, neg)
        if key in canon:
            return False
        canon.add(key)
    return True

# ---------------------------------------------------------------------------
# The search
# ---------------------------------------------------------------------------

class Searcher:
    """Backtracking over the rows p_1 <= p_2 <= ... <= p_r of P for fixed (r, m)."""

    def __init__(self, r, m, require_projective=True, stop_at_first=False):
        self.r, self.m = r, m
        self.require_projective = require_projective
        self.stop_at_first = stop_at_first
        V = all_vectors(m)
        keep = V.sum(axis=1) % 3 == 2                      # condition (a)
        self.V = V[keep]                                   # still in lex order
        self.Vnz = (self.V != 0).astype(np.int64)
        # desc bit l (l >= 1) set iff v_l < v_{l-1}
        desc = np.zeros(len(self.V), dtype=np.int64)
        for l in range(1, m):
            desc |= (self.V[:, l] < self.V[:, l - 1]).astype(np.int64) << l
        self.desc = desc
        self.nodes = 0
        self.solutions = []

    def _candidates(self, q, rows, prev_index):
        """Indices into self.V of admissible next rows."""
        m = self.m
        prefix = np.vstack([q] + rows)                      # rows fixed so far
        # columns l, l-1 in the same block iff equal in every fixed row
        nonbreak = 0
        for l in range(1, m):
            if np.array_equal(prefix[:, l], prefix[:, l - 1]):
                nonbreak |= 1 << l
        ok = (self.desc & nonbreak) == 0                    # column order
        ok[:prev_index] = False                             # row order (lex = index)
        idx = np.nonzero(ok)[0]
        if len(idx) == 0:
            return idx
        lin = [(q != 0).astype(np.int64)]                   # (e)
        sup = [q]                                           # (f), i = j
        for j, pj in enumerate(rows):
            lin.append((pj != 0).astype(np.int64))          # (b) sum_{supp p_j} p = 0
            sup.append(pj)                                  # (b) sum_{supp p} p_j = 0
            lin.append(q * pj)                              # (f), i != j
            for pk in rows[j + 1:]:
                lin.append(pj * pk)                         # (c)
        L = np.array(lin) % 3
        S = np.array(sup) % 3
        C = self.V[idx]
        good = ((C @ L.T) % 3 == 0).all(axis=1)
        good &= ((self.Vnz[idx] @ S.T) % 3 == 0).all(axis=1)
        return idx[good]

    def _rec(self, q, rows, prev_index):
        self.nodes += 1
        if len(rows) == self.r:
            P = np.array(rows)
            if (not self.require_projective) or is_projective_P(P):
                self.solutions.append((q.copy(), P.copy()))
                return self.stop_at_first
            return False
        for k in self._candidates(q, rows, prev_index):
            if self._rec(q, rows + [self.V[k]], k):
                return True
        return False

    def q_vectors(self):
        """Sorted q = (0^z 1^o 2^t) with o - t = 1 mod 3 (i.e. sum q = 1)."""
        m = self.m
        for z in range(m + 1):
            for o in range(m - z + 1):
                t = m - z - o
                if (o - t) % 3 == 1:
                    yield np.array([0] * z + [1] * o + [2] * t, dtype=np.int64)

    def run(self):
        for q in self.q_vectors():
            if self._rec(q, [], 0):
                break
        return self.solutions

# ---------------------------------------------------------------------------
# Double-lex normal form (used to test that a known code survives the pruning)
# ---------------------------------------------------------------------------

def double_lex_form(q, P):
    """Normalise sum(q) = 1, then alternately sort columns of [q;P] and rows of P
    until stable. The result satisfies the search's ordering constraints."""
    q = np.array(q) % 3
    P = np.array(P) % 3
    if q.sum() % 3 == 2:
        q = (-q) % 3
    M = np.vstack([q, P])
    while True:
        cols = sorted(range(M.shape[1]), key=lambda l: tuple(M[:, l]))
        M2 = M[:, cols]
        rows = sorted(range(1, M2.shape[0]), key=lambda i: tuple(M2[i]))
        M2 = np.vstack([M2[0], M2[rows]])
        if np.array_equal(M2, M):
            return M[0], M[1:]
        M = M2


def search_accepts(q, P):
    """True iff the search tree contains the path q, p_1, ..., p_r."""
    r, m = P.shape
    s = Searcher(r, m)
    index = {tuple(v): k for k, v in enumerate(s.V)}
    rows, prev = [], 0
    for i in range(r):
        cand = set(int(k) for k in s._candidates(q, rows, prev))
        k = index.get(tuple(P[i]))
        if k is None or k not in cand:
            return False
        rows.append(s.V[k])
        prev = k
    return is_projective_P(P)

# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def to_reduced(h, A):
    """Bring (h, A) with A projective to (q, P): A = [I|P], h = (0|q)
    after a column permutation and row operations."""
    RA, piv = rref_mod3(A)
    r = len(RA)
    n = A.shape[1]
    perm = piv + [l for l in range(n) if l not in piv]
    RA = RA[:, perm]
    hh = np.array(h)[perm] % 3
    hh = (hh - sum(hh[i] * RA[i] for i in range(r))) % 3
    assert (RA[:, :r] == np.eye(r, dtype=np.int64)).all() and not hh[:r].any()
    return hh[r:], RA[:, r:]


def cyclic19_code():
    """cyclic [[19,1,3]]_3 code of arXiv:2610.08012: puncture the length-20 cyclic code."""
    g = [0] * 15
    for e, c in [(0, 2), (1, 2), (2, 1), (3, 1), (4, 2), (6, 2), (8, 1), (10, 1),
                 (11, 2), (12, 2), (13, 1), (14, 1)]:
        g[e] = c
    N = 20
    W = np.array([[0] * i + g + [0] * (N - 15 - i) for i in range(N - 14)]) % 3
    R, _ = rref_mod3(W)
    gi = [i for i in range(len(R)) if R[i, 0]][0]
    h = np.delete(R[gi], 0)
    A = np.delete(np.array([R[i] for i in range(len(R)) if i != gi]), 0, axis=1)
    return h, A


def naive_relaxed_orbits(r, m):
    """All (q, P) satisfying (a)-(f) with sum q = 1, by brute force, up to
    column permutations and row permutations of P (canonical = lex-min)."""
    rows = [np.array(v) for v in itertools.product(range(3), repeat=m) if sum(v) % 3 == 2]
    qs = [np.array(v) for v in itertools.product(range(3), repeat=m) if sum(v) % 3 == 1]
    out = set()
    for comb in itertools.product(range(len(rows)), repeat=r):
        P = np.array([rows[k] for k in comb])
        c = reduced_conditions(np.ones(m, dtype=np.int64), P)
        if not (c["a"] and c["b"] and c["c"]):
            continue
        for q in qs:
            if all(reduced_conditions(q, P).values()):
                out.add(canonical(q, P))
    return out


def canonical(q, P):
    m = len(q)
    best = None
    for s in itertools.permutations(range(m)):
        key = (tuple(int(q[x]) for x in s),
               tuple(sorted(tuple(int(row[x]) for x in s) for row in P)))
        if best is None or key < best:
            best = key
    return best


def fmt(v):
    return "".join(str(int(x)) for x in v)


def print_code(q, P, out):
    h, A = code_from_reduced(q, P)
    r = P.shape[0]
    out(f"    h  : {fmt(h[:r])} {fmt(h[r:])}")
    for i in range(r):
        out(f"    a{i+1:<2}: {fmt(A[i, :r])} {fmt(A[i, r:])}")


def self_tests(out):
    out("=== Self-test 1: cyclic [[19,1,3]]_3 code of arXiv:2610.08012 ===")
    h, A = cyclic19_code()
    out(f"  n = {len(h)}, dim A = {rank_mod3(A)}, lambda = {int((h ** 3).sum() % 3)}")
    out(f"  triorthogonality violations: {len(definition5_violations(h, A))}")
    out(f"  d_Z (brute force) = {distance_Z(h, A)}")
    q, P = to_reduced(h, A)
    out(f"  reduced form: r = {P.shape[0]}, m = {P.shape[1]}")
    out(f"  conditions (a)-(f): {reduced_conditions(q, P)}")
    out(f"  (g) A projective: {is_projective_P(P)}")
    q2, P2 = double_lex_form(q, P)
    ok = search_accepts(q2, P2)
    out(f"  double-lex form survives every pruning step of the search: {ok}")
    assert ok

    out("")
    out("=== Self-test 2: relaxed search (no projectivity) vs naive brute force ===")
    for r, m in [(2, 3), (2, 4), (2, 5), (3, 4), (3, 5)]:
        t0 = time.time()
        B = naive_relaxed_orbits(r, m)
        S = Searcher(r, m, require_projective=False).run()
        C = {canonical(q, P) for q, P in S}
        out(f"  (r,m)=({r},{m}): brute-force orbits {len(B):3d}, search orbits {len(C):3d}, "
            f"equal: {B == C}  [{time.time() - t0:.1f}s]")
        assert B == C
    out("")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmax", type=int, default=14, help="exhaustive up to this length")
    ap.add_argument("--no-15", action="store_true", help="skip the n = 15 existence search")
    ap.add_argument("--no-selftest", action="store_true")
    ap.add_argument("--log", default="search_output.txt")
    args = ap.parse_args()

    logf = open(args.log, "w")

    def out(s=""):
        print(s)
        logf.write(s + "\n")
        logf.flush()

    out("Qutrit triorthogonal codes (cubic conditions only), d >= 3")
    out(f"python {sys.version.split()[0]}, numpy {np.__version__}")
    out("")
    if not args.no_selftest:
        self_tests(out)

    out(f"=== Exhaustive search, n <= {args.nmax} ===")
    out("  Cases with m < r+1 are excluded by dim(A*A) <= n-r-1;")
    out("  cases with m > (3^r-1)/2 - r are excluded by projectivity.")
    out(f"  {'n':>3} {'r':>3} {'m':>3} {'status':>28} {'nodes':>9} {'codes':>6} {'time':>8}")
    total_codes = 0
    for n in range(2, args.nmax + 1):
        for r in range(1, n):
            m = n - r
            maxpts = (3 ** r - 1) // 2 - r
            if m < r + 1:
                status = "excluded (m < r+1)"
                out(f"  {n:3d} {r:3d} {m:3d} {status:>28}")
                continue
            if m > maxpts:
                status = f"excluded (m > {maxpts})"
                out(f"  {n:3d} {r:3d} {m:3d} {status:>28}")
                continue
            t0 = time.time()
            s = Searcher(r, m)
            sols = s.run()
            total_codes += len(sols)
            out(f"  {n:3d} {r:3d} {m:3d} {'searched':>28} {s.nodes:9d} {len(sols):6d} "
                f"{time.time() - t0:7.1f}s")
    out(f"  RESULT: {total_codes} codes with n <= {args.nmax}.")
    out("")

    if not args.no_15:
        out("=== n = 15: search for one example ===")
        n = 15
        for r in range(1, n):
            m = n - r
            if m < r + 1 or m > (3 ** r - 1) // 2 - r:
                continue
            t0 = time.time()
            s = Searcher(r, m, stop_at_first=True)
            sols = s.run()
            out(f"  r={r} m={m}: {'found' if sols else 'none'} "
                f"(nodes {s.nodes}, {time.time() - t0:.1f}s)")
            if sols:
                q, P = sols[0]
                h, A = code_from_reduced(q, P)
                out("  Example [[15,1,3]]_3 code (logical row h, stabilizer rows a_i):")
                print_code(q, P, out)
                out(f"  triorthogonality violations: {len(definition5_violations(h, A))}")
                out(f"  lambda = {int((h ** 3).sum() % 3)}")
                out(f"  d_Z (brute force) = {distance_Z(h, A)}")
                out(f"  d_X (brute force) = {distance_X(h, A)}")
                break
    logf.close()


if __name__ == "__main__":
    main()
