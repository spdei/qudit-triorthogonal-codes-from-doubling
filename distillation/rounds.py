"""Distillation rounds and the search for optimal sequences (cost model of Prakash-Saha, Quantum 9, 1768).
Each round maps the input depolarizing strength delta to (delta_out, acceptance probability). The cost of
a sequence is prod_i n_i / (k_i P_acc,i); optimize() keeps, after each of at most R rounds, the Pareto
front of (delta, cost) and returns the cheapest sequence reaching delta_out <= 10^-x for each x."""
import os, sys, json, itertools, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from codes_lib import Code, ps_family
J = json.load(open(os.path.join(HERE, '..', 'codes', 'codes.json')))

class Fn:
    """A round given by a formula: f(delta) -> (delta_out, acceptance)."""
    def __init__(s, name, n, k, f): s.name, s.n, s.k, s.f = name, n, k, f
    def step(s, d): return s.f(d)

def doubled(name):
    c = J[name]; return Code(name, np.array(c['stabilizer_rows']), np.array([c['logical_row']]), 3)

def ps_published(m):
    """Prakash-Saha [[6m+2,3m-2,2]]_3 with the output error and acceptance formulas as published."""
    import mpmath as mp
    k = 3 * m - 2; n = 6 * m + 2
    def f(d):
        mp.mp.dps = 80; q = 1 - mp.mpf(d)
        Ps = (1 + 2 * q ** 6 + 6 * q ** (2 * k + 4)) / 9
        N = (1 + 2 * q ** 6 + 12 * q ** (4 + k) + 4 * q ** (3 + 2 * k) + 6 * q ** (2 * k + 4) + 2 * q ** (6 + 2 * k)) / 27
        return float(mp.mpf(1.5) * (1 - N / Ps)), float(Ps)
    return Fn(f"[[{n},{k},2]]_3 (published)", n, k, f)

def D118_leading():
    """[[118,1,8]]_3 from its leading output error A_8 eps^8 / 2^8 (A_8 = 245410) and acceptance (1-eps)^118."""
    def f(d):
        e = 2 * d / 3; return 1.5 * 245410 / 2 ** 8 * e ** 8, (1 - e) ** 118
    return Fn('D118_3 (leading order)', 118, 1, f)

# qubit rounds (delta = 2 eps)
def qubit15():
    pts = [v for v in itertools.product([0, 1], repeat=4) if any(v)]
    return Code('qubit 15-to-1', np.array(pts).T, np.ones((1, 15), int), 2)
def mek(d):
    """10-to-2 code, exact polynomials of Meier, Eastin and Knill."""
    e = d / 2
    a = 1 - 10*e + 58*e**2 - 192*e**3 + 400*e**4 - 544*e**5 + 480*e**6 - 256*e**7 + 64*e**8
    u = 9*e**2 - 56*e**3 + 160*e**4 - 256*e**5 + 240*e**6 - 128*e**7 + 32*e**8
    return 2 * u / a, a
def bravyi_haah(k):
    """[[3k+8,k,2]] code, leading-order output error (1+3k) eps^2 and acceptance (1-eps)^n."""
    n = 3 * k + 8
    def f(d):
        e = d / 2; acc = (1 - e) ** n; return 2 * (1 + 3 * k) * e * e / acc, acc
    return Fn(f"BH [[{n},{k},2]]", n, k, f)
def qubit49():
    q = json.load(open(os.path.join(HERE, '..', 'codes', 'Q49_2.json')))['Q49_2']
    return Code('qubit [[49,1,5]]', np.array(q['stabilizer_rows']), np.array([q['logical_row']]), 2)

def code_sets(with49=True):
    fam = [ps_family(m) for m in range(1, 13)]
    new = [doubled('D15_3'), doubled('D28_3'), doubled('D37_3')]
    qubit = [qubit15(), Fn('MEK 10-to-2', 10, 2, mek)] + [bravyi_haah(k) for k in range(2, 41, 2)]
    if with49: qubit.append(qubit49())
    return fam, new, qubit

def optimize(rounds_codes, d0=0.02, R=5, xs=range(1, 31)):
    front = [(d0, 1.0, ())]; allst = list(front)
    for r in range(R):
        new = []
        for d, c, seq in front:
            for code in rounds_codes:
                dout, acc = code.step(d)
                if not (0 < dout < d) or acc <= 0: continue
                new.append((dout, c * code.n / (code.k * acc), seq + (code,)))
        new.sort(key=lambda t: (t[0], t[1]))
        pf = []      # Pareto front: keep a state only if it has lower delta than every cheaper one
        for t in sorted(new, key=lambda t: t[1]):
            if not pf or t[0] < min(x[0] for x in pf): pf.append(t)
        front = pf; allst += pf
    res = {}
    for x in xs:
        ok = [t for t in allst if t[0] <= 10.0 ** (-x)]
        res[x] = min(ok, key=lambda t: t[1]) if ok else None
    return res
