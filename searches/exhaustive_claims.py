#!/usr/bin/env python3
"""Exhaustive searches for (weighted) P-codes. Usage: python3 exhaustive_claims.py <claim> [<claim> ...]

  unweighted10  no unweighted qutrit P-code with d >= 3 and n < 10; the shortest has n = 10
                (small3.py, dim S = 2, 3; about 1 minute)
  weighted7     no weighted qutrit P-code with d >= 3 and n < 7 (wsearch2.py, dim S = 2, 3; seconds)
  weighted9     no weighted qutrit P-code with d >= 4 and n <= 9
                (wsearch2.py for dim S = 3, d4search.py for dim S = 4; about 10 minutes)
  weighted10    no weighted qutrit P-code with d >= 5 and n <= 10
                (wsearch2.py for dim S = 3, d4search.py for dim S = 4; about 25 minutes)
  p7r2          no weighted P-code with d >= 4 and dim S = 2 for p = 5, 7 (n = 5, 6; seconds)
  p7conic       finds a weighted [[7,1,4]]^P_7 code (t7.py, all 7-point configurations containing a
                fixed frame; about 1 minute)
  all           all of the above

Which dimensions suffice: a P-code has n >= 2 dim S + 1, and d <= dim S + 1 (the origin lies in the
affine hull of at most dim S + 1 of the points). An unweighted P-code also has n != 0 mod p, which rules
out n = 9, dim S = 4 for p = 3. Each claim exits nonzero if the search output contradicts it."""
import os, sys, ast, subprocess, itertools, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE)

def run(*a):
    print('   $ python3', ' '.join(a), flush=True)
    out = subprocess.run([sys.executable, *a], check=True, capture_output=True, text=True).stdout
    for l in out.splitlines(): print('     ' + l)
    return out

def unweighted10():
    lengths = []
    for r in ('2', '3'):
        lengths += [int(l.split()[1]) for l in run('small3.py', '3', r).splitlines() if l.startswith('n1=')]
    return min(lengths) == 10

def wbest(r, maxn):              # {d: shortest n} found by wsearch2.py (last line of its output)
    return ast.literal_eval(run('wsearch2.py', r, maxn).splitlines()[-1])

def none_found(out): return out.splitlines()[-1].strip() == 'done, found 0'

def weighted7(): return wbest('2', '6') == {} and wbest('3', '6') == {}
def weighted9(): return all(d < 4 for d in wbest('3', '9')) and none_found(run('d4search.py', '9', '4'))
def weighted10(): return all(d < 5 for d in wbest('3', '10')) and none_found(run('d4search.py', '10', '5'))

def p7r2():
    from wp import proj_points, codes_from_config
    rng = np.random.default_rng(0); ok = True
    for p in (5, 7):
        P = proj_points(p, 2)
        for n in (5, 6):
            found = any(codes_from_config(P[list(sel)], p, 4, rng)
                        for sel in itertools.combinations(range(len(P)), n))
            print(f'     p={p}, dim S=2, n={n}: weighted P-code with d>=4 exists: {found}'); ok &= not found
    return ok

def p7conic():
    return any(l.startswith('FOUND n 7 d 4') for l in run('t7.py', '7', '4').splitlines())

CLAIMS = dict(unweighted10=unweighted10, weighted7=weighted7, weighted9=weighted9, weighted10=weighted10,
              p7r2=p7r2, p7conic=p7conic)
if __name__ == '__main__':
    names = sys.argv[1:]
    if names == ['all']: names = list(CLAIMS)
    if not names or any(n not in CLAIMS for n in names): print(__doc__); sys.exit(2)
    fails = 0
    for n in names:
        print(n, flush=True); ok = CLAIMS[n](); fails += not ok
        print(f"{n}: {'OK' if ok else 'MISMATCH'}", flush=True)
    sys.exit(1 if fails else 0)
