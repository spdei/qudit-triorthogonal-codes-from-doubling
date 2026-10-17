#!/usr/bin/env python3
"""Multi-output codes: the triorthogonal space of each code punctured at k coordinates (logical rows dual
to the punctured columns, so the transversal gate acts as a product of single-qudit gates U^{-c_s}).

    python3 multi_output.py                  # check the stored puncture sets (multi_output_*.json)
    python3 multi_output.py --search D28_3 ...   # rerun the randomized greedy search, writes
                                                 # multi_output_search_<names>.json

The check recomputes, for every stored puncture set, that it is valid and that the Z-distance of the
punctured code is exactly the stored one, and asserts the multi-output table of the paper and the yield
parameters log_3(n/k) = 2.37, 2.24 and 2.46 quoted for [[27,2,3]]_3, [[35,3,3]]_3 and [[15,1,3]]_3.
The check takes a few minutes, mostly for [[118,1,8]]_3."""
import os, sys, json, glob, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from punct import unpuncture, greedy, valid, J
from dz import dZ

# space -> ([[n,k,d]] for d >= 2, [[n,k,d]] for d >= 3), as in the paper's table
TABLE = {'D15_3': ((13, 3, 2), (15, 1, 3)), 'T19_3': ((16, 4, 2), (19, 1, 3)),
         'D28_3': ((24, 5, 2), (27, 2, 3)), 'D37_3': ((30, 8, 2), (35, 3, 3)),
         'D72_3': ((58, 15, 2), (70, 3, 3)), 'D118_3': ((93, 26, 2), (115, 4, 3)),
         'D33_5': ((25, 9, 2), (31, 3, 3)), 'D20_7': ((16, 5, 2), (19, 2, 3))}

def search(names):
    out = {}
    for nm in names:
        p, H, c = unpuncture(J[nm]); r, n = H.shape
        for tgt, tr in ((2, 60), (3, 120)):
            S = greedy(H, p, tgt, trials=tr if n < 60 else 20, seed=1)
            d = dZ(H, p, S, maxw=4 if n > 60 else 5)
            out[f"{nm}_d{tgt}"] = dict(space=nm, p=p, n=n - len(S), k=len(S), d=d, S=[int(s) for s in S])
            print(f"{nm} target>={tgt}: [[{n-len(S)},{len(S)},{d}]]_{p} S={S}", flush=True)
    json.dump(out, open(os.path.join(HERE, 'multi_output_search_' + '_'.join(names) + '.json'), 'w'), indent=1)

def check():
    stored = {}
    for f in sorted(glob.glob(os.path.join(HERE, 'multi_output_*.json'))):
        if 'search' not in os.path.basename(f): stored.update(json.load(open(f)))
    fails = 0
    for nm, (c2, c3) in TABLE.items():
        p, H, c = unpuncture(J[nm]); r, n = H.shape
        for tgt, want in ((2, c2), (3, c3)):
            e = stored[f"{nm}_d{tgt}"]; S = e['S']
            d = dZ(H, p, S, maxw=tgt)          # exact if <= tgt, else tgt + 1
            got = (n - len(S), len(S), d)
            ok = valid(H, p, S) and got == want and (e['n'], e['k'], e['d']) == want
            fails += not ok
            print(f"{J[nm]['parameters']:>14} space, d>={tgt}: [[{got[0]},{got[1]},{got[2]}]]_{p}  "
                  f"{'OK' if ok else 'MISMATCH'}", flush=True)
    y = {f"[[{n},{k}]]": round(math.log(n / k, 3), 2) for n, k in ((27, 2), (35, 3), (15, 1))}
    ok = list(y.values()) == [2.37, 2.24, 2.46]; fails += not ok
    print(f"yield parameters log_3(n/k): {y}  {'OK' if ok else 'MISMATCH'}")
    return fails

if __name__ == '__main__':
    if '--search' in sys.argv: search(sys.argv[sys.argv.index('--search') + 1:])
    else: sys.exit(1 if check() else 0)
