#!/usr/bin/env python3
"""Optimal distillation sequences: the cost figure and cost table of the paper.

For input noise delta_in in {0.02, 0.1, 0.2} and up to five rounds, finds the cheapest sequence reaching
delta_out <= 10^-x for x = 1..30, with three code sets: the Prakash-Saha codes [[6m+2,3m-2,2]]_3
(m <= 12, exact), the same together with [[15,1,3]]_3, [[28,1,4]]_3 and [[37,1,5]]_3, and qubit codes
(15-to-1, 10-to-2, Bravyi-Haah [[3k+8,k,2]] with k <= 40, and [[49,1,5]]). Also runs the Prakash-Saha
codes with their published formulas, to check that the exact cost is never larger. Asserts the costs
and sequences of the cost table and writes seq_<delta>.pkl for fig.py.
    python3 seq.py              # all three noise levels, about 2 minutes"""
import os, sys, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from rounds import code_sets, optimize, ps_published

LABELS = ('PS', 'PS+new', 'qubit')
TARGETS = (10, 15, 20, 25)
# delta_in: {label: costs at TARGETS (None: not reached in five rounds)}, and the sequences with new codes
COST = {
    0.02: {'PS': [49, 98, 110, 153], 'PS+new': [49, 98, 110, 153], 'qubit': [179, 334, 574, 892]},
    0.1:  {'PS': [648, 1182, 2629, 27194], 'PS+new': [567, 1140, 1368, 2407],
           'qubit': [2708, 5486, 10968, 17101]},
    0.2:  {'PS': [19139, None, None, None], 'PS+new': [12978, 30050, 39158, 81050],
           'qubit': [148654, 348392, 1323317, 1323317]},
}
SEQ = {   # code lengths, round by round; None: the new codes are not used
    0.02: [None, None, None, None],
    0.1:  [[15, 26, 68, 68], [15, 50, 74, 74, 74], [15, 20, 50, 74, 74], [15, 14, 20, 32, 56]],
    0.2:  [[8, 15, 26, 68, 74], [15, 15, 74, 74, 74], [15, 15, 26, 38, 68], [15, 28, 74, 74, 74]],
}

def summary(res):
    """x -> (delta_out, cost, code names, code lengths) or None; picklable."""
    return {x: (t[0], t[1], [c.name for c in t[2]], [c.n for c in t[2]]) if t else None for x, t in res.items()}

if __name__ == '__main__':
    fam, new, qubit = code_sets()
    fam_pub = [ps_published(m) for m in range(1, 13)]
    newnames = {c.name for c in new}; fails = 0
    for d0 in (0.02, 0.1, 0.2):
        out = {lab: summary(optimize(cs, d0=d0)) for lab, cs in
               (('PS', fam), ('PS+new', fam + new), ('qubit', qubit), ('PS published', fam_pub))}
        pickle.dump(out, open(os.path.join(HERE, f'seq_{d0}.pkl'), 'wb'))
        print(f"delta_in = {d0}\n   target    " + "".join(f"{l:>12}" for l in LABELS) + "   sequence with new codes")
        for i, x in enumerate(TARGETS):
            row = [out[l][x] for l in LABELS]
            costs = [None if t is None else t[1] for t in row]
            ok = all((c is None) == (e is None) and (c is None or abs(c - e) <= 0.51)
                     for c, e in zip(costs, [COST[d0][l][i] for l in LABELS]))
            t = out['PS+new'][x]; uses = bool(set(t[2]) & newnames)
            ok &= (t[3] == SEQ[d0][i]) if uses else (SEQ[d0][i] is None)
            fails += not ok
            print(f"   1e-{x:<6}" + "".join(f"{'---' if c is None else round(c):>12}" for c in costs) +
                  f"   {', '.join(map(str, t[3])) if uses else 'new codes not used'}   {'OK' if ok else 'MISMATCH'}")
        # exact Prakash-Saha costs never exceed those from the published formulas
        for x in range(1, 31):
            e, pub = out['PS'][x], out['PS published'][x]
            if pub is not None:
                if not (e is not None and e[1] <= pub[1] * (1 + 1e-12)):
                    print(f'   published-formula cost lower than exact at 1e-{x}: MISMATCH'); fails += 1
        if d0 == 0.02:   # the figure: dashed and solid curves coincide
            same = all(out['PS'][x] == out['PS+new'][x] for x in range(1, 26))
            print(f"   PS and PS+new optimal sequences coincide for every target up to 1e-25: {same}")
            fails += not same
    print('all values agree with the paper' if not fails else f'{fails} MISMATCH(ES)')
    sys.exit(1 if fails else 0)
