#!/usr/bin/env python3
"""Statements about distillation made in the paper (the cost table itself is checked by seq.py):
  1. [[15,1,3]]_3 against the 15-qubit code: at equal infidelity, eps_out ~ 6.25 eps^3 against 35 eps^3,
     thresholds 0.230 against 0.141, acceptance 1 - 15 eps to leading order; at equal depolarizing
     strength, 2.78 delta^3 against 8.75 delta^3, thresholds 0.345 against 0.283, acceptance 0.905 and
     0.928 at delta = 0.01.
  2. Adding [[72,1,6]]_3 (exact) or [[118,1,8]]_3 (leading-order output error, acceptance (1-eps)^118)
     leaves every optimal qutrit sequence unchanged (targets 10^-1 .. 10^-25, three noise levels).
  3. The qubit [[49,1,5]] code is used only at delta_in = 0.2, for every target 10^-20 .. 10^-25, and
     lowers the qubit cost there by 19%.
  4. At delta_in = 0.1 the new codes lower the cost by 12% at 10^-10 and by a factor of eleven at
     10^-25; at delta_in = 0.02 qutrit sequences need 3.4 to 5.8 times fewer inputs than qubit ones.
    python3 distillation_claims.py      # about 3 minutes; needs gcc"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from codes_lib import Code, threshold
from rounds import code_sets, optimize, doubled, D118_leading, qubit15

fails = 0
def check(label, ok):
    global fails; fails += not ok; print(f"  {'OK      ' if ok else 'MISMATCH'} {label}", flush=True)
def r(x, nd): return round(x, nd)

print('1. 15-qutrit against 15-qubit code')
q3, q2 = doubled('D15_3'), qubit15()
for C, p, coef, thr, coefd, thrd, acc01 in ((q3, 3, 6.25, 0.230, 2.78, 0.345, 0.905),
                                             (q2, 2, 35, 0.141, 8.75, 0.283, 0.928)):
    e = 1e-5; c_eps = C.eps_stats(e)[0] / e ** 3; a_lead = (1 - C.eps_stats(e)[1]) / e
    t_eps = threshold(lambda x: C.eps_stats(x)[0], hi=(p - 1) / p - 1e-9)
    c_del = C.step(e)[0] / e ** 3; t_del = threshold(lambda x: C.step(x)[0], hi=1 - 1e-9)
    acc = C.step(0.01)[1]
    print(f"  {C.name}: eps_out/eps^3 = {c_eps:.3f}, eps* = {t_eps:.4f}, (1-P_acc)/eps = {a_lead:.3f}; "
          f"delta_out/delta^3 = {c_del:.3f}, delta* = {t_del:.4f}, P_acc(delta=0.01) = {acc:.4f}")
    check(f"{C.name} matches the values in the text",
          r(c_eps, 2) == coef and r(t_eps, 3) == thr and r(a_lead, 1) == 15 and r(c_del, 2) == coefd
          and r(t_del, 3) == thrd and r(acc, 3) == acc01)

fam, new, qubit = code_sets()
qubit_no49 = [c for c in qubit if '49' not in c.name]
d72, d118 = doubled('D72_3'), D118_leading()
XS = range(1, 26)
names = lambda t: None if t is None else [c.name for c in t[2]]
res = {}
for d0 in (0.02, 0.1, 0.2):
    res[d0] = {k: optimize(cs, d0=d0, xs=XS) for k, cs in
               (('PS', fam), ('PS+new', fam + new), ('+72', fam + new + [d72]), ('+118', fam + new + [d118]),
                ('qubit', qubit), ('qubit-49', qubit_no49))}

print('2. adding [[72,1,6]]_3 or [[118,1,8]]_3')
for d0 in res:
    R = res[d0]
    check(f"delta_in = {d0}: optimal sequences unchanged",
          all(names(R['PS+new'][x]) == names(R['+72'][x]) == names(R['+118'][x]) for x in XS))

print('3. qubit [[49,1,5]]')
used = {(d0, x) for d0 in res for x in XS if res[d0]['qubit'][x] and
        any('49' in c.name for c in res[d0]['qubit'][x][2])}
print(f"  used at (delta_in, x): {sorted(used)}")
check('used exactly at delta_in = 0.2 and x = 20..25', used == {(0.2, x) for x in range(20, 26)})
saves = [1 - res[0.2]['qubit'][x][1] / res[0.2]['qubit-49'][x][1] for x in (20, 25)]
print(f"  cost reduction at 1e-20, 1e-25: {saves[0]:.4f}, {saves[1]:.4f}")
check('reduction is 19%', all(round(100 * s) == 19 for s in saves))

print('4. gains')
R = res[0.1]; drop = 1 - R['PS+new'][10][1] / R['PS'][10][1]; fac = R['PS'][25][1] / R['PS+new'][25][1]
print(f"  delta_in = 0.1: reduction at 1e-10 = {100 * drop:.2f}%, factor at 1e-25 = {fac:.2f}")
check('12% at 1e-10 and a factor of eleven at 1e-25', round(100 * drop) == 12 and int(fac) == 11)
R = res[0.02]; ratios = [R['qubit'][x][1] / R['PS'][x][1] for x in (10, 15, 20, 25)]
print(f"  delta_in = 0.02: qubit/qutrit cost at 1e-10, -15, -20, -25: {', '.join(f'{q:.2f}' for q in ratios)}")
check('between 3.4 and 5.8', r(min(ratios), 1) == 3.4 and r(max(ratios), 1) == 5.8)

print('all statements agree with the paper' if not fails else f'{fails} MISMATCH(ES)')
sys.exit(1 if fails else 0)
