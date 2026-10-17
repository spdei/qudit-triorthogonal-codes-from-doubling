#!/usr/bin/env python3
"""Rerun the searches recorded in runs/ and compare everything except CPU times: the node and code counts
of each (r, m) pair and the code count of each length. Exits nonzero on any difference.
    python3 rerun_logs.py              # the logs that take minutes (d >= 3, and d >= 4 up to n = 17, 12, 10, 13)
    python3 rerun_logs.py --all        # also the long d >= 4 logs (about 50 CPU hours, mostly p = 5, n = 14)
Needs ./triorth (gcc -O2 -march=native -o triorth triorth.c)."""
import os, re, sys, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); BIN = os.path.join(HERE, 'triorth')
# log, p, lengths, flags
SHORT = [('p3_n05-16.txt', 3, range(5, 17), []), ('p3_n17.txt', 3, [17], []),
         ('p5_n05-12.txt', 5, range(5, 13), []),
         ('p7_weighted_n05-09.txt', 7, range(5, 10), []), ('p7_weighted_n10.txt', 7, [10], []),
         ('p7_unit_n05-08.txt', 7, range(5, 9), ['-u']), ('p7_unit_n09.txt', 7, [9], ['-u']),
         ('p7_unit_n10.txt', 7, [10], ['-u']),
         ('d4_p3_n09-17.txt', 3, range(9, 18), ['-d', '4']), ('d4_p5_n07-12.txt', 5, range(7, 13), ['-d', '4']),
         ('d4_p7_weighted_n07-10.txt', 7, range(7, 11), ['-d', '4']),
         ('d4_p7_unit_n07-13.txt', 7, range(7, 14), ['-u', '-d', '4'])]
LONG = [('d4_p3_n18.txt', 3, [18], ['-d', '4']), ('d4_p5_n13.txt', 5, [13], ['-d', '4']),
        ('d4_p5_n14.txt', 5, [14], ['-d', '4']), ('d4_p7_weighted_n11.txt', 7, [11], ['-d', '4'])]
PAIR = re.compile(r'r=(\d+) m=(\d+): nodes=(\d+) codes=(\d+)'); TOT = re.compile(r'TOTAL p=(\d) n=(\d+) codes=(\d+)')
def facts(text):
    pairs = {m.groups()[:2]: m.groups()[2:] for m in PAIR.finditer(text)}
    tots = {}
    for m in TOT.finditer(text): tots[m.group(2)] = tots.get(m.group(2), 0) + int(m.group(3))
    return pairs, tots
fails = 0
for log, p, ns, flags in SHORT + (LONG if '--all' in sys.argv else []):
    out = ''.join(subprocess.run([BIN, str(p), str(n), *flags], capture_output=True, text=True, check=True).stdout
                  for n in ns)
    (lp, lt), (op, ot) = facts(open(os.path.join(HERE, 'runs', log)).read()), facts(out)
    ok = lt == ot and all(op.get(k) == v for k, v in lp.items())
    fails += not ok
    print(f"{log:28s} {len(lp):3d} pairs, {len(lt):2d} lengths: {'same' if ok else 'DIFFERENT'}", flush=True)
sys.exit(1 if fails else 0)
