#!/usr/bin/env python3
"""The p = 3 search in triorth.c reproduces the node counts of the independent Python implementation
qutrit_triorth_search.py, for every pair (r, m) with n = r + m <= 14. Exits nonzero on any difference.
    python3 compare_python.py      # about 2 minutes; needs ./triorth"""
import os, re, sys, subprocess, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
with tempfile.TemporaryDirectory() as tmp:
    py = subprocess.run([sys.executable, os.path.join(HERE, 'qutrit_triorth_search.py'), '--nmax', '14', '--no-15',
                         '--log', os.path.join(tmp, 'log.txt')], capture_output=True, text=True, check=True).stdout
P = {(int(a), int(b)): (int(c), int(d)) for n, a, b, c, d in
     re.findall(r'^\s*(\d+)\s+(\d+)\s+(\d+)\s+searched\s+(\d+)\s+(\d+)', py, re.M)}
C = {}
for n in range(5, 15):
    out = subprocess.run([os.path.join(HERE, 'triorth'), '3', str(n)], capture_output=True, text=True, check=True).stdout
    C.update({(int(r), int(m)): (int(a), int(b)) for r, m, a, b in re.findall(r'r=(\d+) m=(\d+): nodes=(\d+) codes=(\d+)', out)})
ok = P == C and len(P) > 0
print(f"{len(P)} pairs (r, m) with n <= 14; node and code counts {'identical' if ok else 'DIFFERENT'}")
sys.exit(0 if ok else 1)
