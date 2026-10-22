# Exhaustive search for short triorthogonal codes (p = 3, 5, 7)

Exhaustive search for triorthogonal $[[n,1,d]]_p$ codes, as described in the paper.

| file | content |
|---|---|
| `triorth.c` | the search, for $p\in\{3,5,7\}$ |
| `qutrit_triorth_search.py`, `compare_python.py` | independent implementation for $p=3$; comparison of node counts |
| `test_triorth.py`, `rerun_logs.py` | brute-force cross-checks of `triorth.c`; re-verification of the logs |
| `test_dcheck.c`, `known_codes.txt`, `mkinputs.py` | test of the $d\ge D$ check on known codes |
| `runs/` | logs |

## Usage

```
gcc -O2 -march=native -o triorth triorth.c
./triorth P N            # all (r, m) with r + m = N; for p = 7 all coefficient classes
./triorth 7 N -u         # p = 7, unit coefficients; a coefficient in the class +-k costs k qudits
./triorth P N -d 4       # require d >= 4 (brute force at the leaves)
./triorth P N -rm R M    # only the pair (r, m)
./triorth P N -v         # print every code (default: first 3 per (r, m))
python3 test_triorth.py                     # cross-checks, about 3 min
python3 test_triorth.py --long              # also p = 7, (r, m) = (2, 4), general coefficients
python3 test_triorth.py verify LOG P        # rebuild every code in LOG, check it and its distance
gcc -O2 -o test_dcheck test_dcheck.c && ./test_dcheck
python3 qutrit_triorth_search.py --nmax 14 --no-15   # p = 3, n <= 14, about 2 min
```

Each pair (r, m) prints `nodes codes time` (CPU seconds) when it finishes; "codes" counts double-lex
representatives, not equivalence classes. Pairs that cannot occur (m < r + 1, or more columns than
points of PG(r-1, p)) are skipped. Long runs can be split with one process per pair (`-rm`).
`known_codes.txt` is written by `mkinputs.py` from `../codes/codes.json`, plus the $[[16,1,4]]_5$
concatenation of $[[4,1,2]]_5$ with itself; `test_dcheck` checks that the $d\ge D$ test accepts each code
for $D\le d$ and rejects it for $D=d+1$.

## Logs

Each log records the runs of `triorth` for the lengths given (CPU times per pair and per length).

| logs | p | mode | lengths |
|---|---|---|---|
| `p3_n05-16.txt`, `p3_n17.txt`, `p3_n15_all.txt` (`-v`) | 3 | $d\ge3$ | 5–17 |
| `p5_n05-12.txt`, `p5_n11_all.txt` (`-v`) | 5 | $d\ge3$ | 5–12 |
| `p7_weighted_n05-09.txt`, `p7_weighted_n10.txt` | 7 | $d\ge3$ | 5–10 |
| `p7_unit_n05-08.txt`, `p7_unit_n09.txt`, `p7_unit_n10.txt` | 7 | $d\ge3$, `-u` | 5–10 |
| `d4_p3_n09-17.txt`, `d4_p3_n18.txt`, `d4_p3_n19.txt` | 3 | `-d 4` | 9–19 |
| `d4_p5_n07-12.txt`, `d4_p5_n13.txt`, `d4_p5_n14.txt` | 5 | `-d 4` | 7–14 |
| `d4_p7_weighted_n07-10.txt`, `d4_p7_weighted_n11.txt` | 7 | `-d 4` | 7–11 |
| `d4_p7_unit_n07-13.txt` | 7 | `-d 4 -u` | 7–13 |

The logs for $n=18,19$ ($p=3$), $n=13,14$ ($p=5$) and $n=11$ ($p=7$) were run one process per pair
(`-rm`) and concatenated; the longest, $p=5$, $n=14$, took 160,916 CPU seconds. `rerun_logs.py` reruns
the logs and compares node and code counts. `test_crosschecks.txt` and `test_verify_logs.txt` are the
output of `test_triorth.py` and of `test_triorth.py verify` on the logs containing codes.
