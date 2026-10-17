# Short qudit triorthogonal codes from doubling

Shiroman Prakash and K. L. Ramya

Code and data for the paper of the same title (arXiv:XXXX.XXXXX), which describes the codes and the
computations. Most checks are routine linear algebra; the substantial computations are the exhaustive
searches in `exhaustive/` and the weight enumerators used for the distillation statistics.

## Running

Requires Python 3 with numpy and sympy, matplotlib and mpmath (for `distillation/`), and gcc.

```
./run_all.sh                                          # all checks that take minutes; about 30 min
nohup ./run_long.sh > /dev/null 2>&1 < /dev/null &    # long checks; logs in ~/qudit-triorthogonal-logs
```

Each script lists in its header what it checks, asserts it, and exits nonzero on failure;
`distillation/fig.py` only draws the figure. Times are for one core.

| command | time |
|---|---|
| `python3 tests/selftest.py` | 15 s |
| `python3 verify_codes.py` | seconds |
| `python3 build_all_codes.py` | 2 min |
| `python3 distillation/properties.py` | 1 min |
| `python3 distillation/seq.py` | 2 min |
| `python3 distillation/distillation_claims.py` | 3 min |
| `python3 distillation/ps_m2_check.py` | seconds |
| `python3 searches/Ad_doubled.py` | 1 min |
| `python3 searches/exactness_mod9.py` | seconds |
| `python3 searches/multi_output.py` | 3 min |
| `python3 searches/find_q49.py` | 1 min |
| `python3 searches/grs_codes.py` | 1 min |
| `python3 searches/exhaustive_claims.py CLAIM` | minutes each |
| `python3 searches/random_searches.py` | 30 min |
| `python3 searches/dsearch.py P N MAXW` | hours |
| `exhaustive/` (see `exhaustive/README.md`) | minutes to days |

## What is checked

| topic | scripts |
|---|---|
| shared code, against naive reimplementations | `tests/selftest.py` |
| construction, triorthogonality and distance of every code | `build_all_codes.py`, `verify_codes.py` |
| searches for $P$-codes | `searches/exhaustive_claims.py`, `searches/grs_codes.py`, `searches/random_searches.py`, `searches/dsearch.py` |
| code properties | `distillation/properties.py`, `searches/Ad_doubled.py`, `searches/exactness_mod9.py` |
| multi-output codes | `searches/multi_output.py` |
| shortest triorthogonal codes | `exhaustive/` |
| distillation | `distillation/`, `searches/find_q49.py` |

## Data

`codes/<name>.txt` and `codes/codes.json`: `P` $P$-code, `W` weighted $P$-code, `T` seed, `D` doubled
code, named `<letter><length>_<p>`. For a triorthogonal code the first row is the logical row; for a
$P$-code the rows span $S$. Spectral supports refer to a root of the first irreducible factor of
$\Phi_n$ over $\mathbb F_p$ returned by SymPy. `codes/C40_3.json` is from arXiv:2610.08012;
`codes/Q49_2.json` is a qubit code used for comparison.

## Citation

```bibtex
@misc{PrakashRamya2026doubling,
  title         = {Short qudit triorthogonal codes from doubling},
  author        = {Shiroman Prakash and K. L. Ramya},
  year          = {2026},
  eprint        = {XXXX.XXXXX},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph}
}
```

## See also

- [cyclic-triorthogonal-codes](https://github.com/spdei/cyclic-triorthogonal-codes) (arXiv:2610.08012)
- [standard-model-code](https://github.com/spdei/standard-model-code) (arXiv:2610.21879)

## License

MIT; see [LICENSE](LICENSE).
