#!/usr/bin/env bash
# Run every check that takes minutes (about 30 minutes in total); exits nonzero if any fails.
# Long searches are in run_long.sh.
cd "$(dirname "$0")" || exit 1
fail=0
step() { echo; echo "=== $*"; "$@" || { echo "*** FAILED: $*"; fail=1; }; }
tmp=$(mktemp -d)
step python3 tests/selftest.py
step python3 verify_codes.py
step python3 build_all_codes.py --out "$tmp"
step diff -r -x cache.json -x C40_3.json -x Q49_2.json "$tmp" codes
step python3 distillation/properties.py
step python3 distillation/seq.py
step python3 distillation/distillation_claims.py
step python3 distillation/ps_m2_check.py
step python3 distillation/fig.py
step python3 searches/Ad_doubled.py
step python3 searches/exactness_mod9.py
step python3 searches/multi_output.py
step python3 searches/find_q49.py
step python3 searches/grs_codes.py
step python3 searches/exhaustive_claims.py unweighted10 weighted7 p7r2 p7conic
step gcc -O2 -march=native -o exhaustive/triorth exhaustive/triorth.c
step gcc -O2 -o exhaustive/test_dcheck exhaustive/test_dcheck.c
step bash -c 'cd exhaustive && ./test_dcheck'
step python3 exhaustive/test_triorth.py
step python3 exhaustive/compare_python.py
step bash -c 'cd exhaustive && for a in "p3_n05-16.txt 3" "p5_n05-12.txt 5" "p7_weighted_n05-09.txt 7" "p7_weighted_n10.txt 7" "p7_unit_n05-08.txt 7" "p5_n11_all.txt 5" "p3_n15_all.txt 3"; do python3 test_triorth.py verify runs/${a% *} ${a#* } || exit 1; done | diff - runs/test_verify_logs.txt'
rm -rf "$tmp"
echo; [ $fail = 0 ] && echo "ALL CHECKS PASSED" || echo "SOME CHECKS FAILED"
exit $fail
