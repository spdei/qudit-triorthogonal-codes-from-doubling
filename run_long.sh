#!/usr/bin/env bash
# The long checks, one after another, each with its own log. Usage:
#   nohup ./run_long.sh [LOGDIR] > /dev/null 2>&1 < /dev/null &
# Logs go to LOGDIR (default ~/qudit-triorthogonal-logs, outside the repository); LOGDIR/summary.txt gets one
# line per check (PASS/FAIL and wall time) and is the file to look at first.
# Approximate CPU times are in README.md; --all for rerun_logs.py adds about 50 CPU hours.
cd "$(dirname "$0")" || exit 1
LOGDIR=${1:-$HOME/qudit-triorthogonal-logs}; mkdir -p "$LOGDIR"; S="$LOGDIR/summary.txt"
echo "started $(date -u +%Y-%m-%dT%H:%MZ)" > "$S"
gcc -O2 -march=native -o exhaustive/triorth exhaustive/triorth.c
run() {  # name command...
  local name=$1; shift; local t0=$(date +%s)
  "$@" > "$LOGDIR/$name.log" 2>&1; local rc=$?
  printf '%-22s %s  %6d s\n' "$name" "$([ $rc = 0 ] && echo PASS || echo FAIL)" $(( $(date +%s) - t0 )) >> "$S"
}
run weighted9_10      python3 searches/exhaustive_claims.py weighted9 weighted10
run random_searches   python3 searches/random_searches.py
run rerun_logs        python3 exhaustive/rerun_logs.py
run test_triorth_long python3 exhaustive/test_triorth.py --long
run cyclic_p7         python3 searches/dsearch.py 7 30 6
run cyclic_p5         python3 searches/dsearch.py 5 39 7
run cyclic_p3         python3 searches/dsearch.py 3 50 8
echo "finished $(date -u +%Y-%m-%dT%H:%MZ)" >> "$S"
