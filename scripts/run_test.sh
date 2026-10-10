#!/usr/bin/env bash
# The single test-window run (D027). Run from the repo root at the freeze tag:
#     git checkout freeze-v1 && caffeinate -i bash scripts/run_test.sh
# Three processes, cheapest first (LightGBM and torch never share one). A crash is resumed by
# re-running this same command at the same commit: evaluate() skips checkpointed folds (D044).
set -euo pipefail
cd "$(dirname "$0")/.."
test -z "$(git status --porcelain -- . ':!results')" || { echo "refusing: uncommitted changes"; exit 1; }
git describe --exact-match --tags HEAD >/dev/null 2>&1 || { echo "refusing: HEAD is not a tagged freeze"; exit 1; }
df -k . | awk 'NR==2 && $4 < 10*1024*1024 { print "refusing: < 10 GB free disk"; exit 1 }'
export OMP_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=8 PYTHONUNBUFFERED=1  # pinned (D044)
LOG=results/test_run.log
echo "$(date '+%F %T') start $(git describe --tags) $(git rev-parse --short HEAD)" | tee -a "$LOG"
for family in linear trees nn; do
  echo "$(date '+%F %T') family $family" | tee -a "$LOG"
  uv run python scripts/run_test.py --family "$family" 2>&1 | tee -a "$LOG"
done
echo "$(date '+%F %T') all predictions written; next: uv run python scripts/final_report.py" | tee -a "$LOG"
