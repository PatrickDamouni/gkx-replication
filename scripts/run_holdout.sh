#!/usr/bin/env bash
# Phase 5 launcher (D050). From the repo root:
#     nohup bash scripts/run_holdout.sh --check2016 >/dev/null 2>&1 &   # pre-tag, any clean commit
#     git checkout freeze-v2 && nohup bash scripts/run_holdout.sh --holdout >/dev/null 2>&1 &
# Linear, then trees, then the six NN specs one at a time (concurrent NN processes are refused in
# Phase 5: ~13-14 GB each on a 24 GB machine). A crash: re-run the same command at the same commit
# (D044: finished runs reused, unfinished resumed). Progress: timestamps in the log and
# `grep -c fold_done results/trials.jsonl` only — never ledger rows (they carry 2017+ validation scores).
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-}"
[[ "$MODE" == --check2016 || "$MODE" == --holdout ]] || { echo "usage: $0 --check2016|--holdout" >&2; exit 2; }
NAME=${MODE#--}
mkdir -p results
LOG=results/${NAME}_run.log
refuse() { echo "$(date '+%F %T') refusing: $*" | tee -a "$LOG" >&2; exit 1; }

[[ "${NN_CONCURRENT:-0}" == 1 ]] && refuse "concurrent NN processes are not allowed in Phase 5 (D050 §4)"
test -z "$(git status --porcelain -- . ':!results')" || refuse "uncommitted changes"
FROZEN_SHA=dfffda9aada2ddddbd9a74ecf1acdabaa91ea74f40f38f039a83b796ec1d56fc  # D044
[[ "$(shasum -a 256 data/processed/panel_datashare.parquet | cut -d' ' -f1)" == "$FROZEN_SHA" ]] \
  || refuse "frozen panel sha256 differs from D044"
if [[ $NAME == holdout ]]; then
  git tag --points-at HEAD | grep -qE '^freeze-v2(\.[0-9]+)?$' || refuse "HEAD is not tag freeze-v2"
  [[ -f results/holdout_panel.sha256 ]] || refuse "no holdout panel lock (run build_panel_holdout.py --build)"
  [[ "$(shasum -a 256 data/processed/panel_holdout.parquet | cut -d' ' -f1)" == "$(cut -d' ' -f1 results/holdout_panel.sha256)" ]] \
    || refuse "holdout panel differs from its lock"
fi
df -k . | awk 'NR==2 && $4 < 30*1024*1024 { exit 1 }' || refuse "< 30 GB free disk"

export OMP_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=8 PYTHONUNBUFFERED=1  # pinned (D044)
caffeinate -dimsu -w $$ &

note() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
run() { /usr/bin/time -l uv run --locked python scripts/run_holdout.py "$MODE" "$@" >>"$LOG" 2>&1; }

note "start $NAME $(git describe --tags --always) $(git rev-parse --short HEAD)"
for family in linear trees; do
  note "family $family"
  run --family "$family" || { note "FAILED $family (re-run this command)"; exit 1; }
done
for spec in NN1 NN1-dm NN2 NN2-dm NN3 NN3-dm; do
  note "nn $spec"
  run --family nn --only "$spec" || { note "FAILED $spec (re-run this command)"; exit 1; }
done
grep -h "maximum resident set size" "$LOG" | tee -a "$LOG" || true
note "all $NAME predictions written; next: uv run --locked python scripts/phase5_report.py $MODE"
