#!/usr/bin/env bash
# Phase 5 launcher (D050). From the repo root:
#     nohup bash scripts/run_holdout.sh --check2016 >/dev/null 2>&1 &   # pre-tag, any clean commit
#     git checkout freeze-v2 && nohup bash scripts/run_holdout.sh --holdout >/dev/null 2>&1 &
# Linear, then trees, then the six NN specs one at a time; one launcher at a time (results/.phase5.lock;
# ~13-14 GB per NN process on a 24 GB machine, D050 §4). A crash: wait for this launcher to exit,
# then re-run the same command at the same commit (D044: finished runs reused, unfinished resumed).
# Progress: timestamps in the log and `grep -c fold_done results/trials.jsonl` only — never ledger
# rows (they carry 2017+ validation scores). The Python code repeats every holdout gate.
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-}"
[[ "$MODE" == --check2016 || "$MODE" == --holdout ]] || { echo "usage: $0 --check2016|--holdout" >&2; exit 2; }
NAME=${MODE#--}
mkdir -p results
LOG=results/${NAME}_run.log
refuse() { echo "$(date '+%F %T') refusing: $*" | tee -a "$LOG" >&2; exit 1; }

mkdir results/.phase5.lock 2>/dev/null || refuse "another Phase 5 launcher is running (results/.phase5.lock)"
trap 'rmdir results/.phase5.lock 2>/dev/null || true' EXIT

STATUS=$(git status --porcelain -- . ':!results') || refuse "git status failed"
[[ -z "$STATUS" ]] || refuse "uncommitted changes"
FROZEN_SHA=dfffda9aada2ddddbd9a74ecf1acdabaa91ea74f40f38f039a83b796ec1d56fc  # D044
GOT=$(shasum -a 256 data/processed/panel_datashare.parquet | cut -d' ' -f1) || refuse "cannot hash the frozen panel"
[[ "$GOT" == "$FROZEN_SHA" ]] || refuse "frozen panel sha256 differs from D044"
if [[ $NAME == holdout ]]; then
  TAGS=$(git tag --points-at HEAD) || refuse "git tag failed"
  grep -qxE 'freeze-v2(\.1)?' <<<"$TAGS" || refuse "HEAD is not tag freeze-v2 or freeze-v2.1"
  [[ -f results/holdout_panel.sha256 && -f data/processed/panel_holdout.parquet ]] \
    || refuse "no holdout panel or lock (run build_panel_holdout.py --build)"
  WANT=$(cut -d' ' -f1 results/holdout_panel.sha256)
  GOT=$(shasum -a 256 data/processed/panel_holdout.parquet | cut -d' ' -f1) || refuse "cannot hash the holdout panel"
  [[ -n "$WANT" && "$GOT" == "$WANT" ]] || refuse "holdout panel differs from its lock"
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
