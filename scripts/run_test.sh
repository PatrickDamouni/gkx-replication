#!/usr/bin/env bash
# The single test-window run (D027), or its dry run on the 1975-1986 schedule. From the repo root:
#     git checkout freeze-v1 && nohup bash scripts/run_test.sh >/dev/null 2>&1 &
#     nohup bash scripts/run_test.sh --rehearsal >/dev/null 2>&1 &    # dry run, any clean commit
# nohup (or tmux) keeps it alive if the terminal closes; all output goes to results/<mode>_run.log*.
# Linear, then trees, then the two NN arms as concurrent processes (D046; NN_SERIAL=1 runs them one
# after the other). LightGBM and torch never share a process. A crash is handled by re-running the
# same command at the same commit: finished runs are reused and unfinished ones resume from their
# fold checkpoints (D044). Progress: tail -f the log; fold_done rows in results/trials.jsonl.
set -euo pipefail
cd "$(dirname "$0")/.."

MODE=test
FLAG=""  # a string, not an array: macOS bash 3.2 + set -u rejects an empty "${a[@]}"
if [[ "${1:-}" == "--rehearsal" ]]; then MODE=rehearsal; FLAG=--rehearsal; fi
PANEL=data/processed/panel_datashare.parquet
PANEL_SHA=dfffda9aada2ddddbd9a74ecf1acdabaa91ea74f40f38f039a83b796ec1d56fc  # D044

refuse() { echo "refusing: $*" >&2; exit 1; }
test -z "$(git status --porcelain -- . ':!results')" || refuse "uncommitted changes"
if [[ $MODE == test ]]; then
  git tag --points-at HEAD | grep -qx freeze-v1 || refuse "HEAD is not tag freeze-v1"
fi
[[ "$(shasum -a 256 "$PANEL" | cut -d' ' -f1)" == "$PANEL_SHA" ]] || refuse "panel sha256 differs from D044"
df -k . | awk 'NR==2 && $4 < 10*1024*1024 { exit 1 }' || refuse "< 10 GB free disk"

export OMP_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=8 PYTHONUNBUFFERED=1  # pinned (D044)
caffeinate -dimsu -w $$ &  # no sleep while this script runs (D046)

LOG=results/${MODE}_run.log
mkdir -p results
note() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
# One family process; /usr/bin/time -l records its peak RSS (D046's 20 GB condition).
run() {
  local out=$1; shift
  /usr/bin/time -l uv run --locked python scripts/run_test.py $FLAG "$@" >>"$out" 2>&1
}

note "start $MODE $(git describe --tags --always) $(git rev-parse --short HEAD)"
for family in linear trees; do
  note "family $family"
  run "$LOG" --family "$family" || { note "FAILED $family (re-run this command)"; exit 1; }
done

RAW=NN1,NN2,NN3
DM=NN1-dm,NN2-dm,NN3-dm
if [[ "${NN_SERIAL:-0}" == 1 ]]; then
  note "family nn (serial)"
  run "$LOG.nn-raw" --family nn --only "$RAW" || { note "FAILED nn raw"; exit 1; }
  run "$LOG.nn-dm" --family nn --only "$DM" || { note "FAILED nn dm"; exit 1; }
else
  note "family nn (two concurrent processes)"
  run "$LOG.nn-raw" --family nn --only "$RAW" & raw=$!
  run "$LOG.nn-dm" --family nn --only "$DM" & dm=$!
  ok=1
  wait "$raw" || { note "FAILED nn raw"; ok=0; }
  wait "$dm" || { note "FAILED nn dm"; ok=0; }
  [[ $ok == 1 ]] || { note "re-run this command to resume"; exit 1; }
fi
grep -h "maximum resident set size" "$LOG" "$LOG.nn-raw" "$LOG.nn-dm" | tee -a "$LOG"
note "all $MODE predictions written; next: uv run --locked python scripts/final_report.py $FLAG"
