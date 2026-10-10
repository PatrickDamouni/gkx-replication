# Freeze checklist (before tag `freeze-v1` and the single 1987–2016 run, D027)

Everything here must be true at the tagged commit. Decisions are in `DECISIONS.md`; this is the
operational list only.

## Code and data
- [ ] All tests green: `uv run pytest -q -W error` and `GKX_TORCH=1 uv run pytest -q tests/nn`.
- [x] Panel sha256 recorded in D044 and checked by `scripts/run_test.sh`.
- [x] `uv.lock` pins lightgbm 4.7.0, torch 2.14.1 (D038); runs use `uv run --locked` (D044).
- [x] Frozen model list = `scripts/run_test.py` specs: 4 baselines (not trials) + 14 candidate
      variants (D034, D041, D043); NN seeds 0–4. `tests/test_frozen_list.py` ties it to the report.
- [x] Pre-specified analysis = `scripts/final_report.py`: headline NN3, confirmatory family with
      Holm, secondary tests, verdict printed mechanically (D045).
- [x] GKX Table 1 reference values committed (`d322049`, D044 disclosure).
- [ ] Frozen-runner dry run: `bash scripts/run_test.sh --rehearsal` at the candidate commit, all
      three families, NN arms concurrent; predictions bit-identical to the earlier rehearsal; peak
      RSS of the two NN processes ≤ 20 GB (else `NN_SERIAL=1`, D046);
      `final_report.py --rehearsal` runs.

## Determinism (D044)
- [x] LightGBM: `deterministic=True`, `force_row_wise=True`, `num_threads=8`, `seed=0` (trees.COMMON).
- [x] torch: `set_num_threads(4)`, `use_deterministic_algorithms(True)`, per-seed `manual_seed`.
- [x] `OMP_NUM_THREADS=8`, `VECLIB_MAXIMUM_THREADS=8` exported by `scripts/run_test.sh`.

## The run (only after the user's explicit go-ahead)
- [ ] `git tag freeze-v1 <candidate>`; `git checkout freeze-v1`. The launcher refuses any other HEAD,
      a dirty tree, a different panel, or < 10 GB free disk.
- [ ] AC power, lid open. `nohup bash scripts/run_test.sh >/dev/null 2>&1 &` (the script holds
      `caffeinate -dimsu` itself). Linear (~45 min) → trees (~6–7 h) → two NN processes (~11–14 h).
      Predictions only; no metric is printed (council 5).
- [ ] Watch only `results/test_run.log*` timestamps and `fold_done` counts — never metrics.
- [ ] Crash (D044): re-run the same command at the same commit. Finished runs are reused, unfinished
      ones resume from fold checkpoints; not a new look. A code change means a new commit logged as a
      deviation, with every earlier ledger row kept.
- [ ] Then `uv run --locked python scripts/final_report.py` once; commit `results/final_report.md`,
      `results/final_report_tests.csv` and the ledger.
