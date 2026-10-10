# Freeze checklist (before tag `freeze-v1` and the single 1987–2016 run, D027)

Everything here must be true at the tagged commit. Decisions are in `DECISIONS.md`; this is the
operational list only.

## Code and data
- [ ] All tests green: `uv run pytest -q -W error` and `GKX_TORCH=1 uv run pytest -q tests/nn`.
- [ ] `data/processed/panel_datashare.parquet` sha256 recorded in D044.
- [ ] `uv.lock` pins lightgbm 4.7.0, torch 2.14.1 (D038).
- [ ] Frozen model list = `scripts/run_test.py` specs: 4 baselines (not trials) + 14 candidate variants
      (D034, D041, D043); NN seeds 0–4.
- [ ] Pre-specified analysis = `scripts/final_report.py`: headline model and ranking pairs approved.
- [ ] Dry runs done: every model rehearsed from clean commits, and `final_report.py --rehearsal`.

## Determinism
- [ ] LightGBM: `deterministic=True`, `force_row_wise=True`, `num_threads=8`, `seed=0` (trees.COMMON).
- [ ] torch: `torch.set_num_threads(4)`, `use_deterministic_algorithms(True)`, per-seed `manual_seed`.
- [ ] `OMP_NUM_THREADS=8`, `VECLIB_MAXIMUM_THREADS=8` set by `scripts/run_test.sh`.

## The run
- [ ] `git checkout freeze-v1` (clean tree and tag enforced by `run_test.sh` and `evaluate()`).
- [ ] ≥ 10 GB free disk (enforced), AC power, `caffeinate -i`.
- [ ] `bash scripts/run_test.sh` — linear → trees → nn, one process each; predictions only, no
      metrics printed (council 5).
- [ ] Crash rule (D044): re-run the same command at the same commit; `evaluate()` resumes from
      per-fold checkpoints under the same run_id. A re-run at the same commit is not a new look.
      Any code change after a crash means a new commit, logged as a deviation, with every earlier
      ledger row kept.
- [ ] Then `uv run python scripts/final_report.py` once; commit `results/final_report.md`.
