# Local council 9 — verify council-8 fixes before the freeze tag

> **Local council** — these perspectives all come from Claude playing different
> roles, not from different AI vendors. Treat agreement as a shared starting
> point to pressure-test, not as independent confirmation.

Reviewed commits a8b0c2a..01f0936 read-only while the frozen-runner rehearsal ran at 01f0936
(`--roles=devil,simplicity,performance,maintainability`). No member ran code or loaded data.

## 🗳️ Devil's Advocate
**Position.** Don't tag 01f0936: the prediction path is fine, the verdict machinery is not, and
`final_report.py:121` makes report bugs unfixable after the look. **Blockers.** (1)
`report.py:69` prints "partially replicates" at 0 of 7 rejections (test asserts it; D045 says
"some") — the most likely 30-year outcome is primary passes on level (NN3 0.481% vs train-mean
0.470% in rehearsal) with 0–1 rejections. (2) Holm treats NaN p as rejected (`report.py:47-52`);
NaN arises if NN3 equals the train-mean constant everywhere. (3) final_report never checks
`run_sha == freeze-v1`; a stray `run_test.py` at another commit poisons `_latest_runs`. (4) D046's
memory gate is measured on the 1986-truncated panel, so it passes vacuously. (5) "All rejected
under Holm" is too strict for a conjunction — intersection-union (each p ≤ 0.05) is the valid
test for "every leg holds"; Holm is for which legs hold. **Risks.** Any Claude session writing in
the repo during the run dirties the tree and kills the next spec (`.claude/` is tracked or
unignored); commit everything, then tag; ledger `fold_done` rows carry test-year `val_scores` —
only count them; test-mode branches of final_report never executed; re-run the launcher once
after the rehearsal to see 18 `reused` events; write the bit-identity pass rule before the NN
comparison; `refuse()` goes to /dev/null under nohup; label OLS-3+H swap as a prereg deviation.
**Confidence.** high on 1–3; medium on memory and IUT.

## 🗳️ Simplicity Champion
**Position.** Fixes are minimal; one real blocker — the zero-rejection verdict hole (+ its NaN
twin). Don't touch evaluate.py/run_test.py again. **Points.** Fix `report.py:69` (0 rejections →
"does not replicate", or require test 1); fail closed on NaN; report-side edits don't invalidate
the rehearsal — keep `git diff 01f0936 freeze-v1` to report/tests/docs and say so; re-run
`final_report --rehearsal` after. Leave the run_test.py shims alone. **Risks.** D046 gate measured
on small panel — say OOM is handled by resume, not prevented; concurrent ledger appends (one
O_APPEND write per line, final_report's full parse is the check); `run_test.sh:60` grep under
pipefail can exit non-zero after success → `|| true`; prereg deviation unlabelled.
**Confidence.** high.

## 🗳️ Performance Optimizer
**Position.** Nothing in code blocks; two operational blockers. **Points.** D046 memory gate
measures the wrong workload: rehearsal loads ~⅓ of the panel; estimated 10–13 GB per NN process at
test scale (panel 3.5, train/val copies 3.3, per-seed `train[features]` 2.5, float64→float32
2.4/1.3), consistent with council 8's tiled 9–13 GB → 20–26 GB for two on 24 GB with swap already
4.5/5 GB used. Run `NN_SERIAL=1` or knowingly accept swap. Disk: 15 GiB free on a 97%-full volume;
swap grows on the same volume; a torn ledger line makes `_previous`'s `json.loads` crash every
resume → free ≥ 40 GB, optionally raise the 10 GB gate. **Verified right:** resume key includes
sha; reuse costs one hash + read; RSS measurement captures the Python child (8.53 GB linear).
**Risks.** Swapping costs time not correctness, but a 30+ h run invites peeking; Spotlight/Time
Machine load. **Confidence.** medium (test-scale RSS estimated, not measured).

## 🗳️ Maintainability Advocate
**Position.** Nearly freeze-ready; the verdict path has a code/decision mismatch and a fail-open
Holm. **Blockers.** Same two as above (`report.py:69`, `report.py:50/67`). **Should fix before
tag (docs):** D045 must say it amends prereg line 25 / D010 ("OLS-all" → OLS-3+H); D046 claims a
serial-vs-concurrent check nothing records — add an addendum with run_id pairs, method, expected
identity, measured RSS, pass rule written before reading. **Nice-to-have:** pin `len(CONFIRMATORY)
== 7`, no overlap, no `-dm` in confirmatory in test_frozen_list; extend code-changed check to
pyproject/uv.lock; grep only this attempt's RSS lines; D046 tmux vs checklist nohup; regenerate the
stale committed rehearsal report. **Risks.** Report code has never run as committed; RSS rollup
(since verified fine); no written rule for an analysis bug found after the look — add to D044.
**Confidence.** high.

## Synthesis — angles, not consensus
**Shared starting points (common prior — stress-test).** All four: the verdict prints "partially
replicates" at 0 of 7 rejections, against D045's text; Holm/primary fail open on NaN; the
prediction path should not be touched (report-side fixes don't invalidate the rehearsal). Three:
D046's memory gate measured on the truncated panel; the OLS-3+H swap must be labelled as a prereg
amendment. **Main-session checks:** RSS rollup works (linear 8.53 GB logged); tree is clean; disk
15 GiB free, swap 4.5/5 GB used.
**Tensions.** Devil: Holm-for-all is the wrong test for "replicates" (use intersection-union) —
a change to a user-approved rule, so it goes to the user. Performance: run NNs serially (+8–11 h)
vs the user's approved concurrent schedule.
**Unique.** Devil: final_report must require the freeze-v1 commit; no Claude session may write in
the repo during the run; never tail the ledger (test-year val scores); re-run the launcher once to
see 18 `reused` events. Performance: disk headroom and torn-ledger risk. Maintainability: written
rule for an analysis bug after the look.
