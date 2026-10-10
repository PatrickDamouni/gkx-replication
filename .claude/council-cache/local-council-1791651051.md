# Local council 8 — final pre-freeze review

> **Local council** — these perspectives all come from Claude playing different
> roles, not from different AI vendors. Treat agreement as a shared starting
> point to pressure-test, not as independent confirmation.

Reviewed at HEAD a26da9a: scripts/run_test.py, run_test.sh, final_report.py (with the PROPOSED
HEADLINE = NN3 and PAIRS), src/gkx/evaluate.py, docs/freeze_checklist.md, DECISIONS D027–D043,
results/final_report_rehearsal.md. Members (`--roles=devil,simplicity,performance,maintainability`).
No member ran the test-run scripts or touched data after 1986.

---

## 🗳️ Maintainability Advocate
**Position.** Not freeze-ready for auditability: the harness is solid and tested, but the scripts
that will run have never run together, have no tests, and cite a D044 that doesn't exist.
**Blockers.** D044 cited 3× (panel sha256, crash rule, thread pins) but DECISIONS stops at D043;
D038 still says torch 8 threads (nn.py uses 4). HEADLINE/PAIRS still marked "PROPOSED" and in no
decision. D006's seed spread never computed. Prereg's "R² CI excluding zero" and "beats baseline"
not computed exactly. **`paper/gkx_table1.csv` is git-ignored and was never committed** (the
`reference-v1` tag holds only `.gitkeep`). Panel hash not in the ledger; final_report doesn't verify
`preds_sha256`, refuse a dirty tree, or record its own HEAD. **Crash-resume is wrong:** a finished
run is never reused, so re-running the launcher after an NN crash re-fits every linear and tree model
and writes new test-window "started" rows. The frozen entry point never ran; rehearsal predictions
come from 6 commits; the report header's "Commit 1e6e42a" misleads. No test ties run_test specs to
final_report's MODELS/BASELINES/HEADLINE/PAIRS.
**Risks.** No environment recorded (versions, uv.lock hash, OS, thread env); calling run_test.py
directly skips run_test.sh's pins and guards; bit-identity holds only on the same machine. Licensed
CRSP values in prediction files → hashes are what an outside reviewer can check. Nice-to-haves:
stale README; require the `freeze-v1` tag; `uv run --locked`; D034's N≈11 vs 14 in code.
**Confidence.** high.

## 🗳️ Simplicity Champion
**Position.** The analysis is close to minimal; three cheap blockers, no new machinery.
**Blockers.** (1) The primary prereg criterion (best model's R² > 0 with CI excluding zero) is never
computed — add `paired_r2_ci(base, "zero", HEADLINE)`. (2) Write the verdict rule: raw-target pairs
decide "replicates" at DM p < 0.05; ΔR² CI is effect size; say whether NN and trees must both win;
`-dm` pairs are secondary and can never turn a raw failure into "replicates". (3) The frozen runner
never ran: linear and tree rehearsal through `run_test.py` at the candidate commit (~45 min), then
`final_report.py --rehearsal`. (4) Write D044.
**Risks.** Success can pass on the level alone (rehearsal NN3 +0.481% vs train-mean +0.470%) — read
demeaned R² beside it; optionally add (NN3, train_mean). Seed spread code before the tag. Crash
re-fits finished models — let the launcher take family arguments. NN runtime likely 15–25 h → plan
two nights.
**Confidence.** high.

## 🗳️ Devil's Advocate
**Position.** Don't tag: the analysis cannot yet turn the run into a verdict, and the pair set is
built so a story comes out either way.
**Blockers.** Primary criterion not computed; "best model" undefined → fix HEADLINE = NN3 ex ante in
a D-entry and compute its R² CI vs zero; add NN3 vs train-mean as confirmatory. No outcome → verdict
mapping: five tests at 5% with no multiplicity rule; `-dm` pairs were added after rehearsal showed
them winning (NN3-dm p 0.0095; faithful NN3 t −1.06, GBRT+H p 0.989). Write D045: one confirmatory
pair (or Holm across a declared family); `-dm` secondary; print the verdict mechanically; list what
was visible when pairs were chosen. Keep NN3 (it was the worst raw NN in rehearsal — evidence against
cherry-picking). (ENet+H, OLS+H) cannot lose — replace with ENet+H vs OLS-3+H (Clark–West, nested).
ENet+H is the softest opponent — test NN3 against all four linear models with Holm, or a linear
composite. `-dm` pairs mix level and ranking — use demeaned-error DM for them. Report constant-
fallback counts. Frozen runner never ran: all three families at the candidate commit and check
bit-identity against existing rehearsal predictions. Provenance: D044 missing; check the panel hash
in run_test.sh and record it; `uv run --locked`.
**Risks.** Runtime ~3× optimistic (NN ~20–35 h) — plan 1.5–2 days; an unfinished morning invites
peeking. Rehearsal shaped D040–D043 and the pairs — frame 1975–86 as a development set (it is also
fold 1987's validation window); rehearsal findings may not transfer to 18–47-year windows. DSR
ignores abandoned rehearsal variants and treats 14 correlated variants as independent — state both.
**Confidence.** high on blockers; medium on runtime.

## 🗳️ Performance Optimizer
**Position.** Memory, disk and deadlock are fine; the schedule is the blocker — ~22–38 h serial
(best guess ~30 h), NN alone ~15–29 h.
**Key points (measured on tiled rehearsal data).** Linear ≈ 45 min total; trees ≈ 6–7 h (RF ≈ 2.3 h,
RF-dm ≈ 3–4 h, GBRT ≈ 25 min per arm); NN ≈ 3.7 h per arm, 6 arms ≈ 22 h (15–29 h depending on
epochs), only 4 of 12 cores busy. Peak RSS per process ≈ 9–13 GB; final_report ≈ 7–9 GB. Disk ≈ 4.2 GB
of outputs. Fix before the freeze: an arm filter in run_test.py and two concurrent NN processes (4+4
threads, ~20 GB) → NN ≈ 11–14 h, deterministic; never three, never trees alongside.
**Risks.** Sleep/hangup is the realistic failure: `caffeinate -dimsu`, AC, lid open, tmux/nohup (a
closed terminal kills `tee` → broken pipe). No in-model progress — watch `fold_done` rows. ENet/GLM
worst case bounded but large. Post-1986 sizes unmeasured (±10–20%). Thermals over ~30 h.
**Confidence.** medium.

---

## Synthesis — angles, not consensus

**Shared starting points (common prior — stress-test).** All four: D044 missing; the frozen runner
never ran end to end. Three: the prereg's primary criterion isn't computed and the verdict rule is
unwritten. Two: crash-resume re-fits finished models; runtime is 1.5–2 days, not overnight.

**Verified by the main session:** `paper/gkx_table1.csv` is git-ignored (`.gitignore:9:*.csv`) and
untracked — the `reference-v1` claim made to the user was wrong; values are verbatim from the paper
and will be committed with a disclosure.

**Genuine tension.** Simplicity (3 confirmatory pairs, minimal) vs Devil (NN3 against all four
linear models with Holm, replace the can't-lose OLS+H pair, demeaned-error DM for -dm pairs, NN3 vs
train-mean). Both agree the -dm pairs must be secondary and the rule must be fixed before the tag.

**Unique.** Performance: parallel NN processes cut ~22 h to ~11–14 h; `caffeinate -dimsu` + tmux.
Devil: keep NN3 precisely because it was the worst raw NN in rehearsal.
