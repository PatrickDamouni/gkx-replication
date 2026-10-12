# Local council 11 — review of the single 1987–2016 test-window result (cb59143)

> **Local council** — these perspectives all come from Claude playing different
> roles, not from different AI vendors. Treat agreement as a shared starting
> point to pressure-test, not as independent confirmation.

`--roles=devil,simplicity,maintainability,performance`; read-only; no code run, no sealed data.

## 🗳️ Devil's Advocate
**Position.** Run is clean; the label "partially replicates" over-claims. It rests on one test
(NN3 vs train_mean, p 0.0134) that is not a GKX claim, fails Holm, and needs the 5-seed
ensemble. Lead with: on the top-1000 stocks a constant beats NN3 (0.445 vs 0.294), and GKX's
ranking does not reproduce (GBRT+H significantly worse than ENet+H, CI [−0.431, −0.004]).
**Points.** Provenance verified (18 test starts at 46c1584, clean, no test resumes). Report the
label under prereg / D045 / D047 / D048 (D047 → "level only"). "3 of 7" overstates: two passing legs
have ΔR² CIs including zero; `final_report.py:250-256` pairs raw ΔR² with CW and demeaned-DM p's;
`round(4)` prints 1e-9 as "0". Bottom-1000 R² is ~2× GKX for almost every model → suspect the target
build (Shumway delisting imputation D019, all-securities universe D020), unverified. This is a
reproduction on GKX's characteristics (D022), not independent data; OSAP arm (prereg Q1) not run.
-dm arms: never "beats GKX"; secondary. **Risks.** DM under fat tails (ENet vs OLS+H +170 pp → p
0.045); Phase 5 underpowered for test 1 (t ≈ 1.14 expected on 96 months) — write D049 before
unsealing, the only route to confirmatory -dm; OLS+H −170% is an unresolved gap, not "as in GKX";
DSR uninformative (ledger has 42 run names vs N = 14); README stale. **Phase 4:** headline rewrite
+ verdict-under-each-rule table; leave-one-seed-out test-1 p's; where R² lives (NYSE size, price ≥
$5, ex-delisting months, common shares); level vs ranking + Mincer–Zarnowitz; GBRT fallbacks; OLS+H
month contributions; OSAP arm; then stop. **Confidence.** high (delisting hypothesis untested).

## 🗳️ Simplicity Champion
**Position.** Honest and correct; the risk is now process volume. Cut the claim to one sentence and
one small table; no new rules on 1987–2016. **Points.** Lead with ΔR² vs constant 0.27% [0.045,
0.499]; single-seed NN3 (0.163) below constant (0.181); top-1000: constant wins; the one Holm
survivor is the weakest claim (ENet vs OLS-3, CW) with a CI covering zero; replace "3 of 7" with
"test 1 passed; ranking claims not supported"; write "< 1e-4" instead of "0". Can claim: 5-seed NN3
beats a constant on GKX's characteristics; ranking not confirmed. Cannot: independent data (D022),
-dm headline numbers, large-cap skill, strong DSR. **Phase 4:** one-page write-up (5-row table + one
CI figure); why -dm wins from the existing demeaned-R² column; GBRT shortfall from D041/D042 records;
OLS+H via D033; optional per-year ΔR² plot. One short D-entry freezing the Phase 5 family (+ at most
1–2 -dm tests). **Risks.** OSAP arm is a dangling commitment — run it or log deferral; superseded
D045/D047 text may be quoted — cite D048 only. **Confidence.** high.

## 🗳️ Maintainability Advocate
**Position.** Provenance unusually strong; the risk is meaning drift through a 600-line amendment
chain. **Points.** Under D047 (superseded ~30 min before the freeze) the same numbers read "does not
replicate (level only)" — say so. Claimable: NN3 ensemble 0.454% close to GKX 0.40% and beats a
constant; our linear models are stronger than GKX's, shrinking the NN–linear gap. Report display
defects (ΔR² vs test mismatch, "0" p's) fixed in the write-up, never in place. Seed dependence is the
key caveat. Doc drift: README:16 "pending"; D008 (monthly mean) vs D010/metrics (monthly sums —
weights months by stock count, unlike GKX's equal-weighted months) → D049 clarification. One-page
"rules at freeze-v1" table. **Phase 4:** tag result-v1; new work in `scripts/phase4_*.py` →
`results/exploratory/` with a forced EXPLORATORY banner + test; ensemble-size curve; level vs
cross-section; GBRT failure; per-year/crash; DM with equal-weighted months; OLS+H via D033; OSAP arm;
Phase 5 prereg before unsealing. **Risks.** No git remote — tags are self-timestamped, not external
proof: push or OpenTimestamps before opening data/sealed/; predictions are local-only (licensed CRSP)
— back them up and document a hash manifest; `--deviation` takes any string; don't refactor main().
**Confidence.** medium-high.

## 🗳️ Performance Optimizer
**Position.** No number-changing bug; one finished test run per model. The biggest finding: the
headline is produced by seed averaging (NN3 seed mean 0.163% vs ensemble 0.454%; the ensemble gain
exceeds the whole margin over train_mean). **Points.** Score all 31 seed subsets and shrinkage
α·pred + (1−α)·train_mean from saved `pred_seed*` columns (minutes, no refits); test 1 after dropping
the 1/3/5 most influential months; per-year cumulative ΔSSE. The CSV holds exact p's (1.3e-9 etc.).
NN used ~2.3 of 12 cores; peak RSS 10.8–12.0 GB per NN process → serial was right. Float32 features +
one slice per fold would roughly halve NN time — Phase 4 branch only, prove bit-identity first.
Fitted models were not saved: importance needs a NN3 refit (~2.9 h) at a commit that only adds
saving, asserted bit-identical to frozen predictions (a reproduction proof). **Budget.** Phase 4
≈ 8–10 h (saved-prediction analyses < 1 h; linear ablations < 1 h; NN3 refit + importance ≈ 4 h);
Phase 5 (8 folds) ≈ 10 h serial on the frozen path. **Risks.** Phase 5 power for test 1 ≈ 30%;
MDE ≈ 0.6 pp — frame as a decay estimate; NN3-dm ranking (expected t ≈ 2.1) is testable if
pre-registered; log any refit as `counts_as_trial=False` with its own D-entry. **Confidence.** medium.

## Synthesis — angles, not consensus
**Shared starting points (common prior — stress-test).** All four: provenance is clean; no
number-changing bug. All four: the headline must be bounded — "5-seed NN3 ensemble beats a
constant", ranking not reproduced, -dm secondary, single-seed NN3 below the constant, and the
report's "0" p-values / raw-ΔR²-next-to-CW rows need footnotes in the write-up (frozen code untouched).
Three: it is not "independent data" (D022) and the OSAP arm is owed. Three: Phase 5 must be
pre-registered before unsealing; test 1 is underpowered there; only a pre-declared -dm ranking test
can earn confirmatory status. What all four could miss for the same reason: none checked GKX's own
seed/ensemble count or GKX's DM weighting against the paper — verify from the PDF before comparing.
**Tensions.** Simplicity: no new fits, one page, stop. Devil/Performance: cheap saved-prediction
diagnostics (seed subsets, size/price/delisting splits) are needed to know what the result *is*.
Resolution: do the zero-refit diagnostics, all labelled exploratory, then write up.
**Unique.** Devil: bottom-1000 R² ~2× GKX → target-build suspect. Maintainability: no git remote —
tags are not external proof; push before Phase 5; back up predictions. Performance: compute budget;
reproduction refit as proof.
