# Local council 14 — review of D050 (Phase 5 pre-registration, draft 829118c)

> **Local council** — these perspectives all come from Claude playing different
> roles, not from different AI vendors. Treat agreement as a shared starting
> point to pressure-test, not as independent confirmation.

`--roles=devil,simplicity,maintainability,performance`; read-only; nothing under data/sealed/.

## 🗳️ Devil's Advocate
Don't tag yet. H1 silently replaces prereg-v1's ranking question and is mostly a within-month fit
test (NN3-dm adds the same training mean as the baseline) — state the deviation and that it is not
"ML beats linear"; freeze rule is a blacklist that leaves metrics/preprocess/report/align/datashare
free and lets run_test.py change; power HAC ≠ frozen (4.034 vs 4.0366), H1's 4.57 only from
exploratory code — use the frozen call; size 6–7% not adjusted; half strength is the realistic
case; reading sealed datashare = reading holdout returns (mom1m); holdout end could be 24–84 months
(datashare readme "April 2019") — pre-commit no minimum; Arm B spent on overlapping months; seal the
2017+ CRSP pull at ingest, Dec-2016 ME from the frozen file; no rule if bit-identity fails; decay
ratio CI undefined; ledger leaks val_scores; GitHub dates are forgeable — consider OSF/release.

## 🗳️ Simplicity Champion
Core is already simple; cut the edges. freeze-v2 scope must include crsp.py (hard 2017 cutoff) and
report code (final_report is freeze-v1-bound and lacks H1); drop the decay ratio (noisy ratio, new
code) and the holdout D048 label (declared uninformative — free headline for misreading); lean
option: NN3-dm only for H1. Power script divides lag terms by T−k (statsmodels by T). Say H1 was the
top of five candidates; HAC 5 is the frozen constant (D015's rule gives 3 at T≈60); H1 is a
cross-sectional-edge test. Resist adding robustness tests.

## 🗳️ Maintainability Advocate
Not lockable as a code contract: the seal lives in panel.py, walk_forward.py, align.py, crsp.py,
run_test.sh, final_report.py — use a whitelist (additive keyword args with seal-keeping defaults;
byte-identical model/metrics/report/preprocess/build files; all freeze-v1 tests pass unmodified).
H1 needs a new phase5_report.py that reproduces t = 4.568 and 2.213 on 1987–2016 before tagging,
prints H1 first and keeps D048 in a non-deciding section. Name run suffixes (-check2016, -holdout);
load 1987–2016 by commit, not "latest". The 2016 check runs on the old path; datashare splits on
signal month so Jan-2017 returns come from the pre-2017 file — a sealed-only loader drops them.
CRSP pull must start at 2016-12 for me_lag; holdout panel sha pinned in results/. Define a "look";
toolchain drift is the likely bit-identity failure; synthetic tests for the new paths.

## 🗳️ Performance Optimizer
Cheap: 2016-fold check ≈ 1.25 h (linear 2 min, trees 16 min, NNs 52 min); holdout ≈ 6.5–8 h serial
(budget 12 h); RSS ≈ 13–14 GB per process at the 2021 fold → forbid concurrent NNs; disk 52 GB free,
iCloud sync should finish first. Power bootstrap correct but HAC /(T−k) ≠ statsmodels /T. The
bit-identity check tests the wrong path → add 2016 as a canary fold in the holdout run (≈ 1.25 h).
Resume key lacks the panel hash → build panel once, pin sha, never re-pull CRSP; re-pull 2015–2016 to
check the legacy tables. Pre-decide the code-bug crash rule (freeze-v2.1, refit all, canary re-run);
counts-only monitoring is binding.

## Synthesis — angles, not consensus
**Shared (stress-test):** whitelist the freeze-v2 diff (metrics/report/preprocess missing; crsp and
report code must change); compute H1 with the frozen call and reproduce it before tagging; fix the
power script's HAC; HAC lag 5 stated as the frozen constant; CRSP pull from 2016-12 with Dec-2016 ME
from the frozen file; define the single look and keep the ledger closed; pre-decide failure rules.
**Tensions:** keep vs drop the holdout D048 label and decay ratio (Simplicity drop; Devil define) →
dropped both (side-by-side ΔR² instead). Lean NN3-dm-only run vs all 18 → user chose all 18; kept.
**Unique:** Devil — prereg deviation, holdout length 24–84 months, Arm B spent, external
registration. Maintainability — Jan-2017 signal-month trap, run-name hazard. Performance — canary
fold, panel lock, cost/RSS figures.
**Applied (1220892, tagged `phase5-prereg`):** all of the above; power re-run with exact HAC
(matches statsmodels to 6 decimals) at 24–96 months; H1 frozen-code t = 4.569; ENet+H vs constant
t = 1.63 for contrast. **Not applied:** NN3-dm-only lean run (user's 18-model choice stands); OSF
registration (user's option).
