# Local council 13 — review of the draft write-up (WRITEUP.md at 3db58f1)

> **Local council** — these perspectives all come from Claude playing different
> roles, not from different AI vendors. Treat agreement as a shared starting
> point to pressure-test, not as independent confirmation.

`--roles=devil,simplicity,maintainability,dx`; read-only. (First launch failed on a transient
permission-check error; relaunched once.)

## 🗳️ Devil's Advocate
Numbers check out; framing does not. L8 contradicted by the CSV (NN3 vs PLS p 0.048); GBRT "worse"
is a direction never tested (one-sided p 0.972, CI upper −0.004, ~0.13 of 0.21 pp is month level);
"sorting" contradicts the D049 addendum; "no single seed beats" is wrong on point estimates (seeds
2–3 above the constant) — add mean single seed 0.16% < 0.18%; "−170%" cited as a pre-look reason
(dry run −27,153%); "a reviewer" was an AI council — disclose; 25–30% power has no source; title
"pre-registered" overclaims. Missing: post-2002 fade (2003–16: 9 of 14 years negative), top-1000
failure as headline, -dm arms understated (pre-specified secondary, p < 0.001), DM weighting
mismatch, NN3 fixed ex ante as the worst raw NN in rehearsal. Confidence high.

## 🗳️ Simplicity Champion
Honest but too dense; ~8 errors (unsourced power; −170% vs dry-run value; DSR ≈ 1 only for NNs;
departures table missing trees-single-run, GLM inputs, D017; "as in the paper" true only for linear
and NNs; 18 vs 12 models; "reproduced bit-for-bit" ambiguous; Q5 tie vs top-1000 loss). Lead with
the ensemble/variance-reduction insight; collapse the label table; remove D-codes or explain them;
"still" in the title implies Phase 5; disclose AI assistance; private-repo timestamp unverifiable.
Confidence high.

## 🗳️ Maintainability Advocate
L8 contradicted by Fig 1; no trail from numbers to files (hand-copied, double-rounded CIs);
figures unstamped and fig 1 (frozen numbers) under results/exploratory; unpinned matplotlib;
GBRT 6/30 constant fallback unmentioned; departures missing (training-mean baseline, D017, NBER WP
source, late reference commit, prereg pointer edit, DM sums vs mean found after the look); D049
says result-v1 = cb59143 but the tag is d2086a7; README Phase 4 stale; master ahead of origin.
Confidence high.

## 🗳️ Developer Experience
R²_oos never defined; model names drift (OLS-all/OLS+H, train_mean/constant); D-codes and
"test 1/7" unexplained; Fig 1 orange bars with CIs crossing zero need a caption and a Holm marker;
Fig 2 reuses orange; timezones missing; "48 decisions" now 49; dry run was at 01f0936; -dm needs
"training-window mean added back" (no look-ahead); price ≥ $5 deferred; prereg file edited after
its tag; README should lead with the result; give a "what you can check without WRDS" section and
a glossary. Confidence high.

## Synthesis — angles, not consensus
**Shared:** numbers in the table are right; the prose overstated in several places (PLS, GBRT,
"sorting", seeds, DSR, −170%); AI assistance must be disclosed; the post-2002 fade, top-1000 loss,
ensemble dependence and NN3-as-worst-rehearsal-NN belong up front; the -dm arms are understated;
the power number must go until computed. **Tension:** Simplicity wants less process on the page;
Maintainability/DX want more traceability — resolved by a short §5 table + timeline and a
"check without WRDS" section. **Applied:** WRITEUP rewritten; table generated from frozen CSVs and
tested; figures moved to results/writeup/, numbered, Holm-marked, stamped; matplotlib pinned via
`--with`; D049 correction; README leads with the result. **Left to the author:** wording of the AI
disclosure; whether to make the repo public.
