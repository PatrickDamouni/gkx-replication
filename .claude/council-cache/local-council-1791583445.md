# Local council 7 — tree inputs and GBRT's missing ranking skill

> **Local council** — these perspectives all come from Claude playing different
> roles, not from different AI vendors. Treat agreement as a shared starting
> point to pressure-test, not as independent confirmation.

Questions: (1) faithful and feasible tree inputs (D012 is unfaithful: GKX trees use z = x⊗c, no
standalone macro); (2) why GBRT+H has no cross-sectional ranking skill on the rehearsal.
Members (`--roles=devil,simplicity,performance`). Simplicity and Performance failed on network
errors (ENOTFOUND / ENETUNREACH) and were re-run with the same brief; two orphaned scratch jobs
from the failed runs (one 4.8 h benchmark, one 5 h diagnostic) were stopped by the main session
after they pushed load to ~60, and Performance was told to re-time anything measured before then.
All diagnostics used data ≤ 1986 only, in scratch scripts, with no repo writes.

---

## 🗳️ Devil's Advocate
**Position.** Not a bug, not sic2, not Huber: the characteristic encoding carries a time stamp;
trees use it to fit the monthly market level, and validation MSE then selects a ~1-tree model.
Switching to z won't fix it and likely worsens it.
**Key points.** Fold 1986, chars only, depth 2: validation MSE picked iteration 1 for every config
(val R² +0.012%, matching the ledger exactly); first tree splits on rd/divi with rank IC −0.028.
12-fold saved predictions: D012 GBRT mean monthly rank IC −0.027 (t −5.1), GBRT chars+sic2 +0.001,
RF −0.006, vs PCR-raw +0.057 (t 8.8), GLM+H +0.063 (t 9.7). Time stamps: rank-normalized ties make
discrete chars' values shift with cross-sectional shares (rd majority value −0.014 in 1970, −0.141
in 1974); median-filled 0 marks the era (75/94 chars all-zero in 1957, 22 in 1963, 0 in 1986).
Chars-only trees reach +9–19% in-sample pooled R² with ~0% demeaned — fitting month means (15.8% of
pooled SS) — and extrapolate to −2% to −4% monthly means on validation. Linear models can't (a
linear function of ranked chars has ~0 cross-sectional mean). Huber ≈ L2 (ξ clips 0.1%); sic2
categorical vs one-hot identical. Training on the within-month demeaned target: 285/34 trees,
validation rank IC +0.083–0.085 (t ≈ 14), test-1986 IC +0.073–0.078. z cost: GBRT ≈ 6 h, RF ≈ 55 h+.
**Risks.** Third rehearsal-driven change (D031, D040, now trees) — present as two pre-registered
arms (faithful GKX trees with this diagnosis + a disclosed variant). NNs share the exposure — check
in-sample pooled vs demeaned R² before spending hours. Tie encoding of binaries is itself a choice.
Log `best_trees_`. Report rank IC / long-short too. Evidence mostly fold 1986.
**Confidence.** high on the mechanism.

## 🗳️ Simplicity Champion
**Position.** Keep trees on chars + sic2 (95 columns); don't build z now. The failure is selection
on pooled validation MSE, dominated by an unpredictable monthly level. Simplest fix: train and select
on a within-month demeaned target, add a constant level back.
**Key points.** Fold 1975: every raw-target config negative on validation; best at 1–2 trees.
Fold 1981: chosen depth 1, lr 0.1, 476/526 trees, still negative val R², test-year demeaned R²
−0.79% / −0.86% — trees that pull the level toward the validation mean rank badly; the −0.189%
average mixes switched-off folds and anti-ranking folds. sic2 inert. Demeaned target in 1975:
validation R² +0.27% to +0.43% with 49–1000 trees (test year −0.05%, one fold). Fix ≈ 3 lines in
`_LGBM.fit/predict`, applied to RF too. Reject "chars × macro only" (costly, still unfaithful).
D012's rationale is wrong at depth 1–2 (a stump can't express an interaction) — describe as a
characteristics-only tree deviation. Full z only as a later disclosed robustness run.
**Risks.** A second rule reversed after rehearsal (after D040) — pre-register, keep raw-target rows.
Rehearsal is harsher than GKX's windows; "GBRT ≈ 0" may be honest. Pooled R² of a demeaned-target
tree rests on the added constant — report demeaned R². Argmax over all-negative validation scores:
a minimal GKX-faithful fallback ("use the 1-tree model if nothing beats it") would have prevented
the −0.79% fold even without demeaning.
**Confidence.** medium (2 folds; fix on 1 fold).

## 🗳️ Performance Optimizer
**Position.** z is feasible only with a memory-centred rewrite: ~29 h at 8 threads for the full
36-config RF + GBRT grid over 30 folds, ~16 h with a selection shortcut, vs ~5 h for D012 measured
the same way. RAM (24 GB) is the binding constraint.
**Key points (clean fresh-process runs, 3.3M tiled ≤1986 rows, 8 threads).** z via `lgb.Sequence`
(847 features incl. categorical sic2): build 12.5 s, peak ~10 GB, binned ~2.8 GB. Per tree: GBRT
523–558 ms vs 54–87 ms at 103 inputs; RF 388–640 ms vs 60–139 ms. 4 threads is 1.8–1.9× slower
(memory-bandwidth bound) — parallel configs don't help. Use `force_col_wise` at this width
(row-wise 1.4–3.2× slower). Totals: GBRT ≈ 8 h, RF ≈ 20 h; features-per-split doesn't reduce cost.
Shortcut: train all 36 RF configs to 100 trees, select, continue the winner to 300 (`init_model`)
→ RF ≈ 7.5 h (stated deviation). One binned Dataset per fold reusable across the grid; read
validation MSE from recorded history; incremental Datasets not possible in LightGBM 4.7. Demeaned
target: O(n), adds no tree cost as a replacement; doubles cost as an extra arm; ξ must come from
demeaned OLS residuals. Cheapest test of the level hypothesis: demean on D012's inputs first.
**Risks.** Swap thrash past ~21 GB footprint (build 12.5 s → 1,000 s, tree 0.5 s → 33 s, no error).
Don't precompute z to disk (12.7 GB). Tiling understates late-era cost (±20%). 920 dummies vs
categorical changes the k/n denominator. z may worsen level-fitting (x_j·c_k = c_k at a moving
threshold).
**Confidence.** medium.

---

## Synthesis — angles, not consensus

**Shared starting points (common prior — stress-test).** Devil and Simplicity, measuring
separately, found the same mechanism: raw-target validation MSE is dominated by the unpredictable
monthly level, so selection collapses GBRT to ~1 tree or keeps level-chasing, anti-ranking trees;
a within-month demeaned target restores ranking. Performance independently warns z may *worsen*
level-fitting.

**Verified after the council by the main session (scratch, ≤1986, no repo writes):** raw target —
fold 1986 chose 1 tree (test demeaned R² −0.001%, rank IC −0.026); fold 1980 chose 1 tree
(−0.018%, IC −0.001). Demeaned target — fold 1986: 978 trees, test demeaned R² +0.397%, IC +0.094;
fold 1980: 662 trees, +0.211%, IC +0.030.

**Genuine tensions.** Faithfulness (raw target, z inputs) vs a working tree model (demeaned target,
95 inputs). Simplicity's minimal faithful fallback (use the 1-tree model when nothing beats it)
keeps GKX's raw target but yields an essentially constant tree. Forking-path risk: a second
post-rehearsal rule change after D040.

**Blind spot.** Whether the same level-chasing hits NN1–NN3 (nonlinear in era-stamped ranks) — to be
checked on the rehearsal before their compute is spent, and ideally pre-committed now.
