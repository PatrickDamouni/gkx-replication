# Decision log

One entry per decision: what I chose, what I rejected, why. Newest at the bottom.
This file is the interview script.

---

## D001 — Project choice: replicate GKX (2020) instead of a novel model
- **Chose:** Replicate + extend Gu, Kelly & Xiu, "Empirical Asset Pricing via Machine Learning."
- **Rejected:** Building an original trading model; benchmarking TS foundation models; LLMs on filings.
- **Why:** A replication has a ground truth to check against, so mistakes are detectable.
  Not being a PhD student, the strongest signal I can send is evaluation rigor, and a
  replication puts that front and center.

## D002 — Data: CRSP via MSU WRDS + Open Source Asset Pricing signals
- **Chose:** CRSP monthly returns (WRDS, MSU access) merged with OSAP firm-level characteristics.
- **Rejected:** yfinance (survivorship bias — delisted firms drop out); OSAP portfolio-level
  returns only (can't do firm-level return prediction, which is the paper's core task).
- **Why:** Firm-level, survivorship-free returns are required to replicate the paper honestly.

## D003 — Extension: post-publication holdout (2017 → latest)
- **Chose:** Seal everything after the paper's sample and test it once, at the end.
- **Rejected:** Picking an extension after seeing replication results.
- **Why:** Choosing the extension before seeing data avoids a forking-paths problem, and
  "does it survive after publication" is a question a quant researcher actually cares about.

## D004 — Pre-registration locked before data access
- **Chose:** Lock `PREREGISTRATION.md` and tag the commit `prereg-v1` before any WRDS data is pulled.
- **Rejected:** Refining criteria after a first look at the data.
- **Why:** The tag gives a verifiable timestamp that the question, success criteria,
  abandon conditions and splits were fixed ex ante. Any change after this point is a
  logged deviation, not a silent edit.

## D005 — Prereg clarification: R²_oos is pooled over all stock-months (2026-10-07, before any data)
- **Chose:** Headline R²_oos = 1 − Σᵢₜ(r − r̂)² / Σᵢₜ r², pooled over every stock-month in the
  test window, as in GKX. Per-month R² reported only as a diagnostic.
- **Rejected:** Average of monthly cross-sectional R² (the prereg's "monthly R²" could be read this way).
- **Why:** "Monthly" in the prereg referred to return frequency; pooling matches the paper, so the
  replication comparison is like-for-like. Clarified before any model or data was run.

## D006 — Prereg clarification: seed estimand (2026-10-07, before any data)
- **Chose:** Report both: (a) the seed-ensemble (average predictions across ≥5 seeds) as the
  headline, matching GKX; (b) mean ± spread of single-seed results as a stability check.
  The best single seed is never reported.
- **Rejected:** Single-seed mean ± spread only (understates NNs vs. the paper and could flip the
  pre-committed model ranking); ensemble only (hides seed instability).
- **Why:** Keeps the replication faithful while still surfacing variance.

## D007 — Macro-characteristic interactions kept, computed in batches (2026-10-07, before any data)
- **Chose:** Keep GKX's characteristic × macro-predictor interactions, materialized per batch
  inside model fitting instead of stored as a full design matrix.
- **Rejected:** Dropping interactions (weaker replication); a full in-memory matrix
  (~1,800 features × ~4M rows ≈ 29 GB float32 > 26 GB RAM).
- **Why:** Faithful to the paper within this machine's limits. `evaluate` therefore takes
  characteristics and macro predictors separately.

## D008 — Phase 1a test design revised after a local council review
- **Chose:** Seams = load_panel (holdout sealed at load, split at ingest, holdout hash recorded) ·
  align/rank_normalize under a future-invariance property test · walk_forward Folds with a
  snapshot · metrics on per-month summaries (pooled R², DM on monthly mean loss differential via
  statsmodels HAC, shared month-block bootstrap, decile L/S with formation-month ME) · evaluate
  with zero/oracle/cheat models + a power analysis · one slow-marked scale test.
- **Rejected:** Seal inside walk_forward only (a notebook could read 2017+ directly); a planted
  leaky feature caught by an R² threshold (circular, misses realistic ~0.1% leaks); "trees beat
  OLS" in Phase 1a (tests models that don't exist, flaky at realistic SNR); a "noise ⇒ R² ≈ 0"
  assertion (wrong: fitted models score < 0 under the non-demeaned metric).
- **Why:** Tests should sit where leakage actually happens. Record:
  `.claude/council-cache/local-council-1791409997.md` (local, all-Claude reviewers — weak
  corroboration, used for angle coverage).

## D009 — OSAP release 202510 ingested and sealed (2026-10-07)
- **Did:** Downloaded OSAP firm characteristics (release 202510, 209 signals, 5,416,424 rows),
  converted to Parquet, split at ingest, then deleted the full zip and full Parquet so no
  unsealed copy remains on disk.
  - source zip sha256 `aae101c1fd4749c5a68d83734b6ce5e90a0f2f18b456f498e6f3b803081f6ee3`
  - pre-2017 file: 4,482,749 rows, 1925-12 → 2016-12, 31,770 permnos
  - **holdout file sha256 `8dd2dd29b6d54856f8e9612c8dbd9ad3f2015ca86f0060f8d825374e10e286cd`** —
    not opened until Phase 5; this hash proves it is unchanged when it is.
- **Timing convention:** an OSAP signal dated t uses information through end of t and predicts
  the return in t+1. Verified from openassetpricing 0.0.2's own code (STreversal at t = −ret in t),
  pinned in `tests/test_align.py`.
- **Rejected:** Keeping the zip "just in case" (an unsealed copy defeats the seal; it is
  re-downloadable and verifiable by the hash above).
