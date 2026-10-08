# Local council 3 — GKX → OSAP characteristic map

> **Local council** — these perspectives all come from Claude playing different
> roles, not from different AI vendors. Treat agreement as a shared starting
> point to pressure-test, not as independent confirmation.

Question: review the measured map (docs/feature_map.md, commit 2eb612a) for concept
mismatches; propose the final rule; holdout design. Members: Devil's Advocate,
Simplicity Champion, Maintainability Advocate (`--roles=devil,simplicity,maintainability`;
Performance/DX omitted as off-topic and to stay under the usage limit).

---

## 🗳️ Devil's Advocate
**Position.** Don't freeze it: best-|ρ| finds neighbours, not concepts. Three mechanical
failure modes, all reproduced on pre-2017 files.
**Key points.**
1. *Tail-only binaries:* MomRev and AccrualsBM are 0/1 two-way-sort flags overlapping 7–9%
   of datashare rows; pairwise-complete ρ on extremes is inflated (mom36m→MomRev "same"
   −0.97; chmom, acc, sgr likewise). rd_sale→RDcap at 11% overlap.
2. *Predictor-only file:* 209 of 331 documented signals; ~20 GKX chars exist as Placebos
   (cashdebt, roic, sgr, salecash, rd_sale, pchsaleinv, currat, quick, pchcurrat, pchquick,
   depr, pchdepr, salerec, saleinv, secured, securedind, roavol, GrGMToGrSales,
   GrSaleToGrReceivables, BetaSquared, DivYieldAnn, ChPM).
3. *Quarterly timing ~4 months:* roaq 0.65 → 0.95 at k=4; chtx/ChTax 0.31 → 0.85 (k=4);
   ear/AnnouncementReturn 0.04 → 0.84 (k=5); bm/BMdec peaks at k=0 (annual aligned).
- Replace: mvel1 ← CRSP me; mom36m, chmom ← CRSP rebuild (exact 97–98%); roaq ← roaq(+4);
  baspread ← BidAskSpread; std_turn ← std_turn or none; betasq ← beta²; beta by concept.
  Placebo-only rows → (Placebo or) none. lgr, grltnoa → none (duplicates of agr/invest).
  acc ← Accruals (variant). chempia, chatoia: demean base within sic2. roeq ← RoE (proxy).
- None rows with counterparts: mom1m (CRSP exact), indmom (sic2 mean of mom12m, 0.86),
  mve_ia/bm_ia/cfp_ia/chpmia by sic2 demeaning; ear, chtx with shift.
- Rule: priority CRSP-exact (ρ≥0.99, ≥95% equal) → documented-definition OSAP match →
  sic2 demean → missing; one declared shift per frequency (annual 0, quarterly +4); ρ ≥ 0.80
  (0.6 only as labelled proxy); overlap ≥ 70%; ≥5 unique values; injective; sign from docs.
  Rerun datashare on the same matched subset S.
**Risks.** On-disk CRSP has only monthly ret/me (no volume, daily, SIC); holdout needs a
new pull incl. SIC. Thresholds fixed in advance aren't meaningful without a concept gate.
**D022 is a prereg deviation** (Q1 said "independently built data"): say so. Lag per
characteristic chosen by max ρ would be a specification search — fix by frequency.
Coverage-limited OSAP signals (tang, sinAlgo, MS, VolSD) will be mostly imputed.
**Holdout:** don't train datashare / test OSAP. datashare likely covers ~2017–2021
(349k sealed rows / ~5,700 per month; inferred, not checked). Primary datashare→datashare;
secondary OSAP→OSAP 2017–2024; report both on overlap; never splice.
**Confidence.** high.

## 🗳️ Simplicity Champion
**Position.** Stop argmax-over-209 and drop same/close; a fixed lookup table, one OSAP
signal per GKX char, accepted only on concept match + ρ + coverage; nothing else substituted.
**Key points.** Rejections/replacements largely as Devil (mvel1, mom36m→LRreversal or CRSP,
chmom, acc→Accruals, roaq→roaq, baspread→BidAskSpread, beta/betasq→Beta). Reject sgr,
cashdebt, salecash, lgr, dy, roic, rd_sale, grltnoa, std_turn, chempia, pchsaleinv
(duplicates). Exclude binary combo signals (≈390 rows/month vs ≈4,000). Don't hand-build
_ia variants (a data-construction project, not a mapping). Rule: concept + |ρ|≥0.60 +
≥50% coverage + injective + no combo signals; yes/no verdict. Tally ≈ 48 of 94.
Robustness: OSAP-48 vs datashare-48 (not vs 94).
**Risks.** Same holdout view (never stitch). Concept table written after seeing ρ: log it;
defence is no returns touched. Near-zero ρ on exact concepts unexplained (timing?).
**Confidence.** high on rejections; medium on the exact count.

## 🗳️ Maintainability Advocate
**Position.** Treat the measured table as evidence, not the map; commit a hand-reviewed,
version-controlled, test-enforced table (D024) before any returns touch it.
**Key points.** Predictor-only search space (Placebos missing). No coverage check (MomRev 7–9%).
Source columns reused (OperProfRD ×3; 11 others ×2) — enforce injectivity by test.
CRSP-on-disk is monthly only: exact for mom1m, mvel1, chmom, mom36m; SIC needed for
indmom/_ia; daily data needed for retvol, maxret, turn, ill, zerotrade, std_*, baspread.
Rule: written concept note citing GHZ + OSAP Detailed Definition; |ρ|≥0.60; coverage ≥~70%;
injective; declared sign + transform registry; ρ validates, never selects.
Holdout: two pre-registered arms; before unseal, measure the input-swap cost on pre-2017
validation years.
**Risks.** Generator overwrites its own doc (and docstring says 24 months, MONTHS has 25).
"none" rows display the unmatched signal's description (misleading). Never recompute the
map at runtime: commit `config/feature_map.csv` (gkx, source, signal, sign, transform,
concept_note, rho, coverage, decision_id, OSAP release) + `tests/test_feature_map.py`;
measured output → `results/feature_map_measured.csv`. CRSP-derived chars need their own
module with D023-style agreement tests. Restore GKX sign orientation.
**Confidence.** high on structure; medium on individual replacements.

---

## Synthesis — angles, not consensus

**Shared starting points (common prior — stress-test).** All three: argmax-|ρ| is the wrong
selector; concept must gate and ρ/coverage only validate; MomRev/AccrualsBM-type binary
tail signals produced false "same" verdicts; the file lacks OSAP Placebos; mvel1/mom1m/
mom36m/chmom should come from CRSP; each OSAP column used once; datashare must be rerun on
the same matched subset; the holdout must never splice sources. Shared blind spot to
check: all three relied on recalled GHZ definitions — the concept notes must cite the OSAP
`Detailed Definition` text and GHZ's appendix, not memory.

**Verified after the council (not by it):** OSAP does not distribute Placebos at all — the
package and openassetpricing.com/data offer only the 209 Predictors. Placebo-only GKX
characteristics are therefore *missing* in the OSAP run.

**Genuine tensions.**
- ρ threshold 0.60 (Simplicity, Maintainability) vs 0.80 with 0.60–0.80 as a labelled proxy
  (Devil). Resolution: accept ≥ 0.80 as validated, 0.60–0.80 only with a written proxy
  label, report both counts.
- Build _ia characteristics by sic2 demeaning (Devil) vs don't (Simplicity: a new data
  project). Resolution: datashare already has sic2 pre-2017; demeaning is a small,
  testable transform, but it needs SIC for the OSAP holdout years — defer until the
  holdout CRSP pull adds siccd.
- Quarterly +4 shift: Devil measured it; choosing it by max ρ is a spec search.
  Resolution: justify it from datashare's readme ("quarterly with at least 4 months lag"),
  i.e. documentation, and apply it to all quarterly signals uniformly.

**Blind spots.** None of the members asked whether the OSAP check run is worth the cost at
48-ish characteristics versus simply reporting it as a partial-coverage robustness check —
it is still worthwhile because it is the only path to 2022–2024 holdout characteristics.

**Suggested direction.** Freeze the rule (D024) and the holdout two-arm design (D025)
before any return-based fit; build `config/feature_map.csv` + tests; move CRSP-derived
characteristics into a tested module; regenerate the measured table as evidence only.
