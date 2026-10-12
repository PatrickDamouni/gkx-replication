# Local council 12 — review of Phase 4 step 1 (D049 diagnostics, commit 2d795fd)

> **Local council** — these perspectives all come from Claude playing different
> roles, not from different AI vendors. Treat agreement as a shared starting
> point to pressure-test, not as independent confirmation.

`--roles=devil,simplicity,maintainability`; read-only.

## 🗳️ Devil's Advocate
**Position.** Arithmetic correct; framing too generous. Honest reading: a characteristic model beats
the historical mean only in microcaps, largely via a few 2001 months — not GKX-specific.
**Points.** Three unlogged deviations from D049 (CW leg skipped in item 7; last CRSP vs last panel
month; NYSE breakpoints from all share codes). Size: NN3 *loses* to train_mean in Q2–Q4 and ties Q5;
Q1 contributes ≈ +0.318 pp, more than the whole +0.273 gain. "Within-month" is not pure ranking
(MZ slopes 0.73). Influential months one-sided; 3 of 360 months ≈ 35% of ΔR²; excluding last months is
look-ahead; 4-seed subsets share seeds, so they understate seed variance. Multi-year runs (1993–98
positive, 2004–08 negative) exceed HAC 5 lags / 12-month blocks → inference may be optimistic.
**Next.** Back up + push now; power check before D050 (test 1 ≈ 23–31% power on the holdout);
build D050 around a test with power (pre-declared NN3-dm ranking DM or CW, ≥ 10 seeds), disclosed as
informed by 1987–2016; skip OSAP. **Risks.** Pre-registration is not externally verifiable (local
tags); GBRT+H's loss is mostly level (between-month SSE 8.82 vs 8.69); trees have no seed ensemble;
sealed data on the same unbacked disk. **Confidence.** high (power: medium).

## 🗳️ Simplicity Champion
**Position.** Diagnostics correct and lean; k = 5 reproduces 0.4539 / p 0.0134. Back up, write the
one-pager, log D050 + OSAP deferral; don't open OSAP now. **Points.** Rename "ranking" →
cross-sectional; 261 rows unassigned; proxy includes mergers; trimming best months measures how many
months the result rests on. Restatement: train_mean beats NN3 in Q2–Q4; the edge is within the
NYSE bottom quintile. One sentence + 5-row table + per-year bar chart. Order: push/backup → one-pager
before Phase 5 → D050 (withdraw the "independent data" promise honestly, fix Phase 5 family with
~30% power stated) → decide Phase 5 (~10 h). Don't promote -dm. **Risks.** process volume; 2001
overlaps GBRT fallback years; delisting rows 0.7% of rows / 2.9% of Σr², all models negative — one
caveat line; wrong council-11 filename cited (actual 1791771854); weak "sealed" guard.
**Confidence.** high.

## 🗳️ Maintainability Advocate
**Position.** Arithmetic right, but the commit broke reproducibility of the frozen report:
`scripts/phase4_diagnostics.py` sits under FROZEN_PATHS, so `final_report.py` now refuses at HEAD.
**Points.** Move to `exploratory/`, add a test that FROZEN_PATHS are unchanged since freeze-v1.
Durability: `*.parquet` ignored, no remote — back up predictions with a hash manifest, push. SSE
split correct; label "cross-sectional"; count unassigned quintile rows; exchcd/me_lag timing; label
the proxy as last CRSP month; assert the 5-seed ensemble equals the frozen pred; seed labels are
column positions; note CW excluded from item 7; rename "test-1 p" in the trim table and add the
symmetric trim; HAC over gapped months. **Risks.** thin tests (no between+within identity); coupling
to private final_report functions — comment it; no provenance header (commit + input hashes); OSAP
only after push, backup, D050. **Confidence.** high.

## Synthesis — angles, not consensus
**Shared (stress-test, not corroboration).** All three: no number-changing bug; labels overstate
("ranking", "test-1 p", "excluding last month"); the size result is stronger than summarised — the
edge is entirely in NYSE quintile 1, with train_mean ahead in Q2–Q4; back up and push before anything
else; don't promote -dm without pre-declaring it; OSAP not now. Common blind spot: none checked GKX's
own seed count or its size-split method against the paper.
**Tension.** Devil: redesign D050 around a test with power (choice informed by the seen window,
disclosed). Simplicity: reuse D048 machinery unchanged and state ~30% power. The user decides.
**Unique.** Maintainability: frozen-path regression (fixed in 0905d6a). Devil: serial dependence
beyond HAC/bootstrap windows; GBRT loss is mostly level.
**Applied in 0905d6a / 27707d2:** moved to `exploratory/`; frozen-path test; between+within test;
ensemble and provenance asserts; labels; symmetric trim; D049 addendum logging the deviations.
