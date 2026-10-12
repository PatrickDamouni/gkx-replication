# Local council 15 — review of the freeze-v2 code (90e41c9..da7c0e3) against D050

> **Local council** — these perspectives all come from Claude playing different
> roles, not from different AI vendors. Treat agreement as a shared starting
> point to pressure-test, not as independent confirmation.

`--roles=devil,simplicity,maintainability,security`; read-only; nothing under data/sealed/.

## 🗳️ Devil's Advocate
Seal holds; the look is not fail-closed. Report ignores uncommitted edits; NaN would bias DM/R²
silently; a crash after computing H1 forces a second look (inputs read after metrics; no
refuse-if-exists; no look record); any freeze-v2.N accepted; "pre-crash runs must reproduce"
undefined; crsp.fetch is a new function, not a kwarg; macro cap unwritten; 1987–2016 comparisons
from working-tree CSVs matched by position; secondary tests lack 1987–2016 values; build and runner
have no Python tag gate; NN_CONCURRENT is only an env check. Canary and mirror never exercise
build_holdout_panel's boundary logic — rehearse it on real data with 2016 as the holdout.
## 🗳️ Simplicity Champion
Lean; keep build_panel_through (forced by the whitelist, proven on real data) and the three modes.
Blockers: the dry run doesn't rehearse holdout() — share one path; move fetch out of crsp.py; join
1987–2016 values by key from frozen predictions; mutex; atomic sealed writes; assert instead of the
"training windows end ≤ 2008" comment; batch all edits before the pull.
## 🗳️ Maintainability Advocate
Untested one-shot paths: pull() and build() lack non-empty/contiguity guards; through should use
the last month with a non-null return (rf lag); fail-closed CRSP sha placeholder should be pinned by
a test; holdout() needs an end-to-end synthetic test; extend the whitelist to tests; refuse a second
report; mutex; compare() test; D051 should record the sha, checks and a runbook (detached freeze-v2
checkout; commit results only afterwards).
## 🗳️ Security Auditor
Hash checks fail closed and no licensed rows reach git. Gaps: the panel lock vouches for itself
(editable, uncommitted) — publish it and log the hash per fit; tag gate only in bash; freeze-v2.N
unlimited and report picks max over a glob matching freeze-v20; ledger/tag not tamper-evident;
mismatch output could print licensed values into a public council record; holdout logs and ledger
val_scores must not be committed mid-run; bind SQL parameters; bash `$(...)` failures pass silently.

## Synthesis — angles, not consensus
**Shared:** gate in Python; exact tags (v2, v2.1 only); refuse a second look; check everything
before computing; publish/log the panel hash; rehearse the real unseal path; move fetch out of
crsp.py; mutex; atomic writes. **Applied (8bb58a7, 13eb8ed, b5c1179, D051):** all of the above;
unseal builder rehearsed on real 2016 data (68,730 rows exact); full-path dry run reproduces
H1 4.569 / test 1 2.213; 185 + 7 tests. **Deferred to the pull step:** CRSP sha pinned by a test in
the same commit as the sha. **Not applied:** hash-chained ledger (single-user threat model; the
public tags and phase5_inputs rows cover the realistic risk); dropping the default WRDS username
(already public).
