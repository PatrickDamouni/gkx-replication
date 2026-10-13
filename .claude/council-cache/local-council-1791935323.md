# Local council 19 — review of the council-18 fixes (1b539a8, da2a555, 19bc004, release edits, settings, archives)

> **Local council** — these perspectives all come from Claude playing different
> roles, not from different AI vendors. Treat agreement as a shared starting
> point to pressure-test, not as independent confirmation.

`--roles=devil,simplicity,security`; read-only.

## 🗳️ Devil's Advocate
Every timestamp and id verifies (activity log, ledger, releases, rulesets, SWH, Wayback). New error
from council 18: "no 2017+ row entered a model or metric before the look" contradicts D050 §10 —
fits used 2017+ rows from 14:05:39Z; anchor on the unseal (22:01:36Z). "32 hours before the look"
measures to the wrong event (≈ 7h45m before the unseal, ≈ 24h before the first fit). Missing the
strongest point: 0541701 pins every sealed input's sha256 before the unseal, and the run is
reproducible. Rulesets don't bind the owner; SWH/Wayback postdate the look. Residual risk to name:
a peek between 2026-10-08 and the unseal rests on the author's word. One append-only fix, then stop.
## 🗳️ Simplicity Champion
Stop. Fixes were right and cheap; some settings protect little in a closed project but cost nothing;
don't tidy D054 (append-only); more councils cost more than they find; reviewers read the README.
## 🗳️ Security Auditor
Settings configured and enforced; archives hold only already-reviewed public material. Low gaps:
owner token can disable rulesets (fix wording), existing releases mutable (cite the Wayback capture),
Actions enabled with no workflows, master protected via ~DEFAULT_BRANCH, WRDS username default in
code (already archived; crsp.py is frozen), iCloud terms unverified.

## Synthesis — angles, not consensus
**Applied (author OK):** append-only D054 bullet (anchor = unseal; pinned hashes; residual risk;
rulesets wording; archives postdate the look; reproducibility), corrected `phase5-prereg` note,
WRITEUP §3. **Not applied:** removing the WRDS username default (frozen file, already archived);
further settings. **Repo frozen after this commit; no further councils.**
