# Local council 10 (short) — final check of candidate freeze commit dc893f0

> **Local council** — these perspectives all come from Claude playing different
> roles, not from different AI vendors. Treat agreement as a shared starting
> point to pressure-test, not as independent confirmation.

`--roles=devil,simplicity,maintainability`; read-only review of `git diff 01f0936 dc893f0`.

## 🗳️ Devil's Advocate
**Position.** Fixes are mechanically correct; one design problem must be settled before the tag:
the verdict mapping is non-monotone. **Points.** "Replicates" uses intersection-union (every raw
p ≤ .05) but "partially" needs test 1 under Holm across 7: seven p = 0.01 → "replicates" while the
header says "test 1 not rejected"; raise only leg 7 to 0.06 → "does not replicate (level only)",
skipping "partially". Coherent fix: "partially" = primary ∧ test 1 at raw α (a two-leg
intersection-union, valid at 5%), so "replicates" ⊂ "partially"; Holm stays descriptive. Verified:
NaN handling, provenance (tag missing → fails closed), FROZEN_PATHS, `rejected[0]` is test 1,
serial default, refusals logged, 0555155→dc893f0 touches only results/, run_test.py never imports
report. **Risks.** `.claude/council-cache/` not ignored (a council during the run kills the next
spec); `--deviation` is an unchecked string; the serial branch never ran at dc893f0 (low risk).
**Confidence.** medium (design call is the user's).

## 🗳️ Simplicity Champion
**Position.** No blockers; tag dc893f0 or a docs-only successor and stop adding rules.
**Points.** verdict/holm/provenance_errors small and fail closed; short-sha prefix match correct;
rehearsal report produced at 0555155 = candidate code; launcher change minimal. Docs drift:
checklist:28 still "< 10 GB", checklist:19 unticked. **Risks.** process complexity is now the main
threat — any edit under src/scripts/pyproject/uv.lock voids D047's "report/launcher only" claim;
the verdict has a cliff (all p = 0.04 → replicates; six at 0.04 + one at 0.06 → level only) —
document it; `--deviation` takes any string. **Confidence.** high.

## 🗳️ Maintainability Advocate
**Position.** No code blocker; one docs-only successor commit before the tag. **Points.** Code
matches D047 and tests pin the edges. Superseded rules lack forward pointers: D045 verdict bullet,
D046 heading, PREREGISTRATION.md:24-25, README.md:31 — add "amended by D047" before the tag so the
timestamp shows the amendment preceded the look. Checklist:28 and :19 stale. Ignore
`.claude/council-cache/` in git. **Risks.** `--deviation` should require `## {id}` in
DECISIONS.md; `check_bit_identity.py` exits 0 on DIFFERS (frozen now, leave it). **Confidence.** high.

## Synthesis — angles, not consensus
**Shared (stress-test, not corroboration).** No code blocker in the council-9 fixes; checklist
lines 19/28 stale; `.claude/council-cache/` should be git-ignored; `--deviation` is honor-system.
**Tension.** Devil: the verdict is non-monotone and must be fixed before the tag (user decision).
Simplicity: stop changing rules — but its own "cliff" is the same defect seen as documentation.
Resolution: the fix touches only `report.py`/`final_report.py` (not the prediction path), so it
costs one line + one test, not a re-rehearsal.
**Unique.** Maintainability: forward pointers from prereg/D045/D046/README to D047 before the tag.
