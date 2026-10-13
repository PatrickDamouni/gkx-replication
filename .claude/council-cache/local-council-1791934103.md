# Local council 18 — review of the GitHub releases and the D054 addendum (dd08e7f)

> **Local council** — these perspectives all come from Claude playing different
> roles, not from different AI vendors. Treat agreement as a shared starting
> point to pressure-test, not as independent confirmation.

`--roles=devil,simplicity,security`; read-only (one member ran a no-op `python3 -c 1` by mistake;
no project code, no data).

## 🗳️ Devil's Advocate
Numbers and hashes all correct. Provenance wording overclaims: release `created_at` echoes the
client-set tagger date (only `published_at`, after the look, is server-set); GitHub's activity log
records branch pushes only, so nothing server-side shows `phase5-panel` before the fits (no push
between 2026-10-12T16:22:41Z and 46e7466); "before any holdout data were read" is literally false
(the 2026-10-08 ingest split read the file). Strong evidence that exists: D050 pushed
2026-10-12T14:16:23Z, freeze-v2 code 15:03:37Z. Fix notes, D054 addendum and WRITEUP §3; archive
the activity page with a third party; no more releases; don't delete.
## 🗳️ Simplicity Champion
Repo essentially finished. Keep three releases; add none; keep dry-run and council files (they are
referenced). One README commit at most (duplicate WRITEUP pointer, "step 1 done"); then stop.
## 🗳️ Security Auditor
No licensed or private data in releases (no assets; source archives = 131 tracked files, aggregates
only). Integrity: no rulesets, master unprotected, release immutability off, tags unsigned →
add tag and master rulesets, enable immutability, Software Heritage snapshot. Identity: commit
email exposes macOS user and hostname; console log has a local path; use a noreply email going
forward, don't rewrite history. iCloud note is public; settle WRDS terms. Enable secret scanning and
push protection; disable the wiki.

## Synthesis — angles, not consensus
**Shared:** nothing leaked; no new releases; no deletions. **Applied (1b539a8 and release edits):**
corrected phase5-prereg and phase5-panel notes, D054 correction, WRITEUP §3 caveat, README tidy.
**Pending the author's explicit OK (persistent settings / third-party services):** tag and master
rulesets, release immutability, secret scanning, wiki off, third-party archive, git noreply email.
