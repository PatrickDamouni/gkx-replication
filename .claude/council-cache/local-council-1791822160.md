# Local council 16 — final pre-unseal review of freeze-v2 (0541701) and its evidence (e0d2a09)

> **Local council** — these perspectives all come from Claude playing different
> roles, not from different AI vendors. Treat agreement as a shared starting
> point to pressure-test, not as independent confirmation.

`--roles=devil,simplicity,maintainability,security`; read-only; nothing under data/sealed/.

## 🗳️ Devil's Advocate
Code fails closed; the runbook doesn't work first time. Blocker 1: checking out freeze-v2 drops the
54 `-check2016` ledger rows (committed at e0d2a09) → the report refuses after 8 h; fix: restore the
ledger from master after checkout, and a copy-aside return to master. Blocker 2: bare `nohup` writes
`nohup.out` (not ignored) → launcher refuses; use the header's redirect. Fine as coded: multiple tags,
phase5-panel from detached HEAD, partial final year, training-mean windows, disk. Mundane
refusals: network at ls-remote, stale mutex after kill, AC power, CWD. Pre-declare the post-compute
failure path (analyse computes before isfinite). Optional early canary check. Build-time refusals
after sealed data is in memory need a policy; confirm the pre-2017 files hold signal month 201612.
## 🗳️ Simplicity Champion
Ready with one extra git command: `git checkout master -- results/` after the detached checkout;
exact command sequence; watch the 18-row grep, builder months, the lock push, counts-only
monitoring, one report run. Don't add safeguards; don't restore docs/.
## 🗳️ Maintainability Advocate
Ledger rollback confirmed (1,066 vs 1,120 lines; append-only); the unit test mocks `runs`, hiding it.
Restore before --build; rehearse `phase5_report --check2016` at the detached HEAD; copy-aside return
(never -f). All council-15 findings resolved at freeze-v2 except an end-to-end test of the detached
ledger state. Post-run: results commit unedited + tag; docs commit with D-entry, report sha256,
manifest extension, council record.
## 🗳️ Security Auditor
No licensed-data exposure in e0d2a09 or planned pushes (aggregates only; row-level data stays in
ignored parquet). Gates sound for an honest operator; remote lock tag matched by name only — verify
by hand. Nothing scheduled reads the holdout; keep monitoring to counts and keep agents off the
ledger. iCloud copy holds row-level CRSP — check MSU WRDS terms, enable ADP. Public timestamp:
create GitHub Releases (server-side created_at). The single look is enforced by honour; run one
command and keep its transcript.

## Synthesis — angles, not consensus
**Shared:** the ledger-rollback trap (all four) and the fix (restore results/ from master after the
detached checkout; copy-aside return). **Unique:** Devil — nohup.out; post-compute failure policy;
201612 boundary (checked: present, macro complete); Security — iCloud licensing, Releases, honour
system; Maintainability — rehearse --check2016 at the detached HEAD; two-commit closeout.
**Applied:** D053 (binding runbook and policies, no code change).
