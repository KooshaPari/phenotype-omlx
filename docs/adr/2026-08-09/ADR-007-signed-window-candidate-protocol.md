# ADR-007 — Signed execution-window + candidate-manifest protocol (design-only draft)

**Status:** Draft (no implementation)
**Date:** 2026-08-09
**Related:** BACKLOG-OMLX-002, BACKLOG-OMLX-004, EPIC-OMLX-004 (W1–W5, all blocked),
worktree `restore-niah-qwen35-yaml` @ `92c92f9`

## Context

Backlog items BACKLOG-OMLX-002 ("Address 'no current candidate' failure after
window contract injection") and BACKLOG-OMLX-004 ("Wire Harbor preflight to
signed envelope") describe a future state in which:

1. The Portage/Harbor operator path consumes a **signed envelope** that carries
   an **execution-window ID** (`PHENO_EXECUTION_WINDOW_ID`).
2. The envelope is verified against a **signer public key / key ID** that
   has been provisioned out-of-band (W4).
3. A **candidate manifest** (`scripts/verify_candidate_manifest.py` – does
   not exist today) is bound to the **current source HEAD** of the harbor
   task before `harbor run` is allowed to proceed.
4. The candidate verifier rejects the run if the manifest's `source_head`
   does not match the live manifest refreshed after the signed window ID is
   issued ("no current candidate" failure).
5. The dotenv loader's token-auth fix from Aug 8 (separate `apps/bench-cockpit`
   work) is bridged into the same gate so that an operator cannot bypass
   the envelope by exporting raw env vars.

As of worktree HEAD `92c92f9`, **none of this exists**:

- `PHENO_EXECUTION_WINDOW_ID` is not referenced anywhere in the source tree.
- `scripts/verify_candidate_manifest.py` does not exist.
- `config/portage.env.template` does not exist (only
  `config/langfuse_harbor_kpis.json` and `config/smoke_models.json` are
  present).
- `scripts/evals/run_via_harbor.sh` is a thin shell that only enforces
  `PORTAGE_ROOT`, `LANGFUSE_*`, and `HARBOR_ENV=apple-container`, then
  delegates to `uv run harbor run`. It has no signed-envelope consumption,
  no dotenv loader, no source-head binding check, and no candidate-manifest
  preflight.
- EPIC-OMLX-004 is entirely blocked: W1 (signed envelope) → W2 (authz
  sidecar) → W3 (window ID issuance) → W4 (signer key provisioning) →
  W5 (current candidate manifest refresh). All five nodes are `blocked`
  with 0% progress.

In other words, the two backlog items describe a **forward-looking
infrastructure protocol** whose implementation is gated behind the entire
EPIC-OMLX-004 chain. Neither item can be resolved today without first
landing the W1–W5 chain.

## Decision

Defer both backlog items until EPIC-OMLX-004 has at least W1, W3, and W4
in place. This ADR captures the **design shape** of the future protocol
so that when implementation begins it can be executed mechanically rather
than re-specced from scratch. **No code is shipped here.**

### Protocol shape (future)

#### 1. Envelope (W1)

```
{
  "envelope_version": 1,
  "iss": "portage-issuer",
  "kid": "harbor-2026-08-prod",
  "iat": <unix>,
  "exp": <unix>,
  "window_id": "PHENO_EXECUTION_WINDOW_ID value",
  "task": "omlx-qwen35-policy | omlx-niah-api-smoke | omlx-turbo-ssot",
  "env": "apple-container",
  "operator": "<principal>",
  "signer_scopes": ["harbor.run", "omlx.oracle"],
  "source_head": "<git SHA1 of harbor task tree at issuance>",
  "candidate_manifest_ref": "<sha256 of manifest payload>"
}
```

Signature: Ed25519 over the canonical JCS-envelope bytes (RFC 8785).
Verification key resolved via `kid` from a local keyring seeded by W4.

#### 2. Authorization sidecar (W2)

Out-of-process verifier that the preflight shell calls via Unix socket
(`/var/run/pheno-authz.sock`). The sidecar is the only entity that holds
the long-lived signer private key. The shell never sees the key.

#### 3. Execution window ID (W3)

`PHENO_EXECUTION_WINDOW_ID` is a short-lived (≤ 1h) opaque token issued by
the sidecar on successful envelope verification. The preflight enforces
`exp - iat ≤ 3600s` and rejects reuse (`jti` claim tracked in a local
deny-list).

#### 4. Candidate manifest (W5 + this ADR)

A `candidate_manifest.json` file lives next to each harbor task:

```
{
  "task": "omlx-qwen35-policy",
  "source_head": "<git SHA1>",
  "issued_at": <unix>,
  "fingerprint": {"sha256": "<digest of manifest+envelope>"}
}
```

`scripts/verify_candidate_manifest.py` will:

1. Load the envelope (from W1).
2. Resolve the task's candidate manifest from PORTAGE_ROOT or the repo.
3. Refuse the run if `envelope.source_head != manifest.source_head`.
4. Refuse the run if `manifest.source_head` is not the current HEAD of the
   task tree (the "no current candidate" failure mode).
5. Emit a structured `candidate_manifest_refreshed: true` event for the
   cockpit dashboard.

#### 5. Preflight integration in `scripts/evals/run_via_harbor.sh`

After the existing `PORTAGE_ROOT` / `HARBOR_ENV` / `LANGFUSE_*` checks
and **before** the `uv run harbor run` invocation:

```
# Pseudocode (DO NOT IMPLEMENT until W1+W3+W5 are non-blocked)
load_dotenv  $ROOT/config/portage.env.template  # portage.env.template not yet created
authz_check  --envelope .pheno/envelope.json --scopes harbor.run,omlx.oracle
require      PHENO_EXECUTION_WINDOW_ID
verify       $TASK/candidate_manifest.json --source-head "$(git rev-parse HEAD)"
```

The dotenv loader (Aug 8 fix in `apps/bench-cockpit`) is imported via a
shared `pheno_dotenv` helper so both reading `PORTAGE_ROOT` and binding
`PHENO_EXECUTION_WINDOW_ID` go through the same token-auth path.

## Consequences

Positive:

- The W1–W5 chain has a reference design so the eventual implementation
  passes the audit-described failure modes (`no current candidate`,
  `source-head binding`) on the first try.
- The audit backlog is honest: BACKLOG-OMLX-002 / -004 are marked
  `deferred` with explicit reasons, not silently left in "no status".
- The polyglot / multi-engine story (AGENTS.md §2) is preserved — the
  sidecar is language-agnostic (Unix socket + JSON), the verifier can be
  Rust, the preflight is shell, the manifest is JSON.

Negative:

- This ADR is heavier than a one-line code change. We accept that cost
  because the audit's pointer to these backlog items needs to be
  auditable: future agents or humans will see BACKLOG-OMLX-002 / -004
  and need an answer for why they are deferred.
- The protocol shape above is a **design**, not a working implementation.
  It will change as W1–W5 land. Treat this ADR as a checkpoint, not a
  contract.

## Verification

- Static: `git -C worktrees/phenotype-omlx/restore-niah-qwen35-yaml grep -nE 'PHENO_EXECUTION_WINDOW_ID|verify_candidate_manifest|portage.env.template'` returns zero hits (confirmed at HEAD `92c92f9`).
- Static: `EPIC-OMLX-004` progress in
  `_cockpit/audit-phenotype-omlx.json` is `0%` with all W1–W5
  `blocked` (confirmed at HEAD `92c92f9`).
- Process: BACKLOG-OMLX-002 and -004 are marked `deferred` in the audit
  with `deferred_reason` fields pointing to this ADR.

## Reopen conditions

This ADR should be revisited (and likely split into a real implementation
ADR) when:

1. W1 (signed envelope) lands in the main checkout.
2. W3 (window ID issuance) lands.
3. W5 (candidate manifest refresh) lands.
4. The Aug 8 dotenv loader fix is migrated from `apps/bench-cockpit`
   into a shared `pheno_dotenv` helper that the harbor preflight can
   import.

At that point, replace this ADR's "Decision" section with the actual
implementation commit, link the Verifying PR, and un-defer
BACKLOG-OMLX-002 / -004.
