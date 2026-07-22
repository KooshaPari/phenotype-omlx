# Langfuse self-host (Apple Container / Podman)

Self-host runs the **same codebase** as Langfuse Cloud with **unlimited** units,
users, and retention under the MIT core. Use this as **primary** when Hobby caps
(50k units / 30d / 2 users) or freemium friction appear. Cloud stays an optional
mirror for sharing / backup.

**Never Docker.** Runtime order:

1. Apple Container — `container compose` (after `container system start`)
2. Podman — `podman compose` / `podman-compose`
3. Fail loudly if only Docker is present

## Quick start

```bash
cd apps/bench-cockpit

# Generate secrets once into deploy/langfuse/.env (gitignored)
bash scripts/langfuse/self-host.sh init

# Bring stack up (web :3000)
bash scripts/langfuse/self-host.sh up

# Health
bash scripts/langfuse/self-host.sh status
```

Open `http://127.0.0.1:3000`, create project / keys (or use headless init vars in
`deploy/langfuse/.env`), then point cockpit:

```bash
# apps/bench-cockpit/.env
OBSERVABILITY_BACKEND=langfuse
LANGFUSE_BASE_URL=http://127.0.0.1:3000
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

Re-run Minimax LLM connection + Sync judges against the self-host project
(Settings → LLM Connections → Minimax anthropic adapter).

## Why self-host is “more powerful” than Hobby cloud

| | Cloud Hobby | Self-host MIT |
|--|-------------|-----------------|
| Units / month | 50k hard cap | Unlimited |
| Retention | 30 days | Your disk / policy |
| Users | 2 | Unlimited |
| Product features | Full (capped) | Full (same images) |
| Data residency | US/EU SaaS | Local / VPC |
| Ops | Zero | You own upgrades |

Enterprise-only when self-hosting: SCIM, audit logs, retention *policies* (commercial).

## Sync / dual-write strategies

Langfuse does **not** magically replicate Cloud↔self-host. Choose one:

### A. Self-host primary (recommended)

- Cockpit + Harbor + judges write only to `LANGFUSE_BASE_URL=http://127.0.0.1:3000`
- Keep Cloud project for humans/demos; periodically migrate with Langfuse’s
  [data migration cookbook](https://langfuse.com/docs/api-and-data-platform/features/public-api)
  (Python scripts for traces / prompts / datasets)
- MCP/CLI/Skill point at self-host URL

### B. Dual-write (more filled local + cloud safety net)

- Primary ingest → self-host
- Optional second ingest of the same batch to Cloud US (cockpit can grow a
  `LANGFUSE_MIRROR_BASE_URL` later; until then run `seed` twice with swapped env)
- Judges run on primary only (avoid double Minimax spend)

### C. Cloud primary → migrate later

Fine for smoke; switch to A before agent fleets hit Hobby caps.

## Compose layout

Vendored file: `apps/bench-cockpit/deploy/langfuse/compose.yml`
(from upstream `langfuse/langfuse` `docker-compose.yml`).

Stack: langfuse-web, langfuse-worker, postgres, clickhouse, redis, minio.

Update:

```bash
curl -fsSL https://raw.githubusercontent.com/langfuse/langfuse/main/docker-compose.yml \
  -o apps/bench-cockpit/deploy/langfuse/compose.yml
# re-apply vendored header comment if needed
```

## MCP on self-host

```text
${LANGFUSE_BASE_URL}/api/public/mcp
```

Local HTTP is supported for development (`http://127.0.0.1:3000/api/public/mcp`).
See `docs/guides/LANGFUSE_MCP_CLI.md`.

## Ops notes

- Runtime: Apple Container + `container compose` plugin **or** standalone
  `~/.local/bin/container-compose` (flaticols/container-compose). Never Docker.
- Data: bind mounts under `LANGFUSE_DATA_DIR` (default
  `~/.local/share/phenotype/langfuse`) — never `/tmp`.
- Smoke: `up` waits for `http://127.0.0.1:3000/api/public/health` (or
  `bash scripts/langfuse/self-host.sh smoke`).
- Disk: ClickHouse + MinIO grow with traces — prune data dirs deliberately; never
  recursive `find` from Phenotype root.
- Secrets: `deploy/langfuse/.env` is gitignored; rotate `NEXTAUTH_SECRET`,
  `ENCRYPTION_KEY`, DB/MinIO passwords after `init`.
- Stop: `bash scripts/langfuse/self-host.sh down`
