# Langfuse self-host (Apple Container / Podman)

**Not the default.** Phenotype uses [Langfuse Cloud Hobby](./LANGFUSE_CLOUD.md)
until caps bite or a meaningful fork appears. Self-host is the overflow / lab
path: same codebase as Cloud, unlimited units/users/retention under the MIT core.

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

Stack: langfuse-web (:3000), langfuse-worker (:3030), clickhouse (HTTP
:8123, native host **18123**), redis (host **16379**), minio (:9090).

**Postgres (default):** host Homebrew `postgresql@17` via Apple VM gateway
`192.168.65.1:5432` (`DATABASE_URL=…@192.168.65.1:5432/langfuse`). Compose
`postgres` is profile-gated (`COMPOSE_PROFILES=embedded-postgres`) — library
`postgres:17` pulls are often Hub-429 / unpack-broken on Apple Container.

**Apple DNS gap:** short names (`redis`, `clickhouse`) and even
`langfuse-clickhouse` fail from app containers (resolver = gateway).
`self-host.sh` injects live container IPs into `.env`, then starts apps via
`compose.apple-apps.yml` (no `depends_on`) so `up` does not recreate deps and
invalidate those IPs. Memory: web `mem_limit: 4g`, worker `2g` (1 GiB default
OOMs Next.js init).

Update:

```bash
curl -fsSL https://raw.githubusercontent.com/langfuse/langfuse/main/docker-compose.yml \
  -o apps/bench-cockpit/deploy/langfuse/compose.yml
# re-apply Phenotype header, host-PG defaults, mem_limit, named redis vol
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
- ClickHouse image: **pinned to `clickhouse/clickhouse-server:24.8`**. Apple
  Container hangs unpacking `:latest` (~820MB arm64, stalls ~10% / Zero KB/s)
  and often fails with `HTTPClientError.remoteConnectionClosed` or
  `XPC timeout … networkList` under concurrent load. Prefetch once:
  `container image pull docker.io/clickhouse/clickhouse-server:24.8` with no
  other `container run` in flight; restart (`container system stop` then
  `printf Y | container system start`) if the apiserver wedges.
- Apple Container bring-up: `self-host.sh up` starts deps **serially**
  (clickhouse → redis → minio, settle delay), IP-injects endpoints, then
  web/worker via `compose.apple-apps.yml`. Parallel `compose up` often kills
  the apiserver mid-create; app `up` with `depends_on` recreates deps and
  breaks IP inject.
- Host Postgres: Homebrew role/db `langfuse`/`langfuse`,
  `listen_addresses='*'`, `pg_hba` allow `192.168.0.0/16` (and `10.0.0.0/8` if
  needed). Embedded compose postgres only with
  `COMPOSE_PROFILES=embedded-postgres` and a real library image (not a CNPG
  retag). Never Docker.
- Registry: anonymous Docker Hub pulls hit `429 Too Many Requests` after
  several large images. Prefer already-cached langfuse/redis/clickhouse
  images + host PG; or `container registry login docker.io` then pull.
- ClickHouse perms: compose omits `user: "101:101"` (cannot host-chown to uid
  101 without interactive sudo) and sets `CLICKHOUSE_DO_NOT_CHOWN=1` so the
  entrypoint does not fail on virtiofs binds. `self-host.sh` chmod `a+rwx` on
  data/log dirs. Host native port is `18123` (avoids `:9000` conflicts).
- Redis: named volume `langfuse-redis-data` (virtiofs bind + entrypoint
  `chown` → EPERM). Healthcheck uses `REDISCLI_AUTH`.
- Disk: host Data volume must keep **>~1% free** (~10 GiB on a 1 TB disk) or
  MinIO returns `XMinioStorageFull` and ingestion/seed fail even with an empty
  bucket (virtiofs Use%=100). Free space before `self-host.sh up` / seed.
- Smoke: `up` waits for `http://127.0.0.1:3000/api/public/health` (or
  `bash scripts/langfuse/self-host.sh smoke`). Verified on Apple Container
  with host PG + IP inject + 4 GiB web.
- Disk: ClickHouse + MinIO grow with traces — prune data dirs deliberately; never
  recursive `find` from Phenotype root.
- Secrets: `deploy/langfuse/.env` is gitignored; rotate `NEXTAUTH_SECRET`,
  `ENCRYPTION_KEY`, DB/MinIO passwords after `init`.
- Stop: `bash scripts/langfuse/self-host.sh down`
