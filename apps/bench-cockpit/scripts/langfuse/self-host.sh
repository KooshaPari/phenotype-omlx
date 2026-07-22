#!/usr/bin/env bash
# Langfuse self-host via Apple Container (or Podman) — NEVER Docker.
# Prefer: `container compose` plugin, else standalone `container-compose`.
# shellcheck disable=SC1091
set -euo pipefail

# Stable system tools first (openssl, curl, mkdir) before Homebrew/docker noise.
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin:${HOME}/.local/bin:${PATH:-}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEPLOY="$ROOT/deploy/langfuse"
COMPOSE_FILE="$DEPLOY/compose.yml"
ENV_FILE="$DEPLOY/.env"
# Durable Phenotype path — never /tmp (agent-infra durability).
# Prefer ~/.local/share (no spaces) so compose bind mounts stay reliable.
DEFAULT_DATA_DIR="${HOME}/.local/share/phenotype/langfuse"
DATA_DIR="${LANGFUSE_DATA_DIR:-$DEFAULT_DATA_DIR}"
PROJECT_NAME="${LANGFUSE_COMPOSE_PROJECT:-langfuse}"
HEALTH_URL="${LANGFUSE_HEALTH_URL:-http://127.0.0.1:3000/api/public/health}"
SMOKE_RETRIES="${LANGFUSE_SMOKE_RETRIES:-60}"
SMOKE_SLEEP_SEC="${LANGFUSE_SMOKE_SLEEP_SEC:-2}"

die() { echo "error: $*" >&2; exit 1; }

refuse_docker_escape() {
  if [[ "${LANGFUSE_ALLOW_DOCKER:-}" == "1" ]]; then
    die "LANGFUSE_ALLOW_DOCKER is forbidden by Phenotype policy; unset it"
  fi
  if [[ "${COMPOSE_DOCKER_CLI_BUILD:-}" == "1" ]] || [[ -n "${DOCKER_HOST:-}" && "${LANGFUSE_FORCE_APPLE:-}" != "1" ]]; then
    # DOCKER_HOST alone is common; only warn via resolve. Hard-block explicit docker compose usage below.
    :
  fi
}

guard_data_dir() {
  local resolved
  # Prefer physical path so /tmp → /private/tmp is caught.
  if command -v python3 >/dev/null 2>&1; then
    resolved="$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$DATA_DIR")"
  else
    resolved="$(cd "$(dirname "$DATA_DIR")" 2>/dev/null && pwd -P)/$(basename "$DATA_DIR")" || resolved="$DATA_DIR"
  fi
  case "$resolved" in
    /tmp|/tmp/*|/private/tmp|/private/tmp/*|*/tmp|*/tmp/*)
      die "REFUSED: LANGFUSE_DATA_DIR=$DATA_DIR resolves under /tmp ($resolved). Use a durable path (default: $DEFAULT_DATA_DIR)."
      ;;
  esac
}

ensure_data_dirs() {
  guard_data_dir
  mkdir -p \
    "$DATA_DIR/postgres" \
    "$DATA_DIR/clickhouse" \
    "$DATA_DIR/clickhouse-logs" \
    "$DATA_DIR/minio" \
    "$DATA_DIR/redis"
  # ClickHouse image runs as uid 101; Apple Container cannot chown host binds to
  # 101 without interactive sudo. World-writable dirs let the container write.
  chmod a+rwx "$DATA_DIR/clickhouse" "$DATA_DIR/clickhouse-logs" || true
  export LANGFUSE_DATA_DIR="$DATA_DIR"
  echo "data_dir=$DATA_DIR"
}

docker_only_present() {
  command -v docker >/dev/null 2>&1 || return 1
  # Apple container usable?
  if command -v container >/dev/null 2>&1; then
    if container_compose_available; then
      return 1
    fi
  fi
  if command -v container-compose >/dev/null 2>&1; then
    return 1
  fi
  if command -v podman >/dev/null 2>&1; then
    if podman compose version >/dev/null 2>&1 || command -v podman-compose >/dev/null 2>&1; then
      return 1
    fi
  fi
  return 0
}

container_compose_available() {
  # Official plugin: `container compose`
  if command -v container >/dev/null 2>&1; then
    if container compose version >/dev/null 2>&1; then
      return 0
    fi
  fi
  return 1
}

ensure_apple_runtime() {
  command -v container >/dev/null 2>&1 || die "Apple Container CLI (container) not found on PATH"
  if ! container system status >/dev/null 2>&1; then
    echo "starting Apple Container system services..."
    # Non-interactive: confirm default kata kernel if prompted.
    if ! printf 'Y\n' | container system start >/dev/null; then
      die "container system start failed"
    fi
  fi
}

# Prefetch ClickHouse before compose up. Apple Container drops XPC /
# HTTPClientError.remoteConnectionClosed when unpacking large CH images under
# concurrent load; :latest (~820MB) hangs — compose pins :24.8.
preflight_clickhouse_image() {
  local ref="docker.io/clickhouse/clickhouse-server:24.8"
  if container image list 2>/dev/null | grep -qE 'clickhouse/clickhouse-server[[:space:]]+24\.8'; then
    echo "clickhouse_image=cached ($ref)"
    return 0
  fi
  echo "preflight: pulling $ref (no concurrent container runs — Apple Container XPC)"
  local attempt
  for attempt in 1 2 3; do
    if container image pull "$ref"; then
      echo "clickhouse_image=pulled ($ref)"
      return 0
    fi
    echo "preflight: pull attempt $attempt failed (remoteConnectionClosed/XPC?) — retrying after system settle"
    sleep $((attempt * 5))
  done
  die "BLOCKER: failed to pull $ref after 3 attempts.
Workaround: printf 'Y\\n' | container system stop; sleep 2; printf 'Y\\n' | container system start
then: container image pull $ref
Do not use clickhouse-server:latest on Apple Container (unpack hang)."
}

resolve_runtime() {
  refuse_docker_escape

  if docker_only_present; then
    die "Docker is present but Phenotype forbids it for Langfuse. Install Apple container-compose (plugin or ~/.local/bin/container-compose) or Podman. Do not use docker compose."
  fi

  if command -v container >/dev/null 2>&1; then
    ensure_apple_runtime
    if container_compose_available; then
      echo "apple-container"
      return
    fi
    if command -v container-compose >/dev/null 2>&1; then
      echo "apple-container-standalone"
      return
    fi
    die "Apple Container is running but compose is missing.
Install one of:
  • Plugin: https://github.com/flaticols/container-compose/releases (sudo installer → container compose)
  • Standalone: copy compose binary to ~/.local/bin/container-compose
Docker is not allowed as a fallback."
  fi

  if command -v container-compose >/dev/null 2>&1; then
    die "container-compose found but Apple Container CLI (container) is missing; install apple/container first"
  fi

  if command -v podman >/dev/null 2>&1; then
    if podman compose version >/dev/null 2>&1 || command -v podman-compose >/dev/null 2>&1; then
      echo "podman"
      return
    fi
  fi

  if command -v docker >/dev/null 2>&1; then
    die "Only Docker was found. Phenotype policy: NEVER Docker for Langfuse. Install Apple Container + container-compose (or Podman)."
  fi

  die "Need Apple Container (container compose / container-compose) or Podman. Docker is not allowed."
}

compose() {
  local rt="$1"
  shift
  # container-compose wants: <subcommand> [flags ...] (not docker-style global flags first).
  # Drop docker-style -d; Apple container-compose detaches by default.
  local -a raw=("$@")
  local -a filtered=()
  local a sub rest_start=0
  for a in "${raw[@]}"; do
    case "$a" in
      -d|--detach) ;;
      *) filtered+=("$a") ;;
    esac
  done
  [[ ${#filtered[@]} -ge 1 ]] || die "compose: missing subcommand"
  sub="${filtered[0]}"
  rest_start=1

  local -a inv=()
  case "$rt" in
    apple-container)
      inv=(container compose)
      ;;
    apple-container-standalone)
      inv=(container-compose)
      ;;
    podman)
      if podman compose version >/dev/null 2>&1; then
        inv=(podman compose)
      else
        inv=(podman-compose)
      fi
      ;;
    *)
      die "unknown runtime $rt"
      ;;
  esac

  (
    cd "$DEPLOY"
    export LANGFUSE_DATA_DIR="$DATA_DIR"
    # Subcommand first for apple-container-standalone; docker/podman accept either.
    if [[ "$rt" == "apple-container-standalone" || "$rt" == "apple-container" ]]; then
      "${inv[@]}" "$sub" -p "$PROJECT_NAME" -f compose.yml --env-file .env "${filtered[@]:rest_start}"
    else
      "${inv[@]}" -p "$PROJECT_NAME" -f compose.yml --env-file .env "$sub" "${filtered[@]:rest_start}"
    fi
  )
}

cmd_init() {
  [[ -f "$COMPOSE_FILE" ]] || die "missing $COMPOSE_FILE"
  mkdir -p "$DEPLOY"
  ensure_data_dirs
  if [[ -f "$ENV_FILE" ]]; then
    echo "exists: $ENV_FILE (not overwritten)"
    return
  fi
  local salt enc next
  salt="$(openssl rand -hex 16)"
  enc="$(openssl rand -hex 32)"
  next="$(openssl rand -hex 32)"
  cat >"$ENV_FILE" <<EOF
# gitignored — Langfuse self-host secrets
NEXTAUTH_URL=http://127.0.0.1:3000
NEXTAUTH_SECRET=${next}
SALT=${salt}
ENCRYPTION_KEY=${enc}
TELEMETRY_ENABLED=false
DATABASE_URL=postgresql://postgres:postgres@postgres:5432/postgres
POSTGRES_PASSWORD=postgres
CLICKHOUSE_PASSWORD=clickhouse
REDIS_AUTH=myredissecret
MINIO_ROOT_PASSWORD=miniosecret
LANGFUSE_S3_EVENT_UPLOAD_SECRET_ACCESS_KEY=miniosecret
LANGFUSE_S3_MEDIA_UPLOAD_SECRET_ACCESS_KEY=miniosecret
LANGFUSE_S3_BATCH_EXPORT_SECRET_ACCESS_KEY=miniosecret
# Durable host bind roots (also exported by self-host.sh)
LANGFUSE_DATA_DIR=${DATA_DIR}
# Optional headless project bootstrap:
# LANGFUSE_INIT_ORG_ID=bench
# LANGFUSE_INIT_ORG_NAME=bench
# LANGFUSE_INIT_PROJECT_NAME=bench-cockpit
# LANGFUSE_INIT_PROJECT_PUBLIC_KEY=pk-lf-local
# LANGFUSE_INIT_PROJECT_SECRET_KEY=sk-lf-local
# LANGFUSE_INIT_USER_EMAIL=local@phenotype.local
# LANGFUSE_INIT_USER_NAME=local
# LANGFUSE_INIT_USER_PASSWORD=changeme
EOF
  chmod 600 "$ENV_FILE"
  echo "wrote $ENV_FILE — rotate passwords before any shared use"
}

cmd_smoke() {
  local i code body
  echo "smoke: probing $HEALTH_URL (retries=$SMOKE_RETRIES)"
  for ((i = 1; i <= SMOKE_RETRIES; i++)); do
    if body="$(curl -fsS -m 5 "$HEALTH_URL" 2>/dev/null)"; then
      echo "smoke: ok ($body)"
      return 0
    fi
    code="$(curl -sS -m 5 -o /dev/null -w '%{http_code}' "$HEALTH_URL" 2>/dev/null || echo 000)"
    echo "smoke: attempt $i/$SMOKE_RETRIES http=$code — waiting ${SMOKE_SLEEP_SEC}s"
    sleep "$SMOKE_SLEEP_SEC"
  done
  die "smoke FAILED: Langfuse health not ready at $HEALTH_URL after $SMOKE_RETRIES attempts"
}

cmd_up() {
  [[ -f "$ENV_FILE" ]] || cmd_init
  ensure_data_dirs
  # Keep .env LANGFUSE_DATA_DIR aligned with runtime export.
  if grep -q '^LANGFUSE_DATA_DIR=' "$ENV_FILE" 2>/dev/null; then
    local tmp="$DEPLOY/.env.tmp.$$"
    awk -v d="$DATA_DIR" '
      BEGIN { done=0 }
      /^LANGFUSE_DATA_DIR=/ { print "LANGFUSE_DATA_DIR=" d; done=1; next }
      { print }
      END { if (!done) print "LANGFUSE_DATA_DIR=" d }
    ' "$ENV_FILE" >"$tmp"
    mv "$tmp" "$ENV_FILE"
    chmod 600 "$ENV_FILE"
  else
    echo "LANGFUSE_DATA_DIR=$DATA_DIR" >>"$ENV_FILE"
  fi

  local rt
  rt="$(resolve_runtime)"
  echo "runtime=$rt"
  case "$rt" in
    apple-container|apple-container-standalone)
      preflight_clickhouse_image
      # Ensure compose project network exists before multi-service up.
      container network create "${PROJECT_NAME}-default" >/dev/null 2>&1 || true
      ;;
  esac
  compose "$rt" up
  echo "Langfuse UI: http://127.0.0.1:3000"
  echo "Point cockpit: LANGFUSE_BASE_URL=http://127.0.0.1:3000"
  cmd_smoke
}

cmd_down() {
  ensure_data_dirs
  local rt
  rt="$(resolve_runtime)"
  compose "$rt" down
}

cmd_status() {
  ensure_data_dirs
  local rt
  rt="$(resolve_runtime)"
  echo "runtime=$rt"
  echo "data_dir=$DATA_DIR"
  compose "$rt" ps || true
  if curl -fsS -m 5 "$HEALTH_URL" >/dev/null 2>&1; then
    echo "health: ok $HEALTH_URL"
  else
    echo "health: not ready (web may still be starting) — run: $0 smoke"
  fi
}

cmd_logs() {
  local rt
  rt="$(resolve_runtime)"
  compose "$rt" logs --tail=80
}

usage() {
  cat <<EOF
Usage: $0 <init|up|down|status|logs|smoke>
  init    Write deploy/langfuse/.env + durable data dirs
  up      Start stack (Apple Container / container-compose or Podman) + smoke :3000
  down    Stop stack
  status  Compose ps + health probe
  logs    Show recent logs
  smoke   Loud-fail until http://127.0.0.1:3000/api/public/health is OK

Env:
  LANGFUSE_DATA_DIR   durable bind root (default: ~/.local/share/phenotype/langfuse)
  LANGFUSE_SMOKE_RETRIES / LANGFUSE_SMOKE_SLEEP_SEC
Never Docker. Never /tmp for data.
EOF
}

main() {
  refuse_docker_escape
  local op="${1:-}"
  case "$op" in
    init) cmd_init ;;
    up) cmd_up ;;
    down) cmd_down ;;
    status) cmd_status ;;
    logs) cmd_logs ;;
    smoke) cmd_smoke ;;
    *) usage; exit 1 ;;
  esac
}

main "$@"
