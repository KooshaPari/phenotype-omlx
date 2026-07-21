#!/usr/bin/env bash
# Langfuse self-host via Apple Container or Podman — NEVER Docker.
# shellcheck disable=SC1091
set -euo pipefail

export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin:${PATH:-}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEPLOY="$ROOT/deploy/langfuse"
COMPOSE_FILE="$DEPLOY/compose.yml"
ENV_FILE="$DEPLOY/.env"

die() { echo "error: $*" >&2; exit 1; }

# Refuse Docker even if present on PATH.
if [[ "${LANGFUSE_ALLOW_DOCKER:-}" == "1" ]]; then
  die "LANGFUSE_ALLOW_DOCKER is forbidden by Phenotype policy; unset it"
fi

resolve_runtime() {
  if command -v container >/dev/null 2>&1; then
    if container system status >/dev/null 2>&1 || container system start >/dev/null 2>&1; then
      if container compose version >/dev/null 2>&1; then
        echo "apple-container"
        return
      fi
    fi
  fi
  if command -v podman >/dev/null 2>&1; then
    if podman compose version >/dev/null 2>&1 || command -v podman-compose >/dev/null 2>&1; then
      echo "podman"
      return
    fi
  fi
  die "Need Apple Container (container compose) or Podman. Docker is not allowed."
}

compose() {
  local rt="$1"
  shift
  case "$rt" in
    apple-container)
      (cd "$DEPLOY" && container compose -f compose.yml --env-file .env "$@")
      ;;
    podman)
      if podman compose version >/dev/null 2>&1; then
        (cd "$DEPLOY" && podman compose -f compose.yml --env-file .env "$@")
      else
        (cd "$DEPLOY" && podman-compose -f compose.yml --env-file .env "$@")
      fi
      ;;
    *)
      die "unknown runtime $rt"
      ;;
  esac
}

cmd_init() {
  [[ -f "$COMPOSE_FILE" ]] || die "missing $COMPOSE_FILE"
  mkdir -p "$DEPLOY"
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

cmd_up() {
  [[ -f "$ENV_FILE" ]] || cmd_init
  local rt
  rt="$(resolve_runtime)"
  echo "runtime=$rt"
  compose "$rt" up -d
  echo "Langfuse UI: http://127.0.0.1:3000"
  echo "Point cockpit: LANGFUSE_BASE_URL=http://127.0.0.1:3000"
}

cmd_down() {
  local rt
  rt="$(resolve_runtime)"
  compose "$rt" down
}

cmd_status() {
  local rt
  rt="$(resolve_runtime)"
  echo "runtime=$rt"
  compose "$rt" ps || true
  if curl -fsS -m 5 http://127.0.0.1:3000/api/public/health >/dev/null 2>&1; then
    echo "health: ok http://127.0.0.1:3000/api/public/health"
  else
    echo "health: not ready (web may still be starting)"
  fi
}

cmd_logs() {
  local rt
  rt="$(resolve_runtime)"
  compose "$rt" logs -f --tail=80
}

usage() {
  cat <<EOF
Usage: $0 <init|up|down|status|logs>
  init    Write deploy/langfuse/.env with random secrets
  up      Start stack (Apple Container or Podman)
  down    Stop stack
  status  Compose ps + health probe
  logs    Follow recent logs
EOF
}

main() {
  local op="${1:-}"
  case "$op" in
    init) cmd_init ;;
    up) cmd_up ;;
    down) cmd_down ;;
    status) cmd_status ;;
    logs) cmd_logs ;;
    *) usage; exit 1 ;;
  esac
}

main "$@"
