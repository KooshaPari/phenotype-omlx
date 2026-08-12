#!/usr/bin/env bash
# Shared Apple Container lifecycle guard for Harbor operators.
#
# Apple Container intentionally requires an explicit `container system start`;
# this helper makes that prerequisite deterministic and fails with actionable
# output instead of letting Harbor surface an opaque XPC/build error.

ensure_apple_container_service() {
  local container_bin="${CONTAINER_BIN:-container}"
  # `status` is a readonly special parameter in zsh; use a neutral name so
  # this sourced helper is safe from both bash and zsh operator shells.
  local service_status

  if ! command -v "$container_bin" >/dev/null 2>&1 && [[ ! -x "$container_bin" ]]; then
    echo "ERROR: Apple Container CLI not found: $container_bin" >&2
    echo "  Install Apple Container and run: container system start" >&2
    return 2
  fi

  service_status="$($container_bin system status 2>&1 || true)"
  if ! grep -Eq '^[[:space:]]*status[[:space:]]+running([[:space:]]|$)' <<<"$service_status"; then
    echo "Apple Container service is not running; starting it" >&2
    if ! "$container_bin" system start; then
      echo "ERROR: failed to start Apple Container service" >&2
      echo "$service_status" >&2
      return 1
    fi
    service_status="$($container_bin system status 2>&1 || true)"
  fi

  if ! grep -Eq '^[[:space:]]*status[[:space:]]+running([[:space:]]|$)' <<<"$service_status"; then
    echo "ERROR: Apple Container service is not running after start" >&2
    echo "$service_status" >&2
    echo "  Inspect with: container system logs" >&2
    return 1
  fi
}
