#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_SCRIPT="${ROOT}/perf-core/concurrent-exec-cuda/cuda/build.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

cat >"${TMP}/nvcc" <<'EOF'
#!/usr/bin/env bash
echo "Cuda compilation tools, release ${FAKE_CUDA_RELEASE:-12.4}, V12.4.99"
EOF
cat >"${TMP}/cmake" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
chmod +x "${TMP}/nvcc" "${TMP}/cmake"

if ! FAKE_CUDA_RELEASE=13.0 PATH="${TMP}:${PATH}" CUDA_PROFILE=portable CUDA_ARCH='61;86' \
    "${BUILD_SCRIPT}" >/dev/null 2>&1; then
    :
else
    echo "mixed Pascal/Ampere builds must reject CUDA 13" >&2
    exit 1
fi

FAKE_CUDA_RELEASE=12.4 PATH="${TMP}:${PATH}" CUDA_PROFILE=portable CUDA_ARCH='61;86' \
    "${BUILD_SCRIPT}" >/dev/null

echo "cuda build profile checks: PASS"
