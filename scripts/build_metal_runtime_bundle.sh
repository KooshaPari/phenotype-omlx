#!/usr/bin/env bash
set -euo pipefail

# Compile every checked-in Metal kernel and link one allowlistable metallib.
# This performs no model/evaluation work.
# Usage: OUT_DIR=/tmp/omlx-metal ./scripts/build_metal_runtime_bundle.sh

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out_dir="${OUT_DIR:-$(mktemp -d /tmp/omlx-metal-runtime.XXXXXX)}"
mkdir -p "$out_dir"

metal_bin="$(xcrun --sdk macosx --find metal)"
metallib_bin="$(xcrun --sdk macosx --find metallib)"
shader_dir="$repo_root/perf-core/metal-runtime/shaders"

sources=()
while IFS= read -r source; do
  sources+=("$source")
done < <(find "$shader_dir" -maxdepth 1 -type f -name '*.metal' -print | sort)
if (( ${#sources[@]} == 0 )); then
  echo "no Metal shader sources found under $shader_dir" >&2
  exit 2
fi

air_files=()
for source in "${sources[@]}"; do
  name="$(basename "$source" .metal)"
  air="$out_dir/$name.air"
  "$metal_bin" -c "$source" -o "$air"
  air_files+=("$air")
done

bundle="$out_dir/metal-runtime.metallib"
"$metallib_bin" "${air_files[@]}" -o "$bundle"

echo "METAL_RUNTIME_METALLIB=$bundle"
echo "METAL_RUNTIME_SHADER_COUNT=${#sources[@]}"
shasum -a 256 "$bundle"
