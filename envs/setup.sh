#!/usr/bin/env bash
# Build one virtualenv per model family. Safe to re-run; skips venvs that already exist.
#
# uv is used when present because the system python3 on macOS is 3.9, too old for the
# numpy 2.x that the core family pins. Note `uv venv` does NOT seed pip, so installs go
# through `uv pip install --python <venv>` rather than <venv>/bin/pip.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PY_VER="${PY_VER:-3.12}"
HAVE_UV=0
command -v uv >/dev/null 2>&1 && HAVE_UV=1

mkvenv() { if [ "$HAVE_UV" = 1 ]; then uv venv --python "$PY_VER" "$1" >/dev/null
           else python3 -m venv "$1"; fi }
pipin()  { if [ "$HAVE_UV" = 1 ]; then uv pip install --python "$1" "${@:2}"
           else "$1/bin/pip" install "${@:2}"; fi }

for fam in core toto moirai; do
  V=".venv-$fam"
  if [ -d "$V" ]; then echo "== $V exists, skipping"; continue; fi
  echo "== building $V"
  mkvenv "$V"
  # torch first, so each family resolves the build it actually wants instead of
  # inheriting whatever the first requirement drags in.
  pipin "$V" torch
  pipin "$V" -r "envs/$fam.txt"
  echo "== $V ready"
done

echo
echo "Verify the environments really do differ — this is the whole point:"
for fam in core toto moirai; do
  printf '%-8s ' "$fam"
  ".venv-$fam/bin/python" -c 'import torch,numpy;print("torch",torch.__version__," numpy",numpy.__version__)' 2>/dev/null || echo "(not built)"
done
