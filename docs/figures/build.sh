#!/usr/bin/env bash
# Render every *.dot here to SVG + PNG (200 dpi). Needs graphviz.
set -euo pipefail
cd "$(dirname "$0")"
for f in *.dot; do
  dot -Tsvg "$f" -o "${f%.dot}.svg"
  dot -Tpng -Gdpi=200 "$f" -o "${f%.dot}.png"
done
