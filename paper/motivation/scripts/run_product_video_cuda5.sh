#!/usr/bin/env bash
# Complete Product/Video diagnostics; reuse the existing Chinese-text results.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export GPU=5
export TASKS="product video"
export RUN_TAG="${RUN_TAG:-v1_seed42}"

echo "[motivation] GPU=$GPU; tasks=$TASKS; run_tag=$RUN_TAG"
for stage in preflight e2 e3 plot; do
  echo "[motivation] stage=$stage"
  bash "$SCRIPT_DIR/run_task.sh" "$stage"
done
