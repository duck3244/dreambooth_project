#!/usr/bin/env bash
# Launch backend + frontend dev servers together.
#
# Prints both logs interleaved to the current terminal. Ctrl+C stops both.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pids=()

cleanup() {
  trap - INT TERM EXIT
  for pid in "${pids[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

"$SCRIPT_DIR/dev_backend.sh" 2>&1 | sed -u 's/^/[api] /' &
pids+=($!)

"$SCRIPT_DIR/dev_frontend.sh" 2>&1 | sed -u 's/^/[web] /' &
pids+=($!)

wait
