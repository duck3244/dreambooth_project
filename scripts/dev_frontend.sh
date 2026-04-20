#!/usr/bin/env bash
# Start the Vite dev server for the frontend.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT/frontend"

if [[ ! -d node_modules ]]; then
  echo "[dev_frontend] installing deps..."
  npm install
fi

exec npm run dev
