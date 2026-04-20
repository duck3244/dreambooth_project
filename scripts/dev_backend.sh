#!/usr/bin/env bash
# Start the FastAPI backend with hot-reload.
#
# Activates the py310_pt conda env (which has torch + diffusers installed)
# and runs uvicorn with reload=True so code edits take effect without a
# manual restart.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CONDA_ENV="${DREAMBOOTH_CONDA_ENV:-py310_pt}"
HOST="${DREAMBOOTH_API_HOST:-127.0.0.1}"
PORT="${DREAMBOOTH_API_PORT:-8765}"

# Activate conda if available
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"
fi

cd "$ROOT"
export PYTHONPATH="$ROOT"

exec uvicorn backend.api.app:app \
  --host "$HOST" \
  --port "$PORT" \
  --reload \
  --reload-dir "$ROOT/backend"
