#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSTEM_CODE="$SCRIPT_DIR/SystemCode"
BACKEND_DIR="$SYSTEM_CODE/backend"
FRONTEND_DIR="$SYSTEM_CODE/frontend"
VENV_DIR="$BACKEND_DIR/.venv"
MODELS_DIR="$SYSTEM_CODE/data/training"

REINSTALL=0
if [ "${1:-}" = "--reinstall" ]; then
  REINSTALL=1
fi

command -v python3 >/dev/null 2>&1 || { echo "python3 not found. Install Python 3.8+ and retry."; exit 1; }
command -v node    >/dev/null 2>&1 || { echo "node not found. Install Node.js 20+ and retry."; exit 1; }
command -v npm     >/dev/null 2>&1 || { echo "npm not found. Install npm (bundled with Node.js) and retry."; exit 1; }

for m in model_resnet50.pth model_yolo.pt model_efficientnetb0.pth; do
  if [ ! -f "$MODELS_DIR/$m" ]; then
    echo "Missing model file: $MODELS_DIR/$m"
    echo "Your checkout is incomplete. Ensure the repo (including tracked weights under SystemCode/data/training/) is fully pulled."
    exit 1
  fi
done

if [ ! -d "$VENV_DIR" ]; then
  echo "[setup] creating backend virtualenv"
  python3 -m venv "$VENV_DIR"
fi

VENV_PY="$VENV_DIR/bin/python"
REQ_FILE="$BACKEND_DIR/requirements.txt"
STAMP_FILE="$VENV_DIR/.req-stamp"
if command -v shasum >/dev/null 2>&1; then
  REQ_HASH="$(shasum "$REQ_FILE" | awk '{print $1}')"
else
  REQ_HASH="$(sha1sum "$REQ_FILE" | awk '{print $1}')"
fi

if [ "$REINSTALL" = "1" ] || [ ! -f "$STAMP_FILE" ] || [ "$(cat "$STAMP_FILE" 2>/dev/null)" != "$REQ_HASH" ]; then
  echo "[setup] installing backend dependencies"
  "$VENV_PY" -m pip install --upgrade pip
  "$VENV_PY" -m pip install -r "$REQ_FILE"
  echo "$REQ_HASH" > "$STAMP_FILE"
else
  echo "[setup] backend dependencies up to date"
fi

if [ "$REINSTALL" = "1" ] || [ ! -d "$FRONTEND_DIR/node_modules" ] || [ "$FRONTEND_DIR/package-lock.json" -nt "$FRONTEND_DIR/node_modules" ]; then
  echo "[setup] installing frontend dependencies"
  ( cd "$FRONTEND_DIR" && npm install )
else
  echo "[setup] frontend dependencies up to date"
fi

echo "[run] starting backend on http://localhost:8000"
( cd "$BACKEND_DIR" && exec "$VENV_PY" -m app.main ) > >(awk '{ print "[backend] " $0; fflush() }') 2>&1 &
BACKEND_PID=$!

echo "[run] starting frontend on http://localhost:5173"
( cd "$FRONTEND_DIR" && exec npm run dev ) > >(awk '{ print "[frontend] " $0; fflush() }') 2>&1 &
FRONTEND_PID=$!

cleanup() {
  echo
  echo "[run] shutting down"
  pkill -P "$BACKEND_PID" 2>/dev/null || true
  pkill -P "$FRONTEND_PID" 2>/dev/null || true
  kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
  wait "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
  exit 0
}
trap cleanup INT TERM

echo
echo "  Backend:  http://localhost:8000"
echo "  Frontend: http://localhost:5173"
echo "  Press Ctrl+C to stop both."
echo

wait
