#!/bin/bash
# ──────────────────────────────────────────────
#  Boxing Analyser  ·  local dev server
#  Run from Terminal:  bash run.sh
# ──────────────────────────────────────────────

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

VENV="$SCRIPT_DIR/venv"

# ── Detect a broken/stale venv (e.g. built on a different OS) ────────────────
if [ -f "$VENV/bin/python3" ]; then
  if ! "$VENV/bin/python3" -c "import flask" 2>/dev/null; then
    echo "  Removing incompatible venv and rebuilding…"
    rm -rf "$VENV"
  fi
fi

# ── Create venv if needed ─────────────────────
if [ ! -f "$VENV/bin/python3" ]; then
  echo ""
  echo "  First run — setting up virtual environment…"
  python3 -m venv "$VENV"
  echo "  Installing dependencies (may take a few minutes)…"
  "$VENV/bin/pip" install --upgrade pip --quiet
  "$VENV/bin/pip" install -r requirements.txt
  echo ""
  echo "  ✅  Dependencies installed."
fi

# ── Friendly banner ───────────────────────────
echo ""
echo "  ╔══════════════════════════════════════╗"
echo "  ║   🥊  Boxing Analyser is running     ║"
echo "  ║                                      ║"
echo "  ║   Open in Safari:                    ║"
echo "  ║   http://localhost:5001              ║"
echo "  ║                                      ║"
echo "  ║   Press  Ctrl+C  to stop             ║"
echo "  ╚══════════════════════════════════════╝"
echo ""

# ── Launch ───────────────────────────────────
"$VENV/bin/python3" app.py
