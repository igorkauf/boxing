#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_sam2.sh  —  Install SAM2 into a dedicated Python 3.10+ venv
#
# SAM2 requires Python >= 3.10, so it lives in its own venv (sam2_venv/)
# alongside the existing Flask venv.  The Flask app calls sam2_tracker.py
# via subprocess, so the two environments never need to share packages.
#
# Usage:   bash setup_sam2.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAM2_VENV="$SCRIPT_DIR/sam2_venv"
CKPT_DIR="$SCRIPT_DIR/sam2_checkpoints"

echo "=== Boxing App — SAM2 Setup ==="
echo ""

# ── 1. Find a Python 3.10+ interpreter ────────────────────────────────────
PY=""
for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" &>/dev/null; then
        VER=$("$candidate" -c "import sys; print(sys.version_info[:2])" 2>/dev/null)
        if "$candidate" -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" 2>/dev/null; then
            PY="$candidate"
            echo "✓ Using $PY  ($VER)"
            break
        fi
    fi
done

if [ -z "$PY" ]; then
    echo "✗ No Python 3.10+ found."
    echo "  Install it via:  brew install python@3.11"
    exit 1
fi

# ── 2. Create venv ─────────────────────────────────────────────────────────
if [ -d "$SAM2_VENV" ]; then
    echo "✓ sam2_venv already exists, skipping creation"
else
    echo "→ Creating sam2_venv ..."
    "$PY" -m venv "$SAM2_VENV"
    echo "✓ sam2_venv created"
fi

PIP="$SAM2_VENV/bin/pip"
PYTHON="$SAM2_VENV/bin/python"

# ── 3. Install SAM2 + deps ─────────────────────────────────────────────────
echo ""
echo "→ Installing SAM2 (this may take a minute) ..."
"$PIP" install --quiet --upgrade pip
"$PIP" install --quiet torch torchvision
"$PIP" install --quiet "git+https://github.com/facebookresearch/sam2.git"
"$PIP" install --quiet opencv-python-headless numpy

echo "✓ SAM2 installed"

# ── 4. Download SAM2.1 tiny checkpoint (~40 MB) ────────────────────────────
mkdir -p "$CKPT_DIR"
TINY_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
TINY_PT="$CKPT_DIR/sam2.1_hiera_tiny.pt"
SMALL_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
SMALL_PT="$CKPT_DIR/sam2.1_hiera_small.pt"

echo ""
if [ -f "$TINY_PT" ]; then
    echo "✓ sam2.1_hiera_tiny.pt already downloaded"
else
    echo "→ Downloading SAM2.1 tiny (~40 MB) ..."
    curl -L --progress-bar "$TINY_URL" -o "$TINY_PT"
    echo "✓ Downloaded sam2.1_hiera_tiny.pt"
fi

echo ""
echo "─────────────────────────────────────────────────────"
echo "Setup complete!"
echo ""
echo "→ Run the smoke test:  bash test_sam2.sh"
echo "   or manually:        $PYTHON test_sam2.py"
echo ""
echo "Optional: also grab the small model (~185 MB) for better accuracy:"
echo "   curl -L '$SMALL_URL' -o '$SMALL_PT'"
echo "─────────────────────────────────────────────────────"
