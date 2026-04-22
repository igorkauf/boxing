#!/usr/bin/env python3
"""Dump per-frame warp components + centroid projection for a session
so we can see where/how the drift accumulates."""
import json, math, sys
from pathlib import Path
import numpy as np

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

TARGET_CLIP = "box_test.mov"

sessions_dir = BASE / "sessions_data"
target = None
for d in sorted(sessions_dir.iterdir()):
    if not d.is_dir(): continue
    m = d / "meta.json"
    if not m.exists(): continue
    meta = json.loads(m.read_text())
    if meta.get("filename") == TARGET_CLIP:
        target = d
        break

if target is None:
    print(f"No session for {TARGET_CLIP}")
    sys.exit(1)

arena = json.loads((target / "arena.json").read_text())
warps = arena.get("warps_to_ref", [])
centroid = arena.get("centroid")
ref_idx = arena.get("ref_index", 0)
fw = arena.get("frame_w", 0)
fh = arena.get("frame_h", 0)
nk = arena.get("n_keyframes", "?")

print(f"Session: {target.name}  clip: {TARGET_CLIP}")
print(f"frame size: {fw}x{fh}  ref_index: {ref_idx}  n_keyframes: {nk}")
print(f"centroid (ref coords): ({centroid[0]:.1f}, {centroid[1]:.1f})")
print()

# Per-warp stats
print(f"{'fi':>5}  {'tx':>8}  {'ty':>8}  {'scale':>6}  {'rot°':>6}  {'c_img_y':>9}")
print("-" * 50)

def decompose(M):
    """Partial affine: [[s·cos, -s·sin, tx], [s·sin, s·cos, ty]]"""
    a, b, tx = M[0]
    c, d, ty = M[1]
    s = math.hypot(a, b)
    rot = math.degrees(math.atan2(b, a))
    return tx, ty, s, rot

def project(M, pt):
    M3 = np.vstack([M, [0, 0, 1]])
    Mi = np.linalg.inv(M3)[:2]
    x, y = pt
    return (Mi[0,0]*x + Mi[0,1]*y + Mi[0,2],
            Mi[1,0]*x + Mi[1,1]*y + Mi[1,2])

nw = len(warps)
for i in range(0, nw, max(1, nw // 40)):   # ~40 samples
    M = np.array(warps[i], dtype=np.float64)
    tx, ty, s, rot = decompose(M)
    cx_img, cy_img = project(M, centroid) if centroid else (0, 0)
    marker = ""
    if cy_img > fh:       marker = "  (centroid BELOW frame)"
    elif cy_img < 0:      marker = "  (centroid ABOVE frame)"
    elif cx_img < 0:      marker = "  (centroid LEFT of frame)"
    elif cx_img > fw:     marker = "  (centroid RIGHT of frame)"
    print(f"{i:>5}  {tx:>+8.1f}  {ty:>+8.1f}  {s:>6.3f}  {rot:>+6.2f}  {cy_img:>+9.1f}{marker}")

# Also summarize end-of-clip state
print()
for i in (0, ref_idx, nw - 1):
    if i < 0 or i >= nw: continue
    M = np.array(warps[i], dtype=np.float64)
    tx, ty, s, rot = decompose(M)
    cx_img, cy_img = project(M, centroid) if centroid else (0, 0)
    kind = "start" if i == 0 else ("ref" if i == ref_idx else "end")
    print(f"  {kind:>5}  fi={i:>5}  tx={tx:+.1f} ty={ty:+.1f} scale={s:.3f} rot={rot:+.2f}°  "
          f"centroid→ ({cx_img:+.0f}, {cy_img:+.0f})")
