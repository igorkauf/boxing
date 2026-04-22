"""
diagnostics.py — assemble per-frame overlay data for the Lab diagnostics viewer.

Each pipeline stage contributes a "layer" of drawable items per frame. The
viewer (canvas over the compressed video in lab.html) toggles layers on and
off without re-rendering the video.

Item shapes (all coords in compressed-video pixels):
    {"type": "box",      "box": [x1,y1,x2,y2], "color": "#RRGGBB", "label": "ME"}
    {"type": "skeleton", "kps": [[x,y,conf], ... 17 entries ...], "color": "#RRGGBB"}
    {"type": "point",    "xy":  [x,y], "color": "#RRGGBB", "radius": 4}
    {"type": "line",     "a":   [x,y], "b": [x,y], "color": "#RRGGBB"}
    {"type": "text",     "xy":  [x,y], "text": "...", "color": "#RRGGBB"}

A layer is a list of frames; each frame has a timestamp and a list of items:
    {"label": "Fighter boxes",
     "frames": [{"t": 0.033, "items": [ ... ]}, ...]}

This module is intentionally minimal — no drawing, no rendering. The client
handles all rendering on canvas.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

COL_ME = "#ff8c00"   # orange
COL_OP = "#3c8ddc"   # blue
COL_NEUTRAL = "#888888"


def build_layers_from_enriched(enriched: dict) -> dict[str, Any]:
    """
    Turn sam2_enriched.json into the layer bundle the viewer consumes.
    Currently emits two layers ("boxes" and "skeleton") as the "baseline"
    view of what the existing pipeline sees. Later stages will append more.
    """
    frames = enriched.get("frames", [])
    fps_pose = enriched.get("fps_pose", 15.0)
    fw = enriched.get("frame_w", 0)
    fh = enriched.get("frame_h", 0)

    boxes_frames = []
    skel_frames = []
    clinch_frames = []

    for f in frames:
        t = f.get("time_s", 0.0)
        box_items = []
        skel_items = []

        for pfx, col, label in (("my", COL_ME, "ME"), ("op", COL_OP, "OP")):
            bb = f.get(f"{pfx}_bbox")
            if bb:
                box_items.append({
                    "type": "box",
                    "box": [float(v) for v in bb],
                    "color": col,
                    "label": label,
                })
            kps = f.get(f"{pfx}_kps")
            if kps:
                skel_items.append({
                    "type": "skeleton",
                    "kps": kps,
                    "color": col,
                })

        boxes_frames.append({"t": t, "items": box_items})
        skel_frames.append({"t": t, "items": skel_items})

        if f.get("clinch"):
            clinch_frames.append({"t": t, "items": [
                {"type": "text", "xy": [0.02, 0.10],
                 "text": "CLINCH", "color": "#ff3333"}
            ]})

    # Re-ID swap markers — a brief badge at the moment of each detected swap.
    swap_frames = []
    for ev in enriched.get("swap_events", []):
        swap_frames.append({
            "t": ev.get("time_s", 0.0),
            "items": [{
                "type": "text", "xy": [0.02, 0.14],
                "text": f"RE-ID SWAP ({ev.get('reason', '?')})",
                "color": "#ffaa00",
            }],
        })

    return {
        "frame_w": fw,
        "frame_h": fh,
        "fps_pose": fps_pose,
        "coords":  "normalized",   # all items expressed in [0,1] × [0,1]
        "layers": {
            "boxes": {
                "label": "Fighter boxes",
                "default_on": True,
                "frames": boxes_frames,
            },
            "skeleton": {
                "label": "Pose skeleton",
                "default_on": True,
                "frames": skel_frames,
            },
            "clinch": {
                "label": "Clinch flag",
                "default_on": False,
                "frames": clinch_frames,
            },
            "reid_swaps": {
                "label": f"Re-ID swaps ({len(swap_frames)})",
                "default_on": True,
                "sparse": True,
                "frames": swap_frames,
            },
        },
    }


def write_layers(session_dir: Path, bundle: dict) -> Path:
    """Persist the diagnostics bundle next to sam2_enriched.json for serving."""
    out = session_dir / "diagnostics.json"
    out.write_text(json.dumps(bundle, separators=(",", ":")))
    return out


def refresh_from_enriched(session_dir: Path) -> bool:
    """Rebuild diagnostics.json from sam2_enriched.json. Returns True on success."""
    enriched_path = session_dir / "sam2_enriched.json"
    if not enriched_path.exists():
        # Seed an empty bundle so other stages (scene-ref etc.) can still merge
        # into something, even before SAM2 has run.
        write_layers(session_dir, {
            "frame_w": 0, "frame_h": 0, "fps_pose": 0,
            "layers": {},
        })
        return False
    try:
        enriched = json.loads(enriched_path.read_text())
    except json.JSONDecodeError:
        return False
    bundle = build_layers_from_enriched(enriched)
    write_layers(session_dir, bundle)
    return True


# ── Arena layer ───────────────────────────────────────────────────────────────
_ARENA_COLOR = "#ff8c00"
_CENTER_COLOR = "#ff4040"


def _invert_affine(M):
    import numpy as np
    M3 = np.vstack([M, [0, 0, 1]])
    return np.linalg.inv(M3)[:2]


def _apply_affine(M, pts):
    import numpy as np
    if not pts:
        return []
    arr = np.asarray(pts, dtype=float)
    ones = np.ones((len(arr), 1))
    return (np.concatenate([arr, ones], axis=1) @ M.T).tolist()


def merge_arena(bundle: dict, arena: dict) -> None:
    """
    Add an arena overlay layer to the diagnostics bundle. The arena geometry
    lives in reference coords; for each enriched frame we pre-project it to
    that frame's image coords (via inv(warp_to_ref)), so the client just
    looks up the nearest frame and draws — no matrix math in JS.

    One layer, on by default. Normalized [0, 1] coords so the client can
    scale them to canvas size.
    """
    import numpy as np

    if not arena.get("ok"):
        return
    W = arena.get("frame_w") or bundle.get("frame_w") or 1
    H = arena.get("frame_h") or bundle.get("frame_h") or 1
    bundle.setdefault("coords", "normalized")
    bundle["frame_w"] = W
    bundle["frame_h"] = H

    centroid   = arena.get("centroid")
    polygon    = arena.get("arena_polygon")
    warps      = arena.get("warps_to_ref", [])
    my_traj    = arena.get("my_traj_ref", [])
    op_traj    = arena.get("op_traj_ref", [])
    tier       = arena.get("confidence_tier", "none")
    vscore     = arena.get("visibility_score", 0.0)
    fps_pose   = arena.get("fps_pose") or bundle.get("fps_pose") or 15.0

    if not warps or not polygon or centroid is None:
        return

    nx = lambda x: x / float(W)
    ny = lambda y: y / float(H)

    frames = []
    for fi, W_ref_from_img in enumerate(warps):
        M_ref_from_img = np.array(W_ref_from_img, dtype=float)
        M_img_from_ref = _invert_affine(M_ref_from_img)

        items = []

        # Arena polygon (projected + normalized).
        if polygon:
            proj = _apply_affine(M_img_from_ref, polygon)
            for i in range(len(proj)):
                a = proj[i]; b = proj[(i + 1) % len(proj)]
                items.append({
                    "type": "line",
                    "a": [nx(a[0]), ny(a[1])],
                    "b": [nx(b[0]), ny(b[1])],
                    "color": _ARENA_COLOR,
                })

        # Centroid. If it projects off the visible canvas (motion-comp drift
        # on a particular frame), clamp to the nearest edge with a tag so
        # the user always sees *where* the centroid sits — silent invisible
        # rendering was the old failure mode.
        c_img = _apply_affine(M_img_from_ref, [centroid])
        if c_img:
            cx, cy = c_img[0]
            nx_c, ny_c = nx(cx), ny(cy)
            on_canvas = (-0.02 <= nx_c <= 1.02) and (-0.02 <= ny_c <= 1.02)
            if on_canvas:
                items.append({
                    "type": "point", "xy": [nx_c, ny_c],
                    "color": _CENTER_COLOR, "radius": 10,
                })
            else:
                cx_clamped = max(0.03, min(0.97, nx_c))
                cy_clamped = max(0.05, min(0.95, ny_c))
                items.append({
                    "type": "point", "xy": [cx_clamped, cy_clamped],
                    "color": _CENTER_COLOR, "radius": 12,
                })
                dx = "→" if nx_c > 1 else ("←" if nx_c < 0 else "")
                dy = "↓" if ny_c > 1 else ("↑" if ny_c < 0 else "")
                items.append({
                    "type": "text",
                    "xy": [min(cx_clamped + 0.015, 0.9), cy_clamped + 0.005],
                    "text": f"CENTER off {dx}{dy}",
                    "color": _CENTER_COLOR,
                })

        # Per-fighter current position dots.
        for traj, col in ((my_traj, COL_ME), (op_traj, COL_OP)):
            p = traj[fi] if fi < len(traj) else None
            if p is not None:
                p_img = _apply_affine(M_img_from_ref, [p])
                if p_img:
                    items.append({
                        "type": "point",
                        "xy": [nx(p_img[0][0]), ny(p_img[0][1])],
                        "color": col, "radius": 8,
                    })

        # Tier badge once per frame (top-left).
        items.append({
            "type": "text", "xy": [0.02, 0.06],
            "text": f"ARENA: {tier.upper()}  visibility {int(vscore*100)}%",
            "color": "#ffffff",
        })

        frames.append({"t": fi / fps_pose, "items": items})

    bundle["layers"]["arena"] = {
        "label":      f"Arena ({tier})",
        "default_on": True,
        "frames":     frames,
    }
