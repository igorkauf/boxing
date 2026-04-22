"""
screenshots.py — burn diagnostic overlays into frame PNGs for visual validation.

Each pipeline stage can call into this module to save a handful of representative
frames (with its overlay drawn on top) so the user can scrub through a folder
of PNGs instead of depending on live canvas rendering in the browser.

Outputs (per stage):
    cache/videos/<hash>/screenshots/<stage>/*.png     (video-content stages)
    sessions_data/<sid>/screenshots/<stage>/*.png     (session-scoped stages)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


# ── Colours (BGR) ────────────────────────────────────────────────────────────
COL_ME = (0,   130, 255)   # orange
COL_OP = (220,  60,   0)   # blue

SKEL_PAIRS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 12), (5, 11), (6, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
]


# ── Low-level drawing ────────────────────────────────────────────────────────
def _label(frame, text, x, y, color):
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = max(0.5, frame.shape[1] / 1400)
    thick = max(1, int(scale * 1.5))
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    cv2.rectangle(frame, (x - 2, y - th - 4), (x + tw + 4, y + 4), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, scale, color, thick, cv2.LINE_AA)


def _draw_box(frame, bbox_px, color, label=None):
    x1, y1, x2, y2 = [int(v) for v in bbox_px]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if label:
        _label(frame, label, x1 + 3, max(y1 - 6, 18), color)


def _draw_skeleton(frame, kps_px, color):
    for a, b in SKEL_PAIRS:
        if a >= len(kps_px) or b >= len(kps_px):
            continue
        ka = kps_px[a]
        kb = kps_px[b]
        if ka is None or kb is None:
            continue
        if len(ka) >= 3 and ka[2] < 0.3:
            continue
        if len(kb) >= 3 and kb[2] < 0.3:
            continue
        cv2.line(frame, (int(ka[0]), int(ka[1])), (int(kb[0]), int(kb[1])),
                 color, 2, cv2.LINE_AA)
    for k in kps_px:
        if k is None or (len(k) >= 3 and k[2] < 0.3):
            continue
        cv2.circle(frame, (int(k[0]), int(k[1])), 3, color, -1, cv2.LINE_AA)


# ── Frame sampler ────────────────────────────────────────────────────────────
def _grab_frame(cap: cv2.VideoCapture, time_s: float, fps: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, time_s * 1000.0))
    ok, frame = cap.read()
    return frame if ok else None


# ── Scene-reference screenshots ──────────────────────────────────────────────
def render_arena(video_path: str, arena: dict, out_dir: Path, n: int = 6) -> int:
    """
    Save n screenshots showing the motion-compensated arena. Each image shows:
      • the dilated arena polygon (filled, orange)
      • the centroid (red crosshair + "CENTER" label)
      • this frame's foot points (green dots)
      • confidence-tier badge

    The arena polygon and centroid live in reference coords; they're mapped
    back to each display frame via the inverse of that frame's warp_to_ref.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        try: old.unlink()
        except OSError: pass

    if not arena.get("ok"):
        return 0
    centroid   = arena.get("centroid")
    polygon    = arena.get("arena_polygon")
    warps      = arena.get("warps_to_ref", [])
    my_traj    = arena.get("my_traj_ref", [])
    op_traj    = arena.get("op_traj_ref", [])
    median_h   = arena.get("median_fighter_height_px", 0.0)
    tier       = arena.get("confidence_tier", "none")
    vscore     = arena.get("visibility_score", 0.0)
    n_total    = arena.get("n_enriched_frames", 0)

    if not warps or not polygon or centroid is None or n_total == 0:
        return 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    # End-biased distribution: the arena polygon is cumulative, so the most
    # informative validation frames are near the end of the clip where we can
    # confirm that the polygon covers the walked area throughout — including
    # corners the fighters only reached late.
    base_fracs = [0.10, 0.30, 0.55, 0.80, 0.92, 0.99]
    if n <= len(base_fracs):
        pick_fracs = base_fracs[:n]
    else:
        # Fallback for larger n: extend the last half.
        extra = list(np.linspace(0.80, 0.99, n - 3))
        pick_fracs = [0.10, 0.30, 0.55] + extra
    picks = sorted({int(round(f * max(n_total - 1, 1))) for f in pick_fracs})

    def _invert_affine(M):
        M3 = np.vstack([M, [0, 0, 1]])
        return np.linalg.inv(M3)[:2]

    def _apply_affine(M, pts):
        if not pts:
            return np.zeros((0, 2), dtype=np.float32)
        arr = np.asarray(pts, dtype=np.float32)
        ones = np.ones((len(arr), 1), dtype=np.float32)
        return np.concatenate([arr, ones], axis=1) @ M.T

    count = 0
    for si, fi in enumerate(picks):
        W_ref_from_img = np.array(warps[fi], dtype=np.float64)
        M_img_from_ref = _invert_affine(W_ref_from_img)

        # Need the underlying raw_fi for accurate frame grabbing — use time
        # equivalent via fi × (expected fps stride). But we don't know the
        # exact raw_fi here; approximate by pulling the frame at fi × fps_pose
        # step. Callers can also pre-resolve. Safer: iterate from start.
        # Use POS_FRAMES with an approximation: the enriched frame indices are
        # consecutive at pose stride; for screenshotting 6 frames it's fine to
        # use POS_MSEC by the midpoint time if we had it. Fallback: sequential
        # read capped.
        # Simpler: read from start up to fi.
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # Step by fi pose-frames: we read one and skip forward using FFmpeg.
        # Since pose stride is unknown here, fall back to reading by index
        # assumption: enriched frames are at time stamps fi / fps_pose.
        # For a preview screenshot this works well enough.
        # Use fps_pose if provided, else video fps.
        fps_pose = arena.get("fps_pose") or fps
        # We stored fps_pose back in enriched; for now approximate uniformly.
        t = fi * (1.0 / fps_pose)
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue

        # Arena polygon → image coords, filled.
        poly_img = _apply_affine(M_img_from_ref, polygon)
        if len(poly_img) >= 3:
            overlay = frame.copy()
            pts_i = poly_img.astype(np.int32)
            cv2.fillPoly(overlay, [pts_i], (40, 140, 255))
            cv2.addWeighted(overlay, 0.28, frame, 0.72, 0.0, dst=frame)
            cv2.polylines(frame, [pts_i], isClosed=True, color=(40, 140, 255),
                          thickness=2, lineType=cv2.LINE_AA)

        # Centroid crosshair. If drift has pushed it off-screen, draw an
        # edge-indicator arrow so the user still sees *where* the centroid
        # ended up (silently rendering it into void was the old failure).
        c_img = _apply_affine(M_img_from_ref, [centroid])
        if len(c_img) == 1:
            raw_cx, raw_cy = float(c_img[0][0]), float(c_img[0][1])
            h_px, w_px = frame.shape[:2]
            r = int(max(median_h * 0.08, 16))
            on_screen = (-r <= raw_cx <= w_px + r) and (-r <= raw_cy <= h_px + r)
            if on_screen:
                cx, cy = int(raw_cx), int(raw_cy)
                cv2.line(frame, (cx - r, cy), (cx + r, cy), (0, 80, 255), 3, cv2.LINE_AA)
                cv2.line(frame, (cx, cy - r), (cx, cy + r), (0, 80, 255), 3, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), r, (0, 80, 255), 2, cv2.LINE_AA)
                _label(frame, "CENTER", cx + r + 6, cy + 6, (0, 80, 255))
            else:
                # Clamp to the nearest edge and draw an arrow + "CENTER ↗".
                cx = int(max(r + 4, min(w_px - r - 4, raw_cx)))
                cy = int(max(r + 4, min(h_px - r - 4, raw_cy)))
                cv2.drawMarker(frame, (cx, cy), (0, 80, 255),
                               markerType=cv2.MARKER_TRIANGLE_UP,
                               markerSize=r * 2, thickness=3,
                               line_type=cv2.LINE_AA)
                dx = "→" if raw_cx > w_px else ("←" if raw_cx < 0 else "")
                dy = "↓" if raw_cy > h_px else ("↑" if raw_cy < 0 else "")
                _label(frame, f"CENTER off {dx}{dy}",
                       cx + r + 6, cy + 6, (0, 80, 255))

        # Current-frame foot dots for each fighter (trajectory positions).
        for traj, color in ((my_traj, COL_ME), (op_traj, COL_OP)):
            p = traj[fi] if fi < len(traj) else None
            if p is not None:
                p_img = _apply_affine(M_img_from_ref, [p])
                if len(p_img) == 1:
                    cv2.circle(frame, (int(p_img[0][0]), int(p_img[0][1])), 10,
                               color, -1, cv2.LINE_AA)
                    cv2.circle(frame, (int(p_img[0][0]), int(p_img[0][1])), 10,
                               (0, 0, 0), 1, cv2.LINE_AA)

        # Badges.
        _label(frame, f"ARENA  (tier: {tier.upper()}, visibility {int(vscore*100)}%)",
               16, 36, (255, 255, 255))
        _label(frame, f"t ≈ {t:.1f}s   ({si + 1}/{len(picks)})",
               16, 68, (200, 200, 200))

        out_path = out_dir / f"arena_{si:02d}_t{t:06.2f}.png"
        cv2.imwrite(str(out_path), frame)
        count += 1

    cap.release()
    return count


# ── Arena heatmap screenshot ──────────────────────────────────────────────────
def render_arena_heatmap(video_path: str, arena: dict, out_dir: Path) -> int:
    """
    Stamp all foot points (per fighter, color-coded) onto the reference frame
    as a Gaussian-blurred density heatmap. One PNG.

    Produces a visual "where did each fighter walk" map burned onto the
    reference frame's camera view. Not metrically top-down (that would need
    a homography), but consistent across frames thanks to motion comp and
    useful as validation + a gamification asset.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        try: old.unlink()
        except OSError: pass

    if not arena.get("ok"):
        return 0
    ref_index = int(arena.get("ref_index", 0))
    my_traj   = arena.get("my_traj_ref", [])
    op_traj   = arena.get("op_traj_ref", [])
    warps     = arena.get("warps_to_ref", [])
    centroid  = arena.get("centroid")
    polygon   = arena.get("arena_polygon")
    median_s  = arena.get("median_scale_px") or arena.get("median_fighter_height_px") or 100
    fps_pose  = arena.get("fps_pose") or 15.0
    tier      = arena.get("confidence_tier", "none")
    my_vis    = arena.get("my_visibility_score", 0.0)
    op_vis    = arena.get("op_visibility_score", 0.0)

    if ref_index >= len(warps) or not my_traj or not op_traj:
        return 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    t_ref = ref_index / fps_pose
    cap.set(cv2.CAP_PROP_POS_MSEC, t_ref * 1000.0)
    ok, ref_frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, ref_frame = cap.read()
    cap.release()
    if not ok:
        return 0

    h, w = ref_frame.shape[:2]

    # Project reference-coord points into the reference frame's image coords.
    # warps_to_ref[ref_index] is (by construction) ≈ identity, but compose
    # inverses anyway to be safe.
    M_ref_from_img = np.array(warps[ref_index], dtype=np.float64)
    M_img_from_ref = np.linalg.inv(np.vstack([M_ref_from_img, [0, 0, 1]]))[:2]

    def _to_img(pt_ref):
        if pt_ref is None:
            return None
        x = M_img_from_ref[0, 0] * pt_ref[0] + M_img_from_ref[0, 1] * pt_ref[1] + M_img_from_ref[0, 2]
        y = M_img_from_ref[1, 0] * pt_ref[0] + M_img_from_ref[1, 1] * pt_ref[1] + M_img_from_ref[1, 2]
        return (x, y)

    # Stamp each fighter's foot points onto a float accumulator, Gaussian-
    # blur to diffuse them into a smooth density, normalize, then colorize.
    my_canvas = np.zeros((h, w), dtype=np.float32)
    op_canvas = np.zeros((h, w), dtype=np.float32)
    my_count = 0
    op_count = 0
    for p in my_traj:
        img = _to_img(p)
        if img is None:
            continue
        x, y = int(round(img[0])), int(round(img[1]))
        if 0 <= x < w and 0 <= y < h:
            my_canvas[y, x] += 1.0
            my_count += 1
    for p in op_traj:
        img = _to_img(p)
        if img is None:
            continue
        x, y = int(round(img[0])), int(round(img[1]))
        if 0 <= x < w and 0 <= y < h:
            op_canvas[y, x] += 1.0
            op_count += 1

    kernel = max(25, int(median_s * 0.45))
    if kernel % 2 == 0:
        kernel += 1
    my_blur = cv2.GaussianBlur(my_canvas, (kernel, kernel), 0)
    op_blur = cv2.GaussianBlur(op_canvas, (kernel, kernel), 0)

    def _norm(a):
        mx = float(a.max())
        return a / mx if mx > 1e-6 else a
    my_blur = _norm(my_blur)
    op_blur = _norm(op_blur)

    out = ref_frame.astype(np.float32)
    for color, heat in ((COL_ME, my_blur), (COL_OP, op_blur)):
        strength = np.clip(heat * 0.75, 0.0, 0.75)[..., None]
        col_layer = np.broadcast_to(np.array(color, dtype=np.float32),
                                    out.shape)
        out = out * (1.0 - strength) + col_layer * strength
    out = np.clip(out, 0, 255).astype(np.uint8)

    # Overlay arena outline + centroid for anchoring.
    if polygon and len(polygon) >= 3:
        poly_img = []
        for p in polygon:
            ix = _to_img(p)
            if ix is None:
                continue
            poly_img.append([int(ix[0]), int(ix[1])])
        if len(poly_img) >= 3:
            cv2.polylines(out, [np.array(poly_img, dtype=np.int32)],
                          isClosed=True, color=(40, 140, 255),
                          thickness=2, lineType=cv2.LINE_AA)
    if centroid:
        c = _to_img(centroid)
        if c is not None:
            cx, cy = int(c[0]), int(c[1])
            r = int(max(median_s * 0.08, 16))
            cv2.line(out, (cx - r, cy), (cx + r, cy),
                     (0, 80, 255), 3, cv2.LINE_AA)
            cv2.line(out, (cx, cy - r), (cx, cy + r),
                     (0, 80, 255), 3, cv2.LINE_AA)
            cv2.circle(out, (cx, cy), r, (0, 80, 255), 2, cv2.LINE_AA)
            _label(out, "CENTER", cx + r + 6, cy + 6, (0, 80, 255))

    _label(out, f"HEATMAP — ME (orange) + OP (blue)  |  tier: {tier.upper()}",
           16, 36, (255, 255, 255))
    _label(out, f"ME visibility {int(my_vis*100)}% ({my_count} pts)  "
                f"|  OP visibility {int(op_vis*100)}% ({op_count} pts)",
           16, 68, (200, 200, 200))

    cv2.imwrite(str(out_dir / "heatmap.png"), out)
    return 1


# ── Baseline (boxes + skeleton) screenshots ──────────────────────────────────
def render_baseline(video_path: str, enriched: dict, out_dir: Path,
                    n: int = 12) -> int:
    """
    Pick n evenly-spaced enriched frames, draw fighter boxes + pose skeletons
    + clinch badge, and save as PNGs. Coords in enriched are normalized [0, 1];
    they get scaled by the compressed video's intrinsic dimensions here.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        try: old.unlink()
        except OSError: pass

    frames = enriched.get("frames", [])
    if not frames:
        return 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Evenly-spaced indices across the enriched frame list.
    if len(frames) <= n:
        picks = list(range(len(frames)))
    else:
        picks = [int(i) for i in np.linspace(0, len(frames) - 1, n)]

    count = 0
    for idx in picks:
        f = frames[idx]
        t = f.get("time_s", 0.0)
        frame = _grab_frame(cap, t, fps)
        if frame is None:
            continue

        for pfx, color, label in (("my", COL_ME, "ME"), ("op", COL_OP, "OP")):
            bbox = f.get(f"{pfx}_bbox")
            if bbox is not None:
                px_bbox = [bbox[0] * W, bbox[1] * H, bbox[2] * W, bbox[3] * H]
                _draw_box(frame, px_bbox, color, label=label)
            kps = f.get(f"{pfx}_kps")
            if kps:
                kps_px = [[k[0] * W, k[1] * H, k[2]] if k else None for k in kps]
                _draw_skeleton(frame, kps_px, color)

        if f.get("clinch"):
            _label(frame, "CLINCH", 16, 70, (0, 50, 220))
        _label(frame, f"frame {idx+1}/{len(frames)}  t={t:.2f}s",
               16, 36, (255, 255, 255))

        out_path = out_dir / f"baseline_{count:03d}_t{t:06.2f}.png"
        cv2.imwrite(str(out_path), frame)
        count += 1

    cap.release()
    return count


# ── Helpers for listing ──────────────────────────────────────────────────────
def list_screenshots(stage_dir: Path) -> list[str]:
    if not stage_dir.exists():
        return []
    return sorted(p.name for p in stage_dir.glob("*.png"))
