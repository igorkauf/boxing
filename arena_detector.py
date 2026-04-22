"""
arena_detector.py — build the arena coordinate system from fighter foot points.

Runs AFTER pose enrichment. Uses SAM2-tracked identities (only the two picked
fighters contribute foot points — no spectators). Produces everything the
metrics need:

    • Per-frame camera-motion warps that map image pixels into a common
      stabilized reference frame (anchored at the middle enriched frame).
    • All foot points collected across the clip, transformed into ref coords.
    • An arena polygon (dilated walked area) + centroid + principal axis.
    • Per-fighter foot trajectory in ref coords (for trail viz + metrics).
    • A confidence tier derived from foot visibility across the clip, so
      downstream metrics can self-weight when the floor isn't reliably
      visible (bottom-up cameras, feet out of frame, etc.).

Output written to sessions_data/<sid>/arena.json.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ── Tunables ──────────────────────────────────────────────────────────────────
ANKLE_IDX       = (15, 16)
SHOULDER_L_IDX, SHOULDER_R_IDX = 5, 6
HIP_L_IDX,      HIP_R_IDX      = 11, 12
MASK_PAD_FRAC   = 0.075   # expand fighter bbox by this fraction when masking

# KLT (Kanade-Lucas-Tomasi) motion-compensation parameters. Replaces the
# previous pairwise ORB matching — long-lived feature tracks keep per-frame
# drift much smaller because each track provides a multi-frame anchor rather
# than a fresh descriptor match.
KLT_MAX_FEATURES = 500    # max tracks to maintain
KLT_MIN_FEATURES = 120    # top up with new corners when we fall below this
KLT_QUALITY      = 0.01   # Shi-Tomasi minimum corner response
KLT_MIN_DISTANCE = 12     # minimum pixel separation between tracks
KLT_WIN_SIZE     = (21, 21)
KLT_PYR_LEVELS   = 3
KLT_FB_MAX_ERR   = 1.5    # forward-backward projection error threshold (px)
# (Keyframe-based BA removed: on real handheld/boxing footage, long-baseline
# LK is not uniformly accurate — features near frame edges track less
# precisely than center features, biasing the RANSAC affine fit toward
# false scale changes. Compounded across ~90 keyframes in a 3-min clip,
# that produced a 4× phantom zoom and a centroid that migrated below the
# frame. Pairwise chain gives per-step error ~0.3 px that sums to ~16 px
# total drift over 2700 frames — well under visual noticeability.)

MIN_MATCHES     = 12
MIN_INLIERS     = 8
# SCALE_LO_HI retained as a historical note: the former partial-affine fit
# had a scale DOF whose bias compounded multiplicatively. The new rigid fit
# (translation + rotation only) has no scale, so this isn't applicable.
SCALE_LO_HI     = (0.85, 1.20)
MAX_TRANSLATION_PER_FRAME = 60   # pixels — rejects LK-failure outliers
MAX_ROTATION_PER_FRAME    = 8.0  # degrees — cameras don't spin that fast
DILATION_FRAC   = 0.35    # dilation radius per foot point, in median-scale units
KP_CONF_MIN     = 0.40    # min keypoint confidence for torso/shoulder measures
BBOX_H_W_MIN    = 0.70    # skip bbox-bottom foot-point when bbox is horizontal
                          # (knockdowns are <0.5; deep crouches rarely <1.0)


# ── Camera motion helpers ─────────────────────────────────────────────────────
class _KLTMotion:
    """
    Pairwise LK feature-track motion estimator.

    Each frame:
      1. Lucas-Kanade tracks existing features forward (prev → cur).
      2. Forward-backward verification prunes unreliable tracks.
      3. Tracks that now sit on a fighter (per cur_mask) are dropped.
      4. If the surviving track count falls below KLT_MIN_FEATURES, top up
         with fresh Shi-Tomasi corners in the background region.
      5. Estimate a 2×3 partial-affine transform from the prev→cur
         correspondences via RANSAC.

    This is the implementation that worked cleanly at v0.1-klt-baseline.
    A keyframe-based BA experiment (v0.6-direct-LK) was reverted because
    long-baseline LK on handheld/boxing footage has non-uniform per-feature
    accuracy, which RANSAC misinterprets as scale change — compounding to
    ~4× phantom zoom on a 3-min clip.

    Per-frame error with 1-frame LK is ~0.3 px; over 2700 pose-frames the
    chain drift is √2700 × 0.3 ≈ 16 px — visually fine.
    """

    def __init__(self):
        self.prev_gray     = None
        self.prev_features = None   # N×1×2 float32, positions in prev_gray
        self.prev_mask     = None

    def step(self, cur_gray, cur_mask):
        """Advance one frame. Returns a 2×3 affine (prev→cur) or None."""
        if self.prev_gray is None:
            # Bootstrap: seed features on the first frame.
            self.prev_gray     = cur_gray
            self.prev_mask     = cur_mask
            self.prev_features = self._detect_features(cur_gray, cur_mask)
            return _identity()

        M = None
        if (self.prev_features is not None
                and len(self.prev_features) >= MIN_MATCHES):
            prev_pts, cur_pts = self._lk_track(cur_gray, cur_mask)
            if (prev_pts is not None
                    and len(prev_pts) >= MIN_MATCHES):
                # Rigid (3-DOF) fit — no scale parameter, so scale can't
                # drift. Translation and rotation still fit with RANSAC +
                # Kabsch refinement.
                M_est = _rigid_ransac(
                    prev_pts.reshape(-1, 2),
                    cur_pts.reshape(-1, 2),
                    threshold=3.0,
                    n_iters=200,
                )
                if M_est is not None:
                    # Sanity: cap per-frame translation and rotation at
                    # physically plausible limits. Anything beyond this is
                    # almost certainly a tracking failure, not real motion.
                    tx, ty = float(M_est[0, 2]), float(M_est[1, 2])
                    rot = math.degrees(math.atan2(M_est[1, 0], M_est[0, 0]))
                    if (abs(tx) < MAX_TRANSLATION_PER_FRAME
                            and abs(ty) < MAX_TRANSLATION_PER_FRAME
                            and abs(rot) < MAX_ROTATION_PER_FRAME):
                        M = M_est
                cur_tracked = cur_pts
            else:
                cur_tracked = None
        else:
            cur_tracked = None

        # Top up features in cur_gray if we're low (or if we lost them all).
        if cur_tracked is None or len(cur_tracked) < KLT_MIN_FEATURES:
            needed = KLT_MAX_FEATURES - (0 if cur_tracked is None else len(cur_tracked))
            extras = self._detect_features(cur_gray, cur_mask, max_corners=needed)
            if cur_tracked is None:
                cur_tracked = extras
            elif extras is not None and len(extras) > 0:
                cur_tracked = np.concatenate([cur_tracked, extras], axis=0)

        self.prev_gray     = cur_gray
        self.prev_mask     = cur_mask
        self.prev_features = cur_tracked
        return M

    def _lk_track(self, cur_gray, cur_mask):
        """Returns (prev_pts_surviving, cur_pts_surviving) as N×1×2 arrays."""
        lk_params = dict(
            winSize=KLT_WIN_SIZE,
            maxLevel=KLT_PYR_LEVELS,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        cur_pts, status_f, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, cur_gray, self.prev_features, None, **lk_params)
        if cur_pts is None:
            return None, None
        back_pts, status_b, _ = cv2.calcOpticalFlowPyrLK(
            cur_gray, self.prev_gray, cur_pts, None, **lk_params)
        if back_pts is None:
            return None, None
        fb_err = np.linalg.norm(
            back_pts.reshape(-1, 2) - self.prev_features.reshape(-1, 2),
            axis=1,
        )
        keep = (
            (status_f.flatten() == 1)
            & (status_b.flatten() == 1)
            & (fb_err < KLT_FB_MAX_ERR)
        )
        if cur_mask is not None:
            cxy = cur_pts.reshape(-1, 2)
            h_, w_ = cur_mask.shape
            xi = np.clip(cxy[:, 0].astype(np.int32), 0, w_ - 1)
            yi = np.clip(cxy[:, 1].astype(np.int32), 0, h_ - 1)
            on_bg = cur_mask[yi, xi] > 0
            keep &= on_bg
        if not keep.any():
            return None, None
        return self.prev_features[keep], cur_pts[keep]

    @staticmethod
    def _detect_features(gray, mask, max_corners=KLT_MAX_FEATURES):
        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=max_corners,
            qualityLevel=KLT_QUALITY,
            minDistance=KLT_MIN_DISTANCE,
            mask=mask,
            blockSize=3,
        )
        return pts if pts is not None else np.zeros((0, 1, 2), dtype=np.float32)


def _compose(A, B):
    A3 = np.vstack([A, [0, 0, 1]])
    B3 = np.vstack([B, [0, 0, 1]])
    return (A3 @ B3)[:2]


def _invert(M):
    M3 = np.vstack([M, [0, 0, 1]])
    return np.linalg.inv(M3)[:2]


def _apply(M, pts):
    if pts is None or len(pts) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    arr = np.asarray(pts, dtype=np.float32)
    ones = np.ones((len(arr), 1), dtype=np.float32)
    return np.concatenate([arr, ones], axis=1) @ M.T


def _identity():
    return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)


# ── Rigid-transform fit (custom — no OpenCV equivalent) ──────────────────────
# Why not cv2.estimateAffinePartial2D? That fits a 4-DOF partial-affine (scale,
# rotation, tx, ty), and the scale DOF has small RANSAC bias every frame that
# compounds multiplicatively over thousands of frames to produce visible
# "phantom zoom" on long clips. Boxing cameras don't actually zoom, so we fit
# a pure rigid transform (3 DOF: tx, ty, rotation) where scale simply doesn't
# exist as an estimated quantity.

def _fit_rigid_2pt(s1: np.ndarray, s2: np.ndarray,
                   d1: np.ndarray, d2: np.ndarray):
    """Analytical rigid transform from exactly 2 source→destination pairs.
    Returns 2×3 or None on degenerate input."""
    s_vec = s2 - s1
    d_vec = d2 - d1
    s_len = float(np.linalg.norm(s_vec))
    d_len = float(np.linalg.norm(d_vec))
    if s_len < 1e-6 or d_len < 1e-6:
        return None
    # Angle between source and destination edge vectors.
    theta = math.atan2(d_vec[1], d_vec[0]) - math.atan2(s_vec[1], s_vec[0])
    c, s = math.cos(theta), math.sin(theta)
    # Rotation about origin, then translation so s1 → d1.
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    t = d1 - R @ s1
    return np.array([[c, -s, t[0]],
                     [s,  c, t[1]]], dtype=np.float64)


def _kabsch_rigid(src: np.ndarray, dst: np.ndarray):
    """Optimal rigid transform by Kabsch algorithm (SVD). Assumes matched
    pairs src[i] ↔ dst[i]. Returns 2×3 or None for degenerate input."""
    if len(src) < 2 or len(dst) != len(src):
        return None
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    s_mean = src.mean(axis=0)
    d_mean = dst.mean(axis=0)
    s_c = src - s_mean
    d_c = dst - d_mean
    H = s_c.T @ d_c
    try:
        U, _, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return None
    # Correct reflection if determinant is negative (flip the last column of V).
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    if d < 0:
        Vt = Vt.copy()
        Vt[-1, :] *= -1
    R = Vt.T @ U.T
    t = d_mean - R @ s_mean
    return np.array([[R[0, 0], R[0, 1], t[0]],
                     [R[1, 0], R[1, 1], t[1]]], dtype=np.float64)


def _rigid_ransac(src: np.ndarray, dst: np.ndarray,
                  threshold: float = 3.0,
                  n_iters: int = 200,
                  min_inliers: int = MIN_INLIERS):
    """
    Robust rigid-transform fit: RANSAC with 2-point samples, then refine
    the best inlier set analytically via Kabsch. 2-point samples are
    cheap and deterministic, so 200 iterations is plenty — at 60-90%
    inlier rate the probability of missing an all-inlier pair is < 10⁻⁴⁰.

    Returns 2×3 affine matrix, or None if no strong consensus.
    """
    n = len(src)
    if n < min_inliers:
        return None
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape:
        return None

    best_M = None
    best_count = 0
    threshold_sq = threshold * threshold
    rng = np.random.default_rng(seed=0xB0D1)   # deterministic for reproducibility

    for _ in range(n_iters):
        idx = rng.choice(n, 2, replace=False)
        M = _fit_rigid_2pt(src[idx[0]], src[idx[1]],
                           dst[idx[0]], dst[idx[1]])
        if M is None:
            continue
        # Forward-project src, count inliers by squared reprojection error.
        projected = src @ M[:, :2].T + M[:, 2]
        err_sq = np.sum((projected - dst) ** 2, axis=1)
        count = int((err_sq < threshold_sq).sum())
        if count > best_count:
            best_count = count
            best_M = M

    if best_M is None or best_count < min_inliers:
        return None

    # Refine on full inlier set (Kabsch).
    projected = src @ best_M[:, :2].T + best_M[:, 2]
    err_sq = np.sum((projected - dst) ** 2, axis=1)
    inliers = err_sq < threshold_sq
    if inliers.sum() < min_inliers:
        return best_M
    refined = _kabsch_rigid(src[inliers], dst[inliers])
    return refined if refined is not None else best_M




# ── Arena geometry ────────────────────────────────────────────────────────────
# New approach: build a Gaussian-blurred density heatmap from foot points,
# threshold it to isolate the meaningfully-walked area, expand by a margin,
# smooth, and extract the outer contour. This tracks the actual walked area
# much better than a uniform disc-dilation, because rare excursions (a single
# chase into a corner) contribute much less density than frequent positions.

POLYGON_KERNEL_FRAC     = 0.18   # Gaussian blur kernel ≈ this × median_scale.
POLYGON_THRESHOLD       = 0.22   # fraction of max density to count as "walked".
POLYGON_EXPAND_FRAC     = 0.0    # no outward expansion — polygon hugs heatmap.
POLYGON_POINT_INCLUDE   = 0.020  # radius of "always-include" disc stamped at
                                 # every foot point, as fraction of scale.
                                 # Guarantees rare excursions (single steps
                                 # into a corner) are inside the polygon even
                                 # if their density is below the threshold.
POLYGON_SIMPLIFY        = 0.004  # Douglas-Peucker epsilon as fraction of perimeter

REF_ANCHOR_FRAC         = 0.95   # where along the clip to anchor the reference
                                 # frame. End-weighted so the final state (which
                                 # we display to validate polygon coverage) has
                                 # zero motion-compensation drift. Trade-off:
                                 # frames early in the clip will have larger
                                 # drift, but those are less important.


def _heatmap_polygon(points_xy: np.ndarray, median_scale: float) -> Optional[list]:
    """
    Convert foot points into a walked-area polygon by way of a Gaussian
    density map. Returns a list of [x, y] pairs in the same coord system
    as the input points (reference frame, possibly unbounded).
    """
    if len(points_xy) < 3 or median_scale <= 0:
        return None

    pad = int(median_scale * 1.5)
    x_min = int(points_xy[:, 0].min()) - pad
    y_min = int(points_xy[:, 1].min()) - pad
    x_max = int(points_xy[:, 0].max()) + pad
    y_max = int(points_xy[:, 1].max()) + pad
    cw, ch = max(1, x_max - x_min), max(1, y_max - y_min)
    if cw * ch > 60_000_000:     # safety — shouldn't happen with valid input
        return None

    # 1. Stamp points then Gaussian-blur for smooth density.
    density = np.zeros((ch, cw), dtype=np.float32)
    for p in points_xy:
        x, y = int(p[0]) - x_min, int(p[1]) - y_min
        if 0 <= x < cw and 0 <= y < ch:
            density[y, x] += 1.0
    ksize = max(25, int(median_scale * POLYGON_KERNEL_FRAC))
    if ksize % 2 == 0:
        ksize += 1
    density = cv2.GaussianBlur(density, (ksize, ksize), 0)

    mx = float(density.max())
    if mx <= 1e-6:
        return None

    # 2. Threshold at a fraction of max density.
    mask = ((density >= mx * POLYGON_THRESHOLD).astype(np.uint8)) * 255

    # 3. Force every individual foot point to be inside the mask by stamping
    #    a small disc at each one. This handles the "single-step excursion"
    #    case: a foot point visited briefly contributes too little density
    #    to survive the threshold, but the fighter WAS there so it must be
    #    inside the walked-area polygon. Disc radius is ~10% of scale, which
    #    is small enough not to extend the polygon toward spectators.
    point_r = max(3, int(median_scale * POLYGON_POINT_INCLUDE))
    for p in points_xy:
        x, y = int(p[0]) - x_min, int(p[1]) - y_min
        if 0 <= x < cw and 0 <= y < ch:
            cv2.circle(mask, (x, y), point_r, 255, -1)

    # 4. Expand by a small margin (in scale units) so the polygon has a
    #    little breathing room around where fighters actually stepped.
    expand_px = max(3, int(median_scale * POLYGON_EXPAND_FRAC))
    k_expand = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (expand_px * 2 + 1, expand_px * 2 + 1))
    mask = cv2.dilate(mask, k_expand)

    # 5. Smooth the edges so the contour isn't jagged.
    k_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_smooth)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_smooth)

    # 5. Largest external contour → convex hull (kills internal reversed
    #    angles — polygon has smooth external boundary only) → simplify with
    #    Douglas-Peucker.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    hull    = cv2.convexHull(largest)
    eps     = cv2.arcLength(hull, True) * POLYGON_SIMPLIFY
    simplified = cv2.approxPolyDP(hull, eps, True)
    return [[float(pt[0][0] + x_min), float(pt[0][1] + y_min)]
            for pt in simplified]


def _principal_axis(points_xy: np.ndarray):
    if len(points_xy) < 3:
        return [1.0, 0.0], 0.0, 0.0
    cov = np.cov(points_xy.T)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    major = evecs[:, order[0]]
    return ([float(major[0]), float(major[1])],
            float(math.sqrt(max(evals[order[0]], 0.0))),
            float(math.sqrt(max(evals[order[1]], 0.0))))


# ── Foot point extraction (per enriched frame) ────────────────────────────────
def _foot_points_for_fighter(kps_n, bbox_n, frame_w, frame_h):
    """
    Extract foot points (image-pixel coords) for one fighter from a normalized
    enriched frame. Prefer high-confidence ankles; include bbox-bottom-center
    only if the bbox is upright (h/w >= BBOX_H_W_MIN) and not pinned to the
    frame edge. Horizontal bboxes indicate a fighter is down or the detector
    got a weird read — bbox bottom doesn't correspond to feet in that case.
    Returns list of [x, y].
    """
    out = []
    if kps_n:
        for ai in ANKLE_IDX:
            if ai < len(kps_n) and kps_n[ai] and kps_n[ai][2] > 0.3:
                out.append([float(kps_n[ai][0]) * frame_w,
                            float(kps_n[ai][1]) * frame_h])
    if bbox_n:
        bx1 = float(bbox_n[0]) * frame_w
        by1 = float(bbox_n[1]) * frame_h
        bx2 = float(bbox_n[2]) * frame_w
        by2 = float(bbox_n[3]) * frame_h
        bw  = max(1.0, bx2 - bx1)
        bh  = max(1.0, by2 - by1)
        upright = (bh / bw) >= BBOX_H_W_MIN
        if upright and by2 < frame_h - 3:
            cx = 0.5 * (bx1 + bx2)
            out.append([cx, by2])
    return out


def _per_frame_scale(kps_n, bbox_n, frame_w, frame_h) -> float:
    """
    Return a size reference for this fighter on this frame, in image pixels.

    Neither signal alone is ideal: torso-diagonal is stance-invariant but
    rotation-sensitive (foreshortens when the fighter turns sideways); bbox
    height is rotation-stable but stance-sensitive (shrinks when crouching).
    We take max(torso_diagonal × 2.2, bbox_height) per frame — the "×2.2"
    scales torso so it's comparable to full body height, and max() picks
    whichever reading isn't currently foreshortened.

    Priority:
      • Composite above when keypoints + upright bbox are available.
      • Shoulder-width × 2.6 as a rotation-robust-ish fallback when hips are
        missing but shoulders are confident.
      • Upright bbox height alone when keypoints are unreliable.
      • 0.0 if nothing usable.
    """
    bh_val = 0.0
    if bbox_n:
        bw = (float(bbox_n[2]) - float(bbox_n[0])) * frame_w
        bh = (float(bbox_n[3]) - float(bbox_n[1])) * frame_h
        if bw > 1.0 and (bh / max(bw, 1.0)) >= BBOX_H_W_MIN:
            bh_val = float(bh)

    torso_val = 0.0
    shoulder_val = 0.0
    if kps_n and len(kps_n) > max(HIP_L_IDX, HIP_R_IDX):
        sl = kps_n[SHOULDER_L_IDX]
        sr = kps_n[SHOULDER_R_IDX]
        hl = kps_n[HIP_L_IDX]
        hr = kps_n[HIP_R_IDX]
        def _c(p):
            return p is not None and len(p) >= 3 and p[2] >= KP_CONF_MIN
        if _c(sl) and _c(sr) and _c(hl) and _c(hr):
            smx = 0.5 * (sl[0] + sr[0]) * frame_w
            smy = 0.5 * (sl[1] + sr[1]) * frame_h
            hmx = 0.5 * (hl[0] + hr[0]) * frame_w
            hmy = 0.5 * (hl[1] + hr[1]) * frame_h
            torso_val = float(math.hypot(smx - hmx, smy - hmy))
        elif _c(sl) and _c(sr):
            shoulder_val = float(math.hypot((sl[0] - sr[0]) * frame_w,
                                            (sl[1] - sr[1]) * frame_h))

    # Composite: pick whichever signal is currently larger (not foreshortened).
    # Torso diagonal is ~0.45 × body height on average, so multiply by 2.2
    # to bring it into comparable units before taking max.
    candidates = []
    if torso_val > 0:
        candidates.append(torso_val * 2.2)
    if shoulder_val > 0:
        candidates.append(shoulder_val * 2.6)
    if bh_val > 0:
        candidates.append(bh_val)
    return max(candidates) if candidates else 0.0


def _mask_for_frame(frame_bgr, bboxes_norm, pad_frac):
    """Build background mask: white everywhere, black over expanded bboxes."""
    h, w = frame_bgr.shape[:2]
    mask = np.full((h, w), 255, dtype=np.uint8)
    for bb in bboxes_norm:
        if not bb:
            continue
        x1 = int(bb[0] * w); y1 = int(bb[1] * h)
        x2 = int(bb[2] * w); y2 = int(bb[3] * h)
        bw = max(1, x2 - x1); bh = max(1, y2 - y1)
        ex1 = max(0, int(x1 - bw * pad_frac))
        ey1 = max(0, int(y1 - bh * pad_frac))
        ex2 = min(w, int(x2 + bw * pad_frac))
        ey2 = min(h, int(y2 + bh * pad_frac))
        mask[ey1:ey2, ex1:ex2] = 0
    return mask


# ── Confidence tier ───────────────────────────────────────────────────────────
def _confidence_tier(visibility_score: float) -> str:
    if visibility_score >= 0.75:
        return "high"
    if visibility_score >= 0.40:
        return "medium"
    if visibility_score >= 0.15:
        return "low"
    return "none"


# ── Main entry point ─────────────────────────────────────────────────────────
def detect_arena(video_path: str, enriched: dict) -> dict:
    """
    Build the arena from an enriched SAM2+pose payload. Returns a plain dict
    ready to JSON-serialize. Caller persists it wherever it wants.
    """
    frames = enriched.get("frames", [])
    frame_w = int(enriched.get("frame_w", 0))
    frame_h = int(enriched.get("frame_h", 0))
    if not frames or frame_w <= 0 or frame_h <= 0:
        return {"ok": False, "error": "empty or malformed enriched payload"}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"ok": False, "error": f"cannot open video {video_path}"}

    n_frames = len(frames)
    # End-weighted reference so late frames (where the polygon's coverage is
    # validated visually) have zero motion-comp drift. See REF_ANCHOR_FRAC.
    ref_index = min(n_frames - 1, int(n_frames * REF_ANCHOR_FRAC))

    # Pairwise KLT chain. Per-step LK error ~0.3 px; over n frames the chain
    # drift scales as √n (≈16 px over 2700 pose-frames for a 3-min clip).
    # See _KLTMotion for why keyframe-based BA was reverted.
    warps_to_first: list[np.ndarray] = [_identity()]
    klt = _KLTMotion()
    motion_failures = 0

    my_traj: list = [None] * n_frames
    op_traj: list = [None] * n_frames
    all_points_img: list[list] = [[] for _ in range(n_frames)]
    my_scales: list[float] = []
    op_scales: list[float] = []
    feet_visible_count = 0
    my_visible_count = 0
    op_visible_count = 0

    # We need to visit frames at their raw_fi positions in order. Sort by
    # raw_fi (they should already be sorted but be defensive).
    idx_order = sorted(range(n_frames), key=lambda i: frames[i].get("raw_fi", i))

    for rank, fi in enumerate(idx_order):
        f = frames[fi]
        raw_fi = int(f.get("raw_fi", 0))
        cap.set(cv2.CAP_PROP_POS_FRAMES, raw_fi)
        ok, frame = cap.read()
        if not ok:
            # Identity for this slot; skip motion/feet.
            if rank > 0:
                warps_to_first.append(warps_to_first[-1])
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = _mask_for_frame(frame, [f.get("my_bbox"), f.get("op_bbox")],
                               MASK_PAD_FRAC)

        # Pairwise KLT — each step returns T_prev→cur (affine) or None.
        if rank == 0:
            klt.step(gray, mask)    # seeds features; first frame → identity
        else:
            M_prev_to_cur = klt.step(gray, mask)
            if M_prev_to_cur is None:
                motion_failures += 1
                M_prev_to_cur = _identity()
            cur_to_prev = _invert(M_prev_to_cur)
            warps_to_first.append(
                _compose(warps_to_first[-1], cur_to_prev)
            )

        # Foot points (image coords).
        my_pts = _foot_points_for_fighter(f.get("my_kps"), f.get("my_bbox"),
                                          frame_w, frame_h)
        op_pts = _foot_points_for_fighter(f.get("op_kps"), f.get("op_bbox"),
                                          frame_w, frame_h)
        if my_pts:
            my_visible_count += 1
        if op_pts:
            op_visible_count += 1
        if my_pts or op_pts:
            feet_visible_count += 1

        # Per-fighter "representative" foot: average of their foot points
        # (single xy per frame, for trajectories & per-frame metrics).
        def _mean(pts):
            if not pts:
                return None
            return [float(np.mean([p[0] for p in pts])),
                    float(np.mean([p[1] for p in pts]))]

        my_traj[fi] = _mean(my_pts)
        op_traj[fi] = _mean(op_pts)
        all_points_img[fi] = my_pts + op_pts

        # Stance-invariant scale anchor per fighter.
        my_scale = _per_frame_scale(f.get("my_kps"), f.get("my_bbox"),
                                    frame_w, frame_h)
        op_scale = _per_frame_scale(f.get("op_kps"), f.get("op_bbox"),
                                    frame_w, frame_h)
        if my_scale > 0:
            my_scales.append(my_scale)
        if op_scale > 0:
            op_scales.append(op_scale)

    cap.release()

    # Re-anchor warps to the chosen reference frame (see REF_ANCHOR_FRAC).
    if len(warps_to_first) < n_frames:
        # Pad tail with the last known warp (shouldn't happen, safety only).
        last = warps_to_first[-1] if warps_to_first else _identity()
        while len(warps_to_first) < n_frames:
            warps_to_first.append(last)

    first_to_ref = _invert(warps_to_first[ref_index])
    warps_to_ref = [_compose(first_to_ref, W) for W in warps_to_first]

    # Transform foot points and trajectories into ref coords.
    all_points_ref = []
    my_traj_ref: list = [None] * n_frames
    op_traj_ref: list = [None] * n_frames
    for fi in range(n_frames):
        W = warps_to_ref[fi]
        if all_points_img[fi]:
            transformed = _apply(W, all_points_img[fi])
            all_points_ref.extend([[float(p[0]), float(p[1])] for p in transformed])
        if my_traj[fi] is not None:
            t = _apply(W, [my_traj[fi]])
            my_traj_ref[fi] = [float(t[0][0]), float(t[0][1])]
        if op_traj[fi] is not None:
            t = _apply(W, [op_traj[fi]])
            op_traj_ref[fi] = [float(t[0][0]), float(t[0][1])]

    visibility_score   = feet_visible_count / max(1, n_frames)
    my_visibility      = my_visible_count   / max(1, n_frames)
    op_visibility      = op_visible_count   / max(1, n_frames)
    tier               = _confidence_tier(visibility_score)

    # Stance-invariant scale anchor. Median is robust to outliers.
    my_median_scale = float(np.median(my_scales)) if my_scales else 0.0
    op_median_scale = float(np.median(op_scales)) if op_scales else 0.0
    combined_scales = my_scales + op_scales
    median_scale    = float(np.median(combined_scales)) if combined_scales else 0.0

    # Arena polygon + centroid + axis.
    pts_np = np.array(all_points_ref, dtype=np.float32) if all_points_ref else np.zeros((0, 2))
    arena_polygon = _heatmap_polygon(pts_np, median_scale) if len(pts_np) >= 3 else None
    centroid = ([float(pts_np[:, 0].mean()), float(pts_np[:, 1].mean())]
                if len(pts_np) >= 3 else None)
    axis_vec, axis_len, minor_len = (_principal_axis(pts_np)
                                     if len(pts_np) >= 3 else ([1.0, 0.0], 0.0, 0.0))

    # Arena span expressed in torso-scale units (the stance-stable body
    # yardstick). One "scale" ≈ shoulder-to-hip diagonal, ~45 cm on adults.
    span_in_scales = (2.0 * axis_len / median_scale) if median_scale > 0 else 0.0

    return {
        "ok":                          True,
        "frame_w":                     frame_w,
        "frame_h":                     frame_h,
        "ref_index":                   ref_index,
        "centroid":                    centroid,
        "arena_polygon":               arena_polygon,
        "axis_vec":                    axis_vec,
        "axis_len":                    axis_len,
        "minor_len":                   minor_len,
        "span_in_scales":              span_in_scales,
        "median_scale_px":             median_scale,
        "my_median_scale_px":          my_median_scale,
        "op_median_scale_px":          op_median_scale,
        # Retained for backward compatibility; downstream should migrate to
        # median_scale_px (stance-invariant) when it matters.
        "median_fighter_height_px":    median_scale,
        "warps_to_ref":                [W.tolist() for W in warps_to_ref],
        "my_traj_ref":                 my_traj_ref,
        "op_traj_ref":                 op_traj_ref,
        "n_foot_points":               int(len(pts_np)),
        "motion_failures":             motion_failures,
        "visibility_score":            visibility_score,
        "my_visibility_score":         my_visibility,
        "op_visibility_score":         op_visibility,
        "confidence_tier":             tier,
        "n_enriched_frames":           n_frames,
    }


def detect_and_save(video_path: str, enriched: dict, out_path: Path) -> dict:
    result = detect_arena(video_path, enriched)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, separators=(",", ":")))
    return result


def refresh_arena_polygon(session_dir: Path) -> Optional[dict]:
    """
    Recompute the arena polygon + re-anchor the reference frame to
    REF_ANCHOR_FRAC (near end of clip) using already-cached data. No SAM2,
    no pose enrichment, no motion-comp re-run. Uses:
      • sam2_enriched.json        — raw foot keypoints + bboxes per frame
      • arena.json → warps_to_ref — stored per-frame motion warps

    Re-anchoring is just a matrix re-composition of the existing warps
    (old_ref → new_ref is a single 2×3 transform applied to every warp and
    every stored point). The polygon is then rebuilt from scratch so the
    latest threshold / kernel / include settings take effect.

    Returns the updated arena dict (also written to arena.json), or None
    if the required caches aren't present.
    """
    enriched_path = session_dir / "sam2_enriched.json"
    arena_path    = session_dir / "arena.json"
    if not enriched_path.exists() or not arena_path.exists():
        return None
    try:
        enriched = json.loads(enriched_path.read_text())
        arena    = json.loads(arena_path.read_text())
    except json.JSONDecodeError:
        return None
    if not arena.get("ok"):
        return None

    frames     = enriched.get("frames", [])
    old_warps  = arena.get("warps_to_ref", [])
    fw         = int(arena.get("frame_w") or 0)
    fh         = int(arena.get("frame_h") or 0)
    if not frames or not old_warps or fw <= 0 or fh <= 0:
        return None

    n_frames = min(len(frames), len(old_warps))

    # ── Re-anchor reference frame to REF_ANCHOR_FRAC of the clip. ──────────
    new_ref = min(n_frames - 1, int(n_frames * REF_ANCHOR_FRAC))
    reanchor = _invert(np.array(old_warps[new_ref], dtype=np.float64))
    warps_to_ref = [
        _compose(reanchor, np.array(W, dtype=np.float64))
        for W in old_warps[:n_frames]
    ]

    # ── Recompute all foot points in the NEW ref coords. ───────────────────
    all_points_ref: list[list[float]] = []
    my_traj_ref:    list              = [None] * n_frames
    op_traj_ref:    list              = [None] * n_frames

    def _mean(pts):
        if not pts:
            return None
        return [float(np.mean([p[0] for p in pts])),
                float(np.mean([p[1] for p in pts]))]

    for fi in range(n_frames):
        f = frames[fi]
        W = warps_to_ref[fi]
        my_pts = _foot_points_for_fighter(f.get("my_kps"),
                                          f.get("my_bbox"), fw, fh)
        op_pts = _foot_points_for_fighter(f.get("op_kps"),
                                          f.get("op_bbox"), fw, fh)
        if my_pts:
            t = _apply(W, [_mean(my_pts)])
            my_traj_ref[fi] = [float(t[0][0]), float(t[0][1])]
        if op_pts:
            t = _apply(W, [_mean(op_pts)])
            op_traj_ref[fi] = [float(t[0][0]), float(t[0][1])]
        combined = my_pts + op_pts
        if combined:
            transformed = _apply(W, combined)
            for p in transformed:
                all_points_ref.append([float(p[0]), float(p[1])])

    if len(all_points_ref) < 3:
        return None

    pts_np = np.array(all_points_ref, dtype=np.float32)
    median_scale = float(arena.get("median_scale_px")
                         or arena.get("median_fighter_height_px") or 100.0)
    new_polygon = _heatmap_polygon(pts_np, median_scale)
    if new_polygon is None:
        return None

    centroid = [float(pts_np[:, 0].mean()), float(pts_np[:, 1].mean())]
    axis_vec, axis_len, minor_len = _principal_axis(pts_np)

    # Update everything that lives in reference coords.
    arena["ref_index"]       = new_ref
    arena["warps_to_ref"]    = [W.tolist() for W in warps_to_ref]
    arena["arena_polygon"]   = new_polygon
    arena["centroid"]        = centroid
    arena["my_traj_ref"]     = my_traj_ref
    arena["op_traj_ref"]     = op_traj_ref
    arena["axis_vec"]        = axis_vec
    arena["axis_len"]        = axis_len
    arena["minor_len"]       = minor_len
    if median_scale > 0:
        arena["span_in_scales"] = 2.0 * axis_len / median_scale
    arena["n_foot_points"]   = int(len(pts_np))
    arena_path.write_text(json.dumps(arena, separators=(",", ":")))
    return arena
