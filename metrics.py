"""
metrics.py — compute the five arena-aware boxing metrics.

Inputs:
    sam2_enriched.json  — per-frame pose + bboxes
    arena.json          — centroid, per-fighter trajectory, confidence tier

Metrics:
    Ring Control        (arena-dependent)   : % of time ME is closer to centroid than OP
    Effective Aggression (mixed)            : advance toward opponent × punch activity
    Work Rate           (mixed)             : movement (arena) + punch rate (body-relative)
    Defense             (mixed)             : range management (arena) + head-mobility (body)
    Guard               (body-relative)     : hands-up posture

Confidence weighting:
    Each metric's arena-dependent components are weighted by the arena
    confidence tier. Body-relative components always contribute at full weight.
    A metric's final 0-100 score is the weighted average of its components.

Output written to sessions_data/<sid>/arena_metrics.json and also stored on
meta["arena_metrics"] for the Lab UI to display.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


# ── Tunables ──────────────────────────────────────────────────────────────────
KP_CONF             = 0.35
KP_WRIST_L          = 9
KP_WRIST_R          = 10
KP_SHOULDER_L       = 5
KP_SHOULDER_R       = 6
KP_HIP_L            = 11
KP_HIP_R            = 12
KP_NOSE             = 0
KP_EYE_L            = 1
KP_EYE_R            = 2
KP_EAR_L            = 3
KP_EAR_R            = 4
KP_ELBOW_L          = 7
KP_ELBOW_R          = 8

# ── Landed-hit gating (Phase 1) ──────────────────────────────────────────────
# Target zones follow boxing convention: head = chin-to-crown square (helmet-
# sized), stomach = belt-to-nipples square (~torso-width). Both derived from
# pose keypoints in normalized [0,1] coords. When the required keypoints
# aren't visible at the peak frame, the punch verdict is UNKNOWN (not MISSED)
# because turns/clinches genuinely occlude the target.
LANDED_HIT_RED_FRAMES = 5      # how many frames a zone flashes red on a hit
HIT_WRIST_PAD_NORM    = 0.015  # dilation of zone test box (glove radius,
                               # normalized). A typical boxing glove is ~12cm
                               # — on a vertical-video frame that's 2-3% of
                               # frame width. Combined with glove-tip
                               # extrapolation below, this covers the glove
                               # reaching the target even when the wrist kp
                               # hasn't quite entered the zone yet.
HIT_GLOVE_EXT_NORM    = 0.35   # fraction of the forearm vector (elbow→wrist)
                               # to extrapolate beyond the wrist to estimate
                               # the glove tip. Forearm ≈ 25-30cm, glove tip
                               # is ~8-10cm beyond wrist → 0.30-0.40 of
                               # forearm length.
HEAD_TOP_PAD_EARS     = 0.4    # extend top of head above ears by 0.4 × ear-
                               # spacing, rough helmet crown allowance
HEAD_CHIN_FROM_NOSE   = 1.0    # chin ≈ nose_y + 1.0 × (nose_y − eye_y)
STOMACH_TOP_FRAC      = 0.25   # nipples ≈ shoulder_y + 0.25 × (hip_y − sh_y)

# 1€ filter defaults (see OneEuroFilter). Tuned for pose at ~15 fps with
# heavy enough β to let real punches pass through — wrist keypoints move
# fast during a punch, and the adaptive cutoff must open meaningfully at
# those velocities or the signal gets smothered and detection fires zero.
#   min_cutoff=1.0 Hz → moderate smoothing of resting jitter
#   beta=0.40         → adaptive cutoff rises strongly with wrist velocity
#   d_cutoff=1.0 Hz   → low-pass on the velocity estimate itself
ONE_EURO_MIN_CUTOFF = 1.0
ONE_EURO_BETA       = 0.40
ONE_EURO_D_CUTOFF   = 1.0

# Motion-blur gate: YOLO confidence drops in motion-blurred frames. If a
# fighter has fewer than this many confident keypoints (of 17) in a frame,
# we don't credit them with a punch or a guard signal that frame. Tuned
# conservatively — close-up boxing footage often only has upper-body kps
# clearly visible so a threshold of 8+ was too strict and rejected normal
# frames. A minimal "head + one shoulder + one arm" set clears this gate.
MIN_CONFIDENT_KPS   = 5

PUNCH_EXT_MIN       = 0.55    # wrist extension / torso (normalized). Used for
                              # per-frame "arm active" flag feeding Aggression.
PUNCH_ACCEL_MIN     = 0.075   # body-relative wrist velocity per pose-frame.
                              # Used for per-frame activity flag (Aggression).

# Peak-based total-count detector (used for `my_punches` / `op_punches`).
# Replaces the earlier "sustained run of True flags" approach, which couldn't
# tell the difference between a real punch and a slow block/guard adjustment
# that happened to meet the extension threshold. The score combines:
#   extension × wrist velocity × direction-toward-opponent
# and counts local maxima of that score above PUNCH_SCORE_MIN with at least
# PUNCH_PEAK_SEP_S between peaks. Each real punch produces one peak; slow
# defensive motion produces low-amplitude signal that fails the threshold,
# and motion away from the opponent gets suppressed by the direction factor.
PUNCH_SCORE_MIN     = 0.060   # peak-score threshold. Re-calibrated under the
                              # v3 punch_score formula (bbox-zoom-corrected
                              # velocity + dir_floor=0.60). Minimizes total
                              # absolute error vs ground-truth counts across
                              # 7 clips with a 2.7× range of camera zoom
                              # (fighter-height 0.26 → 0.70 of frame).
PUNCH_PEAK_SEP_S    = 0.18    # min time between detected peaks (fast 1-2
                              # combos sit around 0.2 s apart)
RANGE_TRIGGER_SCALE = 2.6     # "in range" = opponent-distance <= this × scale-px
ADVANCE_EPS_SCALE   = 0.005   # min per-frame closing speed (in scales) to count

# Punch-score v3 knobs (calibrated to cross-clip ground-truth).
PUNCH_DIR_FLOOR      = 0.60   # floor for the direction_factor — hooks
                              # (perpendicular to me→op axis) used to get
                              # halved to 0.5; floor lets close-range curved
                              # punches score on par with straights.
PUNCH_BBOX_REF_H     = 0.55   # reference fighter-height (normalized).
                              # Velocities on frames where the fighter is
                              # smaller than this get scaled up so the score
                              # distribution stays stable across camera-zoom
                              # levels. Capped at 2× to avoid runaway.
                              # Fighters LARGER than the reference get no
                              # correction (their velocities are already
                              # well-resolved by pixel count).

# ── State classifier (Phase 2) ───────────────────────────────────────────────
# Per-frame fighter-interaction state, used to blend per-state aggression
# signals. The classifier is cheap — all inputs are already computed. Each
# state picks a different formula because signal quality differs sharply:
#   OPEN      — far apart, clear visibility; punch_score + closing are gold
#   CLOSE     — infighting but still visible; punch_score noisy (hooks,
#               short punches), so wrist_activity carries more weight
#   CLINCH    — bodies tangled, 2D punch geometry unreliable; fall back to
#               wrist_activity + body push
#   OCCLUSION — one or both fighters not confidently visible; neither gets
#               new credit, but the score decays toward 50/50 gradually
STATE_OPEN              = "open"
STATE_CLOSE             = "close"
STATE_CLINCH            = "clinch"
STATE_OCCLUSION         = "occlusion"
STATE_CLINCH_IOU        = 0.40   # bbox overlap above this ⇒ CLINCH
STATE_CLOSE_DIST_SCALES = 0.50   # inter-fighter distance (in median-scale
                                 # units = fighter-heights) below this ⇒
                                 # CLOSE. Calibrated against observed
                                 # distributions across our clips:
                                 #   <0.5 = genuine infighting (can't
                                 #          extend, hooks/uppercuts only)
                                 #   0.5-1.2 = jab/normal range (can reach
                                 #          with committed extension)
                                 #   >1.2 = out of range
                                 # Previous 1.5 classified normal range as
                                 # CLOSE, which downweighted punch_score
                                 # and undercounted punches (clip 75ff0142:
                                 # 7/10 detected vs 30/42 ground-truth).
STATE_OCCL_KP_FRAC      = 0.40   # per-fighter keypoint-visibility fraction
                                 # below which we call it OCCLUSION
STATE_DECAY_TAU_S       = 2.5    # exponential decay time constant during
                                 # OCCLUSION toward 50/50 — gentle, not snap.

# Aggression blend weights per state.
AGG_OPEN_W    = {"punch": 0.60, "close_retreat": 0.30, "op_retreat": 0.10}
AGG_CLOSE_W   = {"wrist": 0.50, "punch": 0.30, "body_push": 0.20}
AGG_CLINCH_W  = {"wrist": 0.60, "head_move": 0.20, "bbox_push": 0.20}

# ── Defense (4-pillar, coach-structured) ────────────────────────────────────
# Graded only during frames where the fighter is being ATTACKED (opponent's
# punch peak neighborhood). Four pillars, weighted:
#   distance  (40%) — was the opponent out of effective range?
#   head_move (25%) — did the head move in response?
#   guard     (25%) — were the hands up near the face?
#   counter   (10%) — did the fighter actively engage (parry/counter)?
# When a fighter isn't being attacked in a given second, their defense
# quality defaults to a neutral 50 — "no data, no judgment." The per-second
# share between ME and OP is computed from their qualities.
DEF_W = {"distance": 0.40, "head_move": 0.25, "guard": 0.25, "counter": 0.10}
DEF_ATTACK_WIN_PRE_F  = 2    # frames before opponent-peak to include
DEF_ATTACK_WIN_POST_F = 6    # frames after peak (defensive response phase)
# Distance thresholds (in median-scale units): punch reach ~1.3 scales,
# safely out of range ~2.2. Linear ramp between these.
DEF_DIST_NEAR = 1.3
DEF_DIST_FAR  = 2.2
# Normalization constants — same philosophy as the Phase 2 aggression
# signals: scale each raw input to ~[0, 1] typical range.
DEF_HEAD_NORM    = 0.02     # head-relative motion per frame (body coords)
DEF_COUNTER_NORM = 0.04     # wrist_activity magnitude

# Tier → arena-component weight (body-relative components always = 1.0).
TIER_WEIGHT = {
    "high":   1.00,
    "medium": 0.70,
    "low":    0.40,
    "none":   0.00,
}


# ── 1€ filter + keypoint smoothing ───────────────────────────────────────────
class OneEuroFilter:
    """
    1€ filter (Casiez, Roussel, Vogel, 2012). Adaptive low-pass for noisy
    real-time signals. Cutoff frequency rises with signal velocity so fast
    movements pass through unsmoothed while stationary jitter is killed.

    Mathematically:
        dx[t]  = (x[t] − x[t−1]) / dt
        ẋ[t]  = low_pass(dx, fc=d_cutoff)
        fc[t]  = min_cutoff + beta × |ẋ[t]|
        y[t]   = low_pass(x,  fc=fc[t])
    where low_pass(x, fc) = α·x + (1−α)·y_prev, α = 1/(1 + fc/(2π·dt)).
    """

    __slots__ = ("min_cutoff", "beta", "d_cutoff",
                 "x_prev", "dx_prev", "t_prev")

    def __init__(self,
                 min_cutoff: float = ONE_EURO_MIN_CUTOFF,
                 beta:        float = ONE_EURO_BETA,
                 d_cutoff:    float = ONE_EURO_D_CUTOFF):
        self.min_cutoff = min_cutoff
        self.beta       = beta
        self.d_cutoff   = d_cutoff
        self.x_prev:  float | None = None
        self.dx_prev: float        = 0.0
        self.t_prev:  float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        """
        Canonical 1€ low-pass coefficient (Casiez, Roussel, Vogel 2012):
            τ = 1 / (2π · cutoff)
            α = 1 / (1 + τ / dt)
        α = 1 → no smoothing, α = 0 → infinite smoothing. Higher cutoff
        means higher α means less smoothing.
        """
        tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-6))
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def __call__(self, t: float, x: float) -> float:
        if self.x_prev is None:
            self.x_prev  = x
            self.t_prev  = t
            return x
        dt = max(t - self.t_prev, 1e-6)
        # 1. Smooth the velocity.
        dx   = (x - self.x_prev) / dt
        a_d  = self._alpha(self.d_cutoff, dt)
        dx_s = a_d * dx + (1.0 - a_d) * self.dx_prev
        # 2. Adaptive cutoff rises with velocity.
        cutoff = self.min_cutoff + self.beta * abs(dx_s)
        a      = self._alpha(cutoff, dt)
        x_s    = a * x + (1.0 - a) * self.x_prev
        # 3. Commit.
        self.x_prev  = x_s
        self.dx_prev = dx_s
        self.t_prev  = t
        return x_s


def _smooth_keypoints_inplace(frames: list[dict]) -> None:
    """
    Apply 1€ smoothing to every fighter's keypoint stream in place. Each
    keypoint gets two filters (x, y); if its confidence drops below KP_CONF
    in some frame, we hold the filter state (don't update) and emit the last
    smoothed position for that keypoint — the filter re-engages gracefully
    when the keypoint returns with high confidence.
    """
    filters: dict[tuple[str, int, str], OneEuroFilter] = {}
    for f in frames:
        t = float(f.get("time_s", 0.0))
        for fighter in ("my", "op"):
            kps = f.get(f"{fighter}_kps")
            if not kps:
                continue
            new_kps = []
            for ki, kp in enumerate(kps):
                if kp is None or len(kp) < 3:
                    new_kps.append(kp)
                    continue
                x, y, c = float(kp[0]), float(kp[1]), float(kp[2])
                kx = filters.setdefault((fighter, ki, "x"), OneEuroFilter())
                ky = filters.setdefault((fighter, ki, "y"), OneEuroFilter())
                if c >= KP_CONF:
                    x_s = kx(t, x)
                    y_s = ky(t, y)
                    new_kps.append([x_s, y_s, c])
                elif kx.x_prev is not None and ky.x_prev is not None:
                    # Hold the last smoothed position; don't advance filter.
                    new_kps.append([kx.x_prev, ky.x_prev, c])
                else:
                    new_kps.append([x, y, c])
            f[f"{fighter}_kps"] = new_kps


def _confident_kp_count(kps) -> int:
    if not kps:
        return 0
    return sum(1 for k in kps if k and len(k) >= 3 and k[2] >= KP_CONF)


# ── Helpers ──────────────────────────────────────────────────────────────────
def _kp(kps, i):
    if not kps or i >= len(kps) or not kps[i]:
        return None
    if kps[i][2] < KP_CONF:
        return None
    return (float(kps[i][0]), float(kps[i][1]))


def _midpoint(a, b):
    if a is None or b is None:
        return None
    return (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))


def _dist(a, b):
    if a is None or b is None:
        return None
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _shoulder_center(kps):
    return _midpoint(_kp(kps, KP_SHOULDER_L), _kp(kps, KP_SHOULDER_R))


def _torso_height(kps):
    sc = _shoulder_center(kps)
    hc = _midpoint(_kp(kps, KP_HIP_L), _kp(kps, KP_HIP_R))
    if sc is None or hc is None:
        return None
    return _dist(sc, hc)


def _wrist_positions_rel(kps):
    """Body-relative wrist positions (subtract shoulder center)."""
    sc = _shoulder_center(kps)
    if sc is None:
        return (None, None)
    out = []
    for wi in (KP_WRIST_L, KP_WRIST_R):
        w = _kp(kps, wi)
        if w is None:
            out.append(None)
        else:
            out.append((w[0] - sc[0], w[1] - sc[1]))
    return tuple(out)


def _is_punching(kps, prev_kps):
    """True if at least one wrist is both extended AND accelerating."""
    sc = _shoulder_center(kps)
    th = _torso_height(kps)
    if sc is None or th is None or th <= 0:
        return False
    ext_ok = False
    for wi in (KP_WRIST_L, KP_WRIST_R):
        w = _kp(kps, wi)
        if w is None:
            continue
        d = _dist(w, sc)
        if d is not None and (d / th) >= PUNCH_EXT_MIN:
            ext_ok = True
            break
    if not ext_ok:
        return False
    cur = _wrist_positions_rel(kps)
    prv = _wrist_positions_rel(prev_kps)
    for wc, wp in zip(cur, prv):
        if wc is None or wp is None:
            continue
        if math.hypot(wc[0] - wp[0], wc[1] - wp[1]) >= PUNCH_ACCEL_MIN:
            return True
    return False


def _guard_score_frame(kps):
    """
    0–1 guard score: wrists above shoulder level = 1, below = 0, blended.
    Returns None if insufficient keypoints.
    """
    sc = _shoulder_center(kps)
    if sc is None:
        return None
    th = _torso_height(kps)
    if th is None or th <= 0:
        return None
    vals = []
    for wi in (KP_WRIST_L, KP_WRIST_R):
        w = _kp(kps, wi)
        if w is None:
            continue
        # Wrist y ABOVE shoulder y (smaller y) → score 1.
        # Wrist y below shoulder y by torso-height → score 0.
        dy = (w[1] - sc[1]) / th   # positive = wrist below shoulder
        score = max(0.0, min(1.0, 1.0 - (dy + 0.1) / 1.0))
        vals.append(score)
    if not vals:
        return None
    return float(np.mean(vals))


def _head_pos(kps):
    """Use nose if available, else midpoint of eyes."""
    n = _kp(kps, KP_NOSE)
    if n is not None:
        return n
    return _midpoint(_kp(kps, KP_EYE_L), _kp(kps, KP_EYE_R))


def _bbox_center_norm(bbox):
    """Midpoint of a normalized bbox, in normalized coords."""
    if not bbox:
        return None
    return (0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3]))


# ── Landed-hit zones ─────────────────────────────────────────────────────────
def _head_zone(kps, bbox=None, aspect=1.0):
    """
    Opponent's legal head target: VISUALLY square (correct for frame aspect),
    from top of bbox down to chin (just above shoulder line).

      head_top    = bbox top (YOLO already found the helmet crown)
      head_bottom = chin ≈ shoulder_y − 10% × (shoulder_y − crown_y)
      height      = chin − top        (in normalized coords)
      width       = height × (1 / aspect)   ← key fix
                    where aspect = frame_w / frame_h.
                    Without this, a 1280×720 frame makes any
                    "width_norm = height_norm" box 1.78× wider than tall
                    in actual pixels — the visual "too wide" complaint
                    in user feedback.

    cx falls back nose → eye-mid → ear-mid → bbox-center, clamped inside
    bbox so the square can't drift off the fighter.

    Returns (x1, y1, x2, y2) in normalized coords, or None on heavy
    occlusion (no shoulders or no bbox → UNKNOWN verdict).
    """
    if bbox is None:
        return None
    bx1, by1, bx2, by2 = bbox
    bbox_h = by2 - by1
    if bbox_h < 0.04:
        return None

    sh_l = _kp(kps, KP_SHOULDER_L)
    sh_r = _kp(kps, KP_SHOULDER_R)
    if sh_l is None and sh_r is None:
        return None
    if sh_l is not None and sh_r is not None:
        sh_cy = 0.5 * (sh_l[1] + sh_r[1])
    else:
        sh = sh_l or sh_r
        sh_cy = sh[1]

    head_top_y    = by1                                 # crown of helmet
    chin_gap      = max(0.01 * bbox_h, 0.10 * (sh_cy - head_top_y))
    head_bottom_y = sh_cy - chin_gap
    head_h        = head_bottom_y - head_top_y
    if head_h < 0.02:
        return None

    # Lateral center fallback chain.
    nose  = _kp(kps, KP_NOSE)
    eye_l = _kp(kps, KP_EYE_L)
    eye_r = _kp(kps, KP_EYE_R)
    ear_l = _kp(kps, KP_EAR_L)
    ear_r = _kp(kps, KP_EAR_R)
    bbox_cx = 0.5 * (bx1 + bx2)

    if nose is not None:
        cx = nose[0]
    elif eye_l is not None and eye_r is not None:
        cx = 0.5 * (eye_l[0] + eye_r[0])
    elif ear_l is not None and ear_r is not None:
        cx = 0.5 * (ear_l[0] + ear_r[0])
    elif nose is None and (ear_l or ear_r or eye_l or eye_r):
        face_kp = ear_l or ear_r or eye_l or eye_r
        cx = 0.5 * (bbox_cx + face_kp[0])
    else:
        cx = bbox_cx

    # Aspect-corrected width so the box is a SQUARE IN PIXELS, not in
    # normalized coords. On a 16:9 frame this makes the normalized width
    # 0.56× the normalized height, yielding width_px == height_px.
    width_norm = head_h / max(aspect, 1e-6)
    half_w     = width_norm / 2.0
    cx         = max(bx1 + half_w, min(bx2 - half_w, cx))
    x1 = cx - half_w
    x2 = cx + half_w
    return (x1, head_top_y, x2, head_bottom_y)


def _stomach_zone(kps, aspect=1.0):
    """
    Opponent's legal body target: VISUALLY square (aspect-corrected),
    nipples-to-belt, centered on body midline.

      top    = shoulder_y + STOMACH_TOP_FRAC × (hip_y − shoulder_y)
      bottom = hip_y
      height = bottom − top
      width  = height × (1 / aspect)   ← visually square in pixels
      cx     = midpoint of shoulder-center and hip-center (body axis)

    Like _head_zone, aspect-corrects the width so a landscape frame
    doesn't produce a box that's visually ~1.78× wider than tall.

    Returns None if shoulders or hips can't both be located.
    """
    sh_l = _kp(kps, KP_SHOULDER_L)
    sh_r = _kp(kps, KP_SHOULDER_R)
    hp_l = _kp(kps, KP_HIP_L)
    hp_r = _kp(kps, KP_HIP_R)
    if sh_l is None or sh_r is None or hp_l is None or hp_r is None:
        return None

    sh_cx = 0.5 * (sh_l[0] + sh_r[0])
    sh_y  = 0.5 * (sh_l[1] + sh_r[1])
    hp_cx = 0.5 * (hp_l[0] + hp_r[0])
    hip_y = 0.5 * (hp_l[1] + hp_r[1])
    if hip_y <= sh_y:
        return None

    torso_h = hip_y - sh_y
    top     = sh_y + STOMACH_TOP_FRAC * torso_h
    bottom  = hip_y
    height  = bottom - top
    if height < 0.02:
        return None

    cx = 0.5 * (sh_cx + hp_cx)
    width_norm = height / max(aspect, 1e-6)
    half_w = width_norm / 2.0
    x1 = cx - half_w
    x2 = cx + half_w
    return (x1, top, x2, bottom)


def _pt_in_box(pt, box, pad=0.0):
    """pt and box in same coord system; pad dilates the box on all sides."""
    if pt is None or box is None:
        return False
    x, y = pt
    x1, y1, x2, y2 = box
    return (x1 - pad) <= x <= (x2 + pad) and (y1 - pad) <= y <= (y2 + pad)


def _glove_tip(wrist, elbow):
    """
    Extrapolate glove tip ≈ wrist + HIT_GLOVE_EXT_NORM × (wrist − elbow).
    Falls back to the raw wrist if elbow is missing.
    """
    if wrist is None:
        return None
    if elbow is None:
        return wrist
    dx = wrist[0] - elbow[0]
    dy = wrist[1] - elbow[1]
    return (wrist[0] + HIT_GLOVE_EXT_NORM * dx,
            wrist[1] + HIT_GLOVE_EXT_NORM * dy)


def _bbox_iou(a, b):
    """IoU of two normalized bboxes (x1,y1,x2,y2). 0 if either is None."""
    if a is None or b is None:
        return 0.0
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (a[2] - a[0])) * max(0.0, (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 1e-9 else 0.0


# Minimum arm extension (wrist-from-shoulder / torso_height) for the
# attacker's punching hand at the peak frame. 0.35 was too permissive —
# let through defensive motions that happened to align with the opponent.
# 0.50 requires a genuinely committed arm-out posture.
LANDED_HIT_EXT_MIN   = 0.50

# If the attacker's bbox and target's bbox overlap this much, we're in a
# clinch/tangle — the glove-in-zone test is unreliable because 2D proximity
# doesn't separate "glove inside target" from "bodies pressed together."
# Mark the verdict UNKNOWN rather than risk a false positive.
LANDED_HIT_CLINCH_IOU = 0.55


def _verdict_at_peak(attacker_kps, target_kps,
                     target_bbox=None, attacker_bbox=None,
                     aspect=1.0):
    """
    Classify a punch peak as landed_head / landed_body / missed / unknown.

    UNKNOWN:  opponent has no shoulders visible (can't compute either zone)
              OR attacker has no active wrist we can identify.
    LANDED_*: the ACTIVELY PUNCHING glove (the more-extended of the two
              wrists, measured from shoulder center normalized by torso
              height) sits inside the opponent's head or stomach zone, with
              a glove-radius pad.
    MISSED:   zones computable, active glove outside both zones — covers
              overshoots, stop-shorts, and wide/hook misses.

    Critically we check ONLY the active wrist, never the guard wrist. At a
    punch peak, one hand is extended toward the opponent and the other is
    up near the face guarding. Testing the guard wrist creates false
    positives when its glove extrapolation coincidentally lands inside an
    opponent zone (guard wrist points "up" and can project into the head
    zone of an opponent who happens to be at the right relative position).
    """
    # Clinch guard: if the two bboxes overlap heavily, 2D glove-in-zone is
    # unreliable (bodies pressed together). Mark UNKNOWN rather than guess.
    if _bbox_iou(attacker_bbox, target_bbox) >= LANDED_HIT_CLINCH_IOU:
        return ("unknown", None)

    head = _head_zone(target_kps, target_bbox, aspect=aspect)
    stom = _stomach_zone(target_kps, aspect=aspect)
    if head is None and stom is None:
        return ("unknown", None)

    sc = _shoulder_center(attacker_kps)
    th = _torso_height(attacker_kps)
    if sc is None or th is None or th <= 1e-6:
        return ("unknown", None)

    # Identify the active (punching) wrist by extension-from-shoulder,
    # normalized by torso height. The guard wrist sits near the face, so its
    # shoulder-distance is small; the punching wrist is extended and reads
    # a larger ratio.
    best_wrist = None
    best_elbow = None
    best_ext   = -1.0
    for wi, ei in ((KP_WRIST_L, KP_ELBOW_L), (KP_WRIST_R, KP_ELBOW_R)):
        w = _kp(attacker_kps, wi)
        if w is None:
            continue
        ratio = _dist(w, sc) / th
        if ratio > best_ext:
            best_ext = ratio
            best_wrist = w
            best_elbow = _kp(attacker_kps, ei)

    if best_wrist is None:
        return ("unknown", None)

    # Commitment gate: the active arm must be clearly extended, not just
    # raised in guard or drifting. 0.50 torso-heights away from shoulder
    # center = a genuine reach toward the opponent; below that, call it
    # UNKNOWN so we don't label a defensive/guard motion as a landed hit.
    if best_ext < LANDED_HIT_EXT_MIN:
        return ("unknown", None)

    glove = _glove_tip(best_wrist, best_elbow)

    if _pt_in_box(glove, head, pad=HIT_WRIST_PAD_NORM):
        return ("landed_head", glove)
    if _pt_in_box(glove, stom, pad=HIT_WRIST_PAD_NORM):
        return ("landed_body", glove)
    return ("missed", glove)


def _punch_score(kps, prev_kps, toward_dir, bbox=None):
    """
    Per-frame punch-likelihood score for one fighter (v3).

        score = max over both wrists of
                  extension_ratio × zoom_adjusted_velocity × direction_factor

    • extension_ratio — distance(wrist, shoulder-center) / torso_height.
    • zoom_adjusted_velocity — body-relative wrist displacement (shoulder
      motion subtracted), with a partial normalization that boosts
      velocity on zoomed-out clips. If the fighter's bbox-height is below
      PUNCH_BBOX_REF_H, multiply velocity by PUNCH_BBOX_REF_H/bbox_h
      (capped at 2×). This keeps score distributions comparable across a
      2.7× range of camera zoom. Fighters larger than the reference are
      unchanged — their velocities are already well-resolved by pixels.
      (Direct torso_height normalization was tried and rejected earlier —
      it amplified noise by dividing by a small, jittery keypoint
      distance. bbox-height is stable enough to divide by.)
    • direction_factor — cos(angle) between velocity and me→op vector,
      remapped to [PUNCH_DIR_FLOOR, 1]. Floor lets close-range hooks
      (roughly perpendicular motion) score on par with straight punches
      instead of being halved by the 0.5 midpoint remap.

    Returns 0.0 when required keypoints are missing.
    """
    sc = _shoulder_center(kps)
    th = _torso_height(kps)
    if sc is None or th is None or th <= 1e-6:
        return 0.0
    prev_sc = _shoulder_center(prev_kps) if prev_kps else None

    # Zoom correction: on zoomed-out clips fighters occupy less pixel area
    # so raw velocities compress. Scale velocity up (capped) when fighter
    # bbox is smaller than the reference. No change for large fighters.
    if bbox is not None:
        bbox_h = max(bbox[3] - bbox[1], 0.10)
        zoom_boost = (min(2.0, PUNCH_BBOX_REF_H / bbox_h)
                      if bbox_h < PUNCH_BBOX_REF_H else 1.0)
    else:
        zoom_boost = 1.0

    if toward_dir is not None:
        tx, ty = toward_dir
        tmag = math.hypot(tx, ty)
    else:
        tx, ty, tmag = 0.0, 0.0, 0.0

    best = 0.0
    for wi in (KP_WRIST_L, KP_WRIST_R):
        w = _kp(kps, wi)
        if w is None or prev_kps is None or prev_sc is None:
            continue
        pw = _kp(prev_kps, wi)
        if pw is None:
            continue
        cur_rel  = (w[0]  - sc[0],      w[1]  - sc[1])
        prev_rel = (pw[0] - prev_sc[0], pw[1] - prev_sc[1])
        vx = cur_rel[0] - prev_rel[0]
        vy = cur_rel[1] - prev_rel[1]
        vmag = math.hypot(vx, vy) * zoom_boost
        if vmag < 1e-6:
            continue
        ext = _dist(w, sc) / th
        if tmag < 1e-6:
            dir_factor = 0.5
        else:
            cos_sim = (vx * tx + vy * ty) / (vmag * tmag)
            dir_factor = max(PUNCH_DIR_FLOOR,
                             min(1.0, (cos_sim + 1.0) / 2.0))
        score = ext * vmag * dir_factor
        if score > best:
            best = score
    return best


def _wrist_activity(kps, prev_kps):
    """
    Body-relative wrist-motion magnitude for one fighter in one frame.
    Max over both wrists of:
        || (wrist_t − shoulder_t) − (wrist_{t-1} − shoulder_{t-1}) ||

    This is the "are the hands moving fast?" signal — unlike _punch_score
    it does NOT require arm extension, does NOT require motion toward the
    opponent, and so fires for hooks, short punches, body shots, clinch
    work, parries, and defensive hand motion. Correct tool for CLOSE and
    CLINCH states where _punch_score undercounts real punches (short range
    means no extension; hooks curve so direction isn't "toward opponent").

    Shoulder motion is subtracted so the fighter's own footwork doesn't
    inflate the signal. Returns 0.0 if keypoints missing.
    """
    if kps is None or prev_kps is None:
        return 0.0
    sc = _shoulder_center(kps)
    prev_sc = _shoulder_center(prev_kps)
    if sc is None or prev_sc is None:
        return 0.0
    best = 0.0
    for wi in (KP_WRIST_L, KP_WRIST_R):
        w  = _kp(kps, wi)
        pw = _kp(prev_kps, wi)
        if w is None or pw is None:
            continue
        cur_rel  = (w[0]  - sc[0],      w[1]  - sc[1])
        prev_rel = (pw[0] - prev_sc[0], pw[1] - prev_sc[1])
        vmag = math.hypot(cur_rel[0] - prev_rel[0],
                          cur_rel[1] - prev_rel[1])
        if vmag > best:
            best = vmag
    return best


def _kp_visibility(kps) -> float:
    """Fraction of COCO-17 keypoints confident at or above KP_CONF."""
    if not kps:
        return 0.0
    return _confident_kp_count(kps) / 17.0


def _classify_state(my_kps, op_kps, my_bbox, op_bbox,
                    my_to_op_scaled):
    """
    Classify one frame as OPEN / CLOSE / CLINCH / OCCLUSION.

    Priority:
      1. OCCLUSION if either fighter's keypoint visibility < threshold
         (can't reliably score activity; carry-forward with decay)
      2. CLINCH   if bbox IoU exceeds threshold (2D geometry unreliable,
         fall back to activity-only signals)
      3. CLOSE    if inter-fighter distance (in median-scale units) is
         below threshold (infighting but still visible)
      4. OPEN     otherwise

    Inputs are what we already compute elsewhere, so this is cheap.
    """
    my_vis = _kp_visibility(my_kps)
    op_vis = _kp_visibility(op_kps)
    if my_vis < STATE_OCCL_KP_FRAC or op_vis < STATE_OCCL_KP_FRAC:
        return STATE_OCCLUSION
    if _bbox_iou(my_bbox, op_bbox) >= STATE_CLINCH_IOU:
        return STATE_CLINCH
    if (my_to_op_scaled is not None and
            my_to_op_scaled < STATE_CLOSE_DIST_SCALES):
        return STATE_CLOSE
    return STATE_OPEN


def _decay_toward_neutral(s_prev: float, dt: float,
                          tau: float = STATE_DECAY_TAU_S) -> float:
    """Exponential decay toward 50. s(t) = 50 + (s_prev − 50) × e^(−dt/τ)."""
    return 50.0 + (s_prev - 50.0) * math.exp(-max(dt, 0.0) / max(tau, 1e-6))


def _detect_punch_peaks(scores: list[float], fps: float,
                        min_score: float | None = None,
                        min_sep_s: float | None = None) -> list[int]:
    # Resolve defaults at call time so module-level constant changes take
    # effect without needing to reload the module (pitfall from earlier: a
    # default-arg binding evaluated at def time made threshold sweeps silent
    # no-ops).
    if min_score is None:
        min_score = PUNCH_SCORE_MIN
    if min_sep_s is None:
        min_sep_s = PUNCH_PEAK_SEP_S
    """
    Return indices of local maxima in `scores` above `min_score`, with at
    least `min_sep_s` seconds between successive peaks. Each punch event
    produces exactly one peak — no separate deduplication pass needed.
    """
    n = len(scores)
    if n == 0:
        return []
    half = max(1, int(round(min_sep_s * fps)))
    peaks: list[int] = []
    i = 0
    while i < n:
        if scores[i] < min_score:
            i += 1
            continue
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        is_peak = True
        for j in range(lo, hi):
            if j != i and scores[j] > scores[i]:
                is_peak = False
                break
        if is_peak:
            peaks.append(i)
            i += half + 1
        else:
            i += 1
    return peaks


# ── Main computation ─────────────────────────────────────────────────────────
def compute(session_dir: Path) -> dict:
    enriched_path = session_dir / "sam2_enriched.json"
    arena_path    = session_dir / "arena.json"
    if not enriched_path.exists():
        return {"ok": False, "error": "no sam2_enriched.json"}
    enriched = json.loads(enriched_path.read_text())
    frames   = enriched.get("frames", [])
    fps_pose = float(enriched.get("fps_pose") or 15.0)
    fw       = float(enriched.get("frame_w") or 1)
    fh       = float(enriched.get("frame_h") or 1)
    if len(frames) < 4:
        return {"ok": False, "error": "too few frames"}

    # NOTE: 1€ keypoint smoothing + motion-blur gate are implemented below
    # (OneEuroFilter, _smooth_keypoints_inplace, _confident_kp_count) but are
    # currently NOT invoked. Earlier tuning over-attenuated real punches
    # (count dropped ~0/0) and the user preferred the un-smoothed state after
    # the KLT upgrade. Left in the file so we can revisit with better tuning
    # (likely needs per-keypoint β: high for wrists, low for torso) later.

    arena = json.loads(arena_path.read_text()) if arena_path.exists() else {}
    arena_ok     = bool(arena.get("ok"))
    tier         = arena.get("confidence_tier", "none") if arena_ok else "none"
    tier_weight  = TIER_WEIGHT.get(tier, 0.0)
    centroid     = arena.get("centroid") if arena_ok else None
    my_traj_ref  = arena.get("my_traj_ref", []) if arena_ok else []
    op_traj_ref  = arena.get("op_traj_ref", []) if arena_ok else []
    median_scale = float(arena.get("median_scale_px") or
                         arena.get("median_fighter_height_px") or 100.0)
    my_vis       = float(arena.get("my_visibility_score", 0.0))
    op_vis       = float(arena.get("op_visibility_score", 0.0))
    axis_len     = float(arena.get("axis_len", 0.0))
    arena_radius = max(axis_len, 1.0)

    n = len(frames)

    # ── Per-frame signals ─────────────────────────────────────────────────
    # my_punch / op_punch are per-frame "arm active in punch posture" flags
    # (used by Aggression for sustained-activity credit). my_punch_score is
    # the directional peak-detection score (used for the total punch count
    # via _detect_punch_peaks). Keeping both lets Aggression stay well-
    # defined per-frame while the total count benefits from peak-based
    # precision that rejects blocks/feints/retractions.
    my_punch       = [False] * n
    op_punch       = [False] * n
    my_punch_score = [0.0]   * n
    op_punch_score = [0.0]   * n
    my_guard_px    = [None]  * n    # 0..1 guard score per fighter per frame
    op_guard_px    = [None]  * n
    my_head_img    = [None]  * n
    op_head_img    = [None]  * n
    # Phase 2 signals — needed by the per-state aggression blend.
    my_wrist_act   = [0.0]   * n   # body-relative wrist-motion magnitude
    op_wrist_act   = [0.0]   * n
    my_bbox_push   = [0.0]   * n   # own-bbox-center displacement per frame
    op_bbox_push   = [0.0]   * n
    my_head_move   = [0.0]   * n   # own-head displacement per frame (body rel)
    op_head_move   = [0.0]   * n
    state          = [STATE_OCCLUSION] * n   # frame-level interaction state

    for i, f in enumerate(frames):
        my_kps = f.get("my_kps")
        op_kps = f.get("op_kps")
        prev = frames[i - 1] if i > 0 else None
        prev_my = prev.get("my_kps") if prev else None
        prev_op = prev.get("op_kps") if prev else None

        my_punch[i]    = _is_punching(my_kps, prev_my)
        op_punch[i]    = _is_punching(op_kps, prev_op)
        my_guard_px[i] = _guard_score_frame(my_kps)
        op_guard_px[i] = _guard_score_frame(op_kps)

        # Directional punch score — requires both fighters' centers to
        # compute the "toward opponent" vector. Uses normalized bbox
        # centers, so both wrist velocity and the direction vector live
        # in the same coordinate space.
        my_center = _bbox_center_norm(f.get("my_bbox"))
        op_center = _bbox_center_norm(f.get("op_bbox"))
        if my_center and op_center:
            me_to_op = (op_center[0] - my_center[0],
                        op_center[1] - my_center[1])
            op_to_me = (-me_to_op[0], -me_to_op[1])
        else:
            me_to_op = None
            op_to_me = None
        my_punch_score[i] = _punch_score(my_kps, prev_my, me_to_op,
                                         bbox=f.get("my_bbox"))
        op_punch_score[i] = _punch_score(op_kps, prev_op, op_to_me,
                                         bbox=f.get("op_bbox"))
        # Convert normalized keypoints to image pixels (head position used
        # only as a relative signal, not in arena coords).
        def _hp(kps):
            h = _head_pos(kps)
            return None if h is None else (h[0] * fw, h[1] * fh)
        my_head_img[i] = _hp(my_kps)
        op_head_img[i] = _hp(op_kps)

        # Phase 2: wrist-activity (non-directional, non-extension-gated)
        # and body/head push magnitudes for the per-state blend.
        my_wrist_act[i] = _wrist_activity(my_kps, prev_my)
        op_wrist_act[i] = _wrist_activity(op_kps, prev_op)
        if my_center and prev:
            prev_mc = _bbox_center_norm(prev.get("my_bbox"))
            if prev_mc:
                my_bbox_push[i] = math.hypot(my_center[0] - prev_mc[0],
                                             my_center[1] - prev_mc[1])
        if op_center and prev:
            prev_oc = _bbox_center_norm(prev.get("op_bbox"))
            if prev_oc:
                op_bbox_push[i] = math.hypot(op_center[0] - prev_oc[0],
                                             op_center[1] - prev_oc[1])
        # Head displacement (body-relative by subtracting shoulder motion).
        if i > 0 and my_head_img[i] is not None and my_head_img[i-1] is not None:
            sc   = _shoulder_center(my_kps);     psc = _shoulder_center(prev_my)
            if sc is not None and psc is not None:
                dx = (my_head_img[i][0]/fw - sc[0]) - (my_head_img[i-1][0]/fw - psc[0])
                dy = (my_head_img[i][1]/fh - sc[1]) - (my_head_img[i-1][1]/fh - psc[1])
                my_head_move[i] = math.hypot(dx, dy)
        if i > 0 and op_head_img[i] is not None and op_head_img[i-1] is not None:
            sc   = _shoulder_center(op_kps);     psc = _shoulder_center(prev_op)
            if sc is not None and psc is not None:
                dx = (op_head_img[i][0]/fw - sc[0]) - (op_head_img[i-1][0]/fw - psc[0])
                dy = (op_head_img[i][1]/fh - sc[1]) - (op_head_img[i-1][1]/fh - psc[1])
                op_head_move[i] = math.hypot(dx, dy)

    # ── Per-frame arena signals ───────────────────────────────────────────
    # Distance each fighter is from the centroid (ref coords), in scale units.
    my_to_center = [None] * n
    op_to_center = [None] * n
    # Distance between fighters in scale units.
    my_to_op     = [None] * n
    # Directed closing speed: >0 when ME closed distance to OP this frame.
    my_closing   = [0.0] * n
    op_closing   = [0.0] * n

    if arena_ok and centroid is not None:
        for i in range(n):
            my_r = my_traj_ref[i] if i < len(my_traj_ref) else None
            op_r = op_traj_ref[i] if i < len(op_traj_ref) else None
            if my_r is not None:
                my_to_center[i] = _dist(my_r, centroid) / median_scale
            if op_r is not None:
                op_to_center[i] = _dist(op_r, centroid) / median_scale
            if my_r is not None and op_r is not None:
                my_to_op[i] = _dist(my_r, op_r) / median_scale

        for i in range(1, n):
            if (my_to_op[i] is not None and my_to_op[i - 1] is not None):
                closing = my_to_op[i - 1] - my_to_op[i]    # positive = closing
                # Split credit by who moved more toward the opponent.
                my_r  = my_traj_ref[i]
                my_rp = my_traj_ref[i - 1]
                op_r  = op_traj_ref[i]
                op_rp = op_traj_ref[i - 1]
                if all(p is not None for p in (my_r, my_rp, op_r, op_rp)):
                    my_mv = _dist(my_r, my_rp) / median_scale
                    op_mv = _dist(op_r, op_rp) / median_scale
                    tot = max(my_mv + op_mv, 1e-6)
                    my_closing[i] = closing * (my_mv / tot)
                    op_closing[i] = closing * (op_mv / tot)

    # ── Per-frame state classification (Phase 2) ──────────────────────────
    for i, f in enumerate(frames):
        state[i] = _classify_state(
            f.get("my_kps"), f.get("op_kps"),
            f.get("my_bbox"), f.get("op_bbox"),
            my_to_op[i] if i < len(my_to_op) else None,
        )

    # ── Per-second aggregation ────────────────────────────────────────────
    duration_s = max(1, int(math.ceil(frames[-1].get("time_s", 0.0))) + 1)
    by_sec: dict[int, list[int]] = {}
    for i, f in enumerate(frames):
        s = int(f.get("time_s", 0.0))
        by_sec.setdefault(s, []).append(i)

    # ── Metric 1: Ring Control (arena-dependent) ───────────────────────────
    my_ring_sec = [50] * duration_s
    op_ring_sec = [50] * duration_s
    if arena_ok:
        for s in range(duration_s):
            idxs = by_sec.get(s, [])
            me_closer = 0
            op_closer = 0
            for i in idxs:
                if my_to_center[i] is None or op_to_center[i] is None:
                    continue
                if my_to_center[i] < op_to_center[i]:
                    me_closer += 1
                else:
                    op_closer += 1
            total = me_closer + op_closer
            if total > 0:
                my_ring_sec[s] = round(100 * me_closer / total)
                op_ring_sec[s] = 100 - my_ring_sec[s]

    # ── Metric 2: Effective Aggression (state-aware, Phase 2) ──────────────
    # Per frame, compute a raw aggression score per fighter using a blend
    # chosen by the frame's state (OPEN / CLOSE / CLINCH / OCCLUSION). Then
    # aggregate to a per-second percentage share between ME and OP.
    #
    # OPEN: punch_score + closing-forward + op-retreating. The "clean
    #       distance" regime — _punch_score is reliable, so it carries most
    #       of the weight; closing is the rest.
    # CLOSE: wrist_activity takes over because short punches and hooks
    #       don't produce extension/direction — _punch_score undercounts.
    #       punch_score still contributes (clean short jabs still register)
    #       but down-weighted.
    # CLINCH: no punch_score at all; activity-only signals (wrist motion,
    #       head motion, body pushing each other around). Whoever is
    #       actively working gets the credit.
    # OCCLUSION: no new credit; exponentially decay the last seen
    #       per-second score toward 50/50 with τ = STATE_DECAY_TAU_S.

    # Rough normalization constants for the three signals — keep each
    # in roughly the same dynamic range so the weights above can be
    # interpreted. Tuned against observed magnitudes in real clips.
    PSCORE_NORM = 0.20   # typical punch_score peak ≈ 0.05–0.15
    WACT_NORM   = 0.05   # typical wrist_activity peak ≈ 0.02–0.06
    BPUSH_NORM  = 0.02   # typical per-frame bbox-center displacement
    HMOVE_NORM  = 0.015  # typical head-relative motion

    def _nrm(x, d):
        return max(0.0, min(1.0, x / max(d, 1e-6)))

    my_agg_frame = [0.0] * n
    op_agg_frame = [0.0] * n

    for i in range(n):
        st = state[i]
        if st == STATE_OPEN:
            # Closing = own forward pressure; op-retreat is our fighter's
            # pressure forcing the opponent back (distinct credit).
            me_close = 1.0 if my_closing[i] > ADVANCE_EPS_SCALE else 0.0
            op_close = 1.0 if op_closing[i] > ADVANCE_EPS_SCALE else 0.0
            # "op retreating under ME's pressure" = ME moved forward while
            # distance grew, OR OP moved backward (negative op_closing).
            me_force_op_retreat = 1.0 if (op_closing[i] < -ADVANCE_EPS_SCALE) else 0.0
            op_force_me_retreat = 1.0 if (my_closing[i] < -ADVANCE_EPS_SCALE) else 0.0
            W = AGG_OPEN_W
            my_agg_frame[i] = (W["punch"]         * _nrm(my_punch_score[i], PSCORE_NORM) +
                               W["close_retreat"] * me_close +
                               W["op_retreat"]    * me_force_op_retreat)
            op_agg_frame[i] = (W["punch"]         * _nrm(op_punch_score[i], PSCORE_NORM) +
                               W["close_retreat"] * op_close +
                               W["op_retreat"]    * op_force_me_retreat)
        elif st == STATE_CLOSE:
            W = AGG_CLOSE_W
            my_agg_frame[i] = (W["wrist"]    * _nrm(my_wrist_act[i], WACT_NORM) +
                               W["punch"]    * _nrm(my_punch_score[i], PSCORE_NORM) +
                               W["body_push"]* _nrm(my_bbox_push[i], BPUSH_NORM))
            op_agg_frame[i] = (W["wrist"]    * _nrm(op_wrist_act[i], WACT_NORM) +
                               W["punch"]    * _nrm(op_punch_score[i], PSCORE_NORM) +
                               W["body_push"]* _nrm(op_bbox_push[i], BPUSH_NORM))
        elif st == STATE_CLINCH:
            W = AGG_CLINCH_W
            my_agg_frame[i] = (W["wrist"]     * _nrm(my_wrist_act[i], WACT_NORM) +
                               W["head_move"] * _nrm(my_head_move[i], HMOVE_NORM) +
                               W["bbox_push"] * _nrm(my_bbox_push[i], BPUSH_NORM))
            op_agg_frame[i] = (W["wrist"]     * _nrm(op_wrist_act[i], WACT_NORM) +
                               W["head_move"] * _nrm(op_head_move[i], HMOVE_NORM) +
                               W["bbox_push"] * _nrm(op_bbox_push[i], BPUSH_NORM))
        # OCCLUSION handled at the per-second aggregation step (decay).

    # Aggregate to per-second percentage share (ME vs OP). OCCLUSION seconds
    # exponentially decay from the last non-occlusion second toward 50/50.
    my_agg_sec = [50] * duration_s
    op_agg_sec = [50] * duration_s
    last_nonoccl_s = -1
    for s in range(duration_s):
        idxs = by_sec.get(s, [])
        if not idxs:
            continue
        # What fraction of this second was OCCLUSION?
        n_occl = sum(1 for i in idxs if state[i] == STATE_OCCLUSION)
        occl_frac = n_occl / max(len(idxs), 1)

        if occl_frac >= 0.5:
            # Mostly occluded — decay from last known non-occlusion second.
            if last_nonoccl_s >= 0:
                dt = s - last_nonoccl_s
                my_agg_sec[s] = round(_decay_toward_neutral(my_agg_sec[last_nonoccl_s], dt))
                op_agg_sec[s] = 100 - my_agg_sec[s]
            # else: leaves default 50/50, which is correct (no prior signal)
            continue

        # Normal blend: sum per-frame scores across the second (exclude
        # OCCLUSION frames from the numerator/denominator).
        me_sum = sum(my_agg_frame[i] for i in idxs if state[i] != STATE_OCCLUSION)
        op_sum = sum(op_agg_frame[i] for i in idxs if state[i] != STATE_OCCLUSION)
        tot = me_sum + op_sum
        if tot > 1e-6:
            my_agg_sec[s] = round(100 * me_sum / tot)
            op_agg_sec[s] = 100 - my_agg_sec[s]
            last_nonoccl_s = s

    # ── Metric 3: Movement (arena-only) ────────────────────────────────────
    # Pure footwork/engine: path length per second (in scales), no punch
    # dependency. Tier-weighted since it's arena-dependent. Answers
    # "who's more physically active" independent of our punch detector —
    # useful as a pace-independent cross-check for Volume below.
    my_path = [0.0] * n
    op_path = [0.0] * n
    for i in range(1, n):
        if i < len(my_traj_ref) and my_traj_ref[i] is not None and my_traj_ref[i - 1] is not None:
            my_path[i] = _dist(my_traj_ref[i], my_traj_ref[i - 1]) / median_scale
        if i < len(op_traj_ref) and op_traj_ref[i] is not None and op_traj_ref[i - 1] is not None:
            op_path[i] = _dist(op_traj_ref[i], op_traj_ref[i - 1]) / median_scale

    my_movement_sec = [50] * duration_s
    op_movement_sec = [50] * duration_s
    for s in range(duration_s):
        idxs = by_sec.get(s, [])
        if not idxs:
            continue
        me_mv = sum(my_path[i] for i in idxs) * tier_weight
        op_mv = sum(op_path[i] for i in idxs) * tier_weight
        tot = me_mv + op_mv
        if tot > 1e-6:
            my_movement_sec[s] = round(100 * me_mv / tot)
            op_movement_sec[s] = 100 - my_movement_sec[s]

    # ── Metric 3b: Volume (body-only, punch share) ─────────────────────────
    # Pure punch activity: each fighter's share of detected punch-active
    # frames per second. Body-relative — not tier-weighted. Uses the
    # per-frame activity flag (my_punch[i]) so the per-second series
    # reflects sustained activity; the total count comes from peak
    # detection (my_punches/op_punches in the summary).
    my_volume_sec = [50] * duration_s
    op_volume_sec = [50] * duration_s
    for s in range(duration_s):
        idxs = by_sec.get(s, [])
        if not idxs:
            continue
        me_punches = sum(1 for i in idxs if my_punch[i])
        op_punches = sum(1 for i in idxs if op_punch[i])
        tot = me_punches + op_punches
        if tot > 0:
            my_volume_sec[s] = round(100 * me_punches / tot)
            op_volume_sec[s] = 100 - my_volume_sec[s]

    # Peak-detect on the directional score. Moved UP from post-summary
    # because the new Defense metric needs the peak lists (to define "ME is
    # under attack" windows). Peak counts themselves are reported in the
    # summary at the bottom.
    my_peaks = _detect_punch_peaks(my_punch_score, fps_pose)
    op_peaks = _detect_punch_peaks(op_punch_score, fps_pose)

    # ── Metric 4: Defense (4-pillar, coach-structured) ─────────────────────
    # Graded on how the fighter responds when being ATTACKED by the opponent.
    # Four pillars — distance (was OP out of reach?), head_move (did the head
    # move?), guard (hands up?), counter (active parry/engage?). Weighted by
    # DEF_W above. If a fighter isn't being attacked in a given second, their
    # quality stays neutral (50) — "no data, no judgment."
    #
    # The "under attack" window runs from a few frames before the opponent's
    # punch peak (so anticipatory defense counts) through several frames
    # after (the response phase). Peaks come from my_peaks / op_peaks, which
    # are the same lists the total-punch-count uses.
    my_attacked = [False] * n
    op_attacked = [False] * n
    for pi in op_peaks:                                   # OP throws → ME defends
        for j in range(pi - DEF_ATTACK_WIN_PRE_F,
                       pi + DEF_ATTACK_WIN_POST_F + 1):
            if 0 <= j < n: my_attacked[j] = True
    for pi in my_peaks:                                   # ME throws → OP defends
        for j in range(pi - DEF_ATTACK_WIN_PRE_F,
                       pi + DEF_ATTACK_WIN_POST_F + 1):
            if 0 <= j < n: op_attacked[j] = True

    def _dist_pillar(d):
        """Linear ramp: 0 at DEF_DIST_NEAR, 1 at DEF_DIST_FAR."""
        if d is None:
            return 0.0
        x = (d - DEF_DIST_NEAR) / max(DEF_DIST_FAR - DEF_DIST_NEAR, 1e-6)
        return max(0.0, min(1.0, x))

    # Per-frame pillar values for each fighter (continuous, 0..1).
    my_pillars = [None] * n
    op_pillars = [None] * n
    for i in range(n):
        d = my_to_op[i] if i < len(my_to_op) else None
        dist_val = _dist_pillar(d)
        my_pillars[i] = (
            dist_val,
            _nrm(my_head_move[i], DEF_HEAD_NORM),
            my_guard_px[i] if my_guard_px[i] is not None else 0.0,
            _nrm(my_wrist_act[i], DEF_COUNTER_NORM),
        )
        op_pillars[i] = (
            dist_val,      # distance is symmetric
            _nrm(op_head_move[i], DEF_HEAD_NORM),
            op_guard_px[i] if op_guard_px[i] is not None else 0.0,
            _nrm(op_wrist_act[i], DEF_COUNTER_NORM),
        )

    def _def_score(pillars):
        d, h, g, c = pillars
        return (DEF_W["distance"]  * d +
                DEF_W["head_move"] * h +
                DEF_W["guard"]     * g +
                DEF_W["counter"]   * c)

    # Per-second qualities + pillar breakdowns (for later UI / deep analytics).
    my_def_sec   = [50] * duration_s
    op_def_sec   = [50] * duration_s
    my_def_dist_sec  = [50] * duration_s     # pillar breakdowns 0-100
    my_def_head_sec  = [50] * duration_s
    my_def_guard_sec = [50] * duration_s
    op_def_dist_sec  = [50] * duration_s
    op_def_head_sec  = [50] * duration_s
    op_def_guard_sec = [50] * duration_s

    for s in range(duration_s):
        idxs = by_sec.get(s, [])
        if not idxs:
            continue

        me_att_idx = [i for i in idxs if my_attacked[i] and state[i] != STATE_OCCLUSION]
        op_att_idx = [i for i in idxs if op_attacked[i] and state[i] != STATE_OCCLUSION]

        # Quality 0..1 — average across attacked frames; 0.5 if not attacked.
        my_q = (sum(_def_score(my_pillars[i]) for i in me_att_idx) / len(me_att_idx)
                if me_att_idx else 0.5)
        op_q = (sum(_def_score(op_pillars[i]) for i in op_att_idx) / len(op_att_idx)
                if op_att_idx else 0.5)
        tot = my_q + op_q
        if tot > 1e-6:
            my_def_sec[s] = round(100 * my_q / tot)
            op_def_sec[s] = 100 - my_def_sec[s]

        # Pillar breakdowns (0-100 absolute quality), for drill-down UI later.
        def _pillar_pct(idxs_att, pillar_getter):
            if not idxs_att:
                return 50
            return round(100 * sum(pillar_getter(i) for i in idxs_att) / len(idxs_att))
        my_def_dist_sec[s]  = _pillar_pct(me_att_idx, lambda i: my_pillars[i][0])
        my_def_head_sec[s]  = _pillar_pct(me_att_idx, lambda i: my_pillars[i][1])
        my_def_guard_sec[s] = _pillar_pct(me_att_idx, lambda i: my_pillars[i][2])
        op_def_dist_sec[s]  = _pillar_pct(op_att_idx, lambda i: op_pillars[i][0])
        op_def_head_sec[s]  = _pillar_pct(op_att_idx, lambda i: op_pillars[i][1])
        op_def_guard_sec[s] = _pillar_pct(op_att_idx, lambda i: op_pillars[i][2])

    # ── Metric 4b: Head Movement (body-relative, standalone Tier A) ─────────
    # Body-relative head displacement magnitude, summed per second. "How much
    # is each fighter moving their head in general" — a cheap proxy for
    # slipping, rolling, feinting. Differs from the defense head_move pillar
    # which is graded only during opponent-attack windows: this one is
    # measured continuously. Tier A because the input (nose/ears) is one of
    # the more reliable keypoint sets and it's fully body-relative.
    my_head_mov_sec = [50] * duration_s
    op_head_mov_sec = [50] * duration_s
    for s in range(duration_s):
        idxs = by_sec.get(s, [])
        if not idxs:
            continue
        me_hm = sum(my_head_move[i] for i in idxs)
        op_hm = sum(op_head_move[i] for i in idxs)
        tot = me_hm + op_hm
        if tot > 1e-6:
            my_head_mov_sec[s] = round(100 * me_hm / tot)
            op_head_mov_sec[s] = 100 - my_head_mov_sec[s]

    # ── Metric 5: Guard (body-relative, no tier weighting) ─────────────────
    my_guard_sec = [50] * duration_s
    op_guard_sec = [50] * duration_s
    for s in range(duration_s):
        idxs = by_sec.get(s, [])
        me_vals = [my_guard_px[i] for i in idxs if my_guard_px[i] is not None]
        op_vals = [op_guard_px[i] for i in idxs if op_guard_px[i] is not None]
        if me_vals:
            my_guard_sec[s] = round(100 * float(np.mean(me_vals)))
        if op_vals:
            op_guard_sec[s] = round(100 * float(np.mean(op_vals)))

    # ── Summary scores = per-second average. ───────────────────────────────
    def _avg(xs):
        return round(float(np.mean(xs))) if xs else 50

    # Per-fighter visibility ratio also affects final summary for that
    # fighter's arena-dependent metrics.
    def _apply_fighter_visibility(score, visibility):
        """Pull arena-dependent scores toward 50 (neutral) when visibility is low."""
        if visibility >= 0.75:
            return score
        k = max(0.0, min(1.0, visibility / 0.75))
        return round(50 + (score - 50) * k)

    my_ring      = _apply_fighter_visibility(_avg(my_ring_sec), my_vis)
    op_ring      = _apply_fighter_visibility(_avg(op_ring_sec), op_vis)
    my_aggr      = _apply_fighter_visibility(_avg(my_agg_sec), my_vis)
    op_aggr      = _apply_fighter_visibility(_avg(op_agg_sec), op_vis)
    my_movement  = _apply_fighter_visibility(_avg(my_movement_sec), my_vis)
    op_movement  = _apply_fighter_visibility(_avg(op_movement_sec), op_vis)
    my_volume    = _avg(my_volume_sec)  # body-relative, never downweighted
    op_volume    = _avg(op_volume_sec)
    my_def       = _apply_fighter_visibility(_avg(my_def_sec), my_vis)
    op_def       = _apply_fighter_visibility(_avg(op_def_sec), op_vis)
    my_head_mov  = _avg(my_head_mov_sec)   # body-relative; standalone metric
    op_head_mov  = _avg(op_head_mov_sec)
    my_guard     = _avg(my_guard_sec)   # body-relative, never downweighted
    op_guard     = _avg(op_guard_sec)

    # my_peaks / op_peaks were computed above for the Defense metric.
    my_punches_total = len(my_peaks)
    op_punches_total = len(op_peaks)

    # ── Landed-hit verdict per peak ───────────────────────────────────────
    # At each peak frame, classify the punch as landed_head / landed_body /
    # missed / unknown against the opponent's head & stomach zones. Events
    # are emitted in normalized coords so diagnostics can draw them directly.
    # Frame aspect ratio so zones can be drawn visually-square in pixels.
    aspect = (fw / fh) if (fw > 0 and fh > 0) else 1.0

    def _events_for(peaks, attacker_key, target_key,
                    attacker_bbox_key, target_bbox_key):
        out = []
        for pi in peaks:
            f = frames[pi]
            att    = f.get(attacker_key)
            tgt    = f.get(target_key)
            att_bb = f.get(attacker_bbox_key)
            tgt_bb = f.get(target_bbox_key)
            verdict, wrist = _verdict_at_peak(att, tgt,
                                              target_bbox=tgt_bb,
                                              attacker_bbox=att_bb,
                                              aspect=aspect)
            head_box = _head_zone(tgt, tgt_bb, aspect=aspect) if tgt else None
            stom_box = _stomach_zone(tgt, aspect=aspect)       if tgt else None
            out.append({
                "fi":       pi,
                "t":        float(f.get("time_s", pi / fps_pose)),
                "verdict":  verdict,
                "wrist":    list(wrist) if wrist is not None else None,
                "head_box": list(head_box) if head_box else None,
                "stom_box": list(stom_box) if stom_box else None,
            })
        return out

    my_events = _events_for(my_peaks, "my_kps", "op_kps", "my_bbox", "op_bbox")
    op_events = _events_for(op_peaks, "op_kps", "my_kps", "op_bbox", "my_bbox")

    def _tally(events):
        landed_head = sum(1 for e in events if e["verdict"] == "landed_head")
        landed_body = sum(1 for e in events if e["verdict"] == "landed_body")
        missed      = sum(1 for e in events if e["verdict"] == "missed")
        unknown     = sum(1 for e in events if e["verdict"] == "unknown")
        decisive    = landed_head + landed_body + missed     # excludes unknown
        accuracy    = round(100 * (landed_head + landed_body) / max(decisive, 1)) \
                      if decisive > 0 else None
        return landed_head, landed_body, missed, unknown, accuracy

    (my_lh, my_lb, my_miss, my_unk, my_acc) = _tally(my_events)
    (op_lh, op_lb, op_miss, op_unk, op_acc) = _tally(op_events)

    # ── State-share (Phase 2 "clip makeup") ─────────────────────────────────
    # Percentage of clip time in each interaction state. Useful for:
    #   • surfaces on the UI as top-line context ("55% open, 30% close...")
    #   • coaches / highlight generation can key off "stuck in clinch"
    #   • future defense refinement can ask "was fighter X trapped?"
    state_per_sec = [STATE_OPEN] * duration_s
    state_counts  = {STATE_OPEN: 0, STATE_CLOSE: 0,
                     STATE_CLINCH: 0, STATE_OCCLUSION: 0}
    for s in range(duration_s):
        idxs = by_sec.get(s, [])
        if not idxs:
            continue
        # Mode within the second — majority wins.
        counts = {}
        for i in idxs:
            counts[state[i]] = counts.get(state[i], 0) + 1
        state_per_sec[s] = max(counts, key=counts.get)
    for s in state_per_sec:
        state_counts[s] = state_counts.get(s, 0) + 1
    total = max(1, sum(state_counts.values()))
    state_share = {k: round(100 * v / total) for k, v in state_counts.items()}

    return {
        "ok":              True,
        "duration_s":      duration_s,
        "tier":            tier,
        "tier_weight":     tier_weight,
        "my_visibility":   my_vis,
        "op_visibility":   op_vis,
        "my_ring":         my_ring,
        "op_ring":         op_ring,
        "my_aggression":   my_aggr,
        "op_aggression":   op_aggr,
        "my_movement":     my_movement,
        "op_movement":     op_movement,
        "my_volume":       my_volume,
        "op_volume":       op_volume,
        "my_defense":      my_def,
        "op_defense":      op_def,
        "my_head_movement": my_head_mov,
        "op_head_movement": op_head_mov,
        "my_guard":        my_guard,
        "op_guard":        op_guard,
        "my_punches":      my_punches_total,
        "op_punches":      op_punches_total,

        # State classifier (Phase 2). Per-second label + overall share.
        # Fed to the Lab's state swimlane and state-breakdown stat.
        "state_share":     state_share,      # {"open": 55, "close": 30, ...}

        # ── Landed-hit gating (Phase 1, DEPRECATED for headline metrics) ──
        # Kept in the result for debugging/research but NOT surfaced in the
        # UI or harness. The 2D glove-in-zone heuristic couldn't cleanly
        # separate contact from close-range proximity; Phase 2's state-
        # aware Aggression captures close-range credit without needing a
        # landed/missed call. Uncomment UI wiring + harness columns to
        # revive if a future approach (depth estimation, 3D reconstruction,
        # wrist-velocity-profile matching) gives reliable per-hit verdicts.
        "my_landed_head":  my_lh,
        "my_landed_body":  my_lb,
        "my_missed":       my_miss,
        "my_unknown":      my_unk,
        "my_accuracy":     my_acc,
        "op_landed_head":  op_lh,
        "op_landed_body":  op_lb,
        "op_missed":       op_miss,
        "op_unknown":      op_unk,
        "op_accuracy":     op_acc,
        "my_landed_events": my_events,
        "op_landed_events": op_events,

        "series": {
            "my_ring":       my_ring_sec,
            "op_ring":       op_ring_sec,
            "my_aggression": my_agg_sec,
            "op_aggression": op_agg_sec,
            "my_movement":   my_movement_sec,
            "op_movement":   op_movement_sec,
            "my_volume":     my_volume_sec,
            "op_volume":     op_volume_sec,
            "my_defense":    my_def_sec,
            "op_defense":    op_def_sec,
            # Defense pillar breakdowns (0-100 absolute quality; null = no
            # signal to grade that second because fighter wasn't attacked).
            "my_def_distance": my_def_dist_sec,
            "my_def_head":     my_def_head_sec,
            "my_def_guard":    my_def_guard_sec,
            "op_def_distance": op_def_dist_sec,
            "op_def_head":     op_def_head_sec,
            "op_def_guard":    op_def_guard_sec,
            "my_head_movement": my_head_mov_sec,
            "op_head_movement": op_head_mov_sec,
            "my_guard":      my_guard_sec,
            "op_guard":      op_guard_sec,
            "state":         state_per_sec,
        },
    }


def compute_and_save(session_dir: Path) -> dict:
    result = compute(session_dir)
    (session_dir / "arena_metrics.json").write_text(
        json.dumps(result, separators=(",", ":")))
    return result
