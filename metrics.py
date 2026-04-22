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
PUNCH_SCORE_MIN     = 0.050   # peak-score threshold. Calibrated against user-
                              # provided ground-truth counts across 6 clips —
                              # this value minimizes total absolute error with
                              # torso-normalized velocity (see _punch_score).
PUNCH_PEAK_SEP_S    = 0.18    # min time between detected peaks (fast 1-2
                              # combos sit around 0.2 s apart)
RANGE_TRIGGER_SCALE = 2.6     # "in range" = opponent-distance <= this × scale-px
ADVANCE_EPS_SCALE   = 0.005   # min per-frame closing speed (in scales) to count

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


def _punch_score(kps, prev_kps, toward_dir):
    """
    Per-frame punch-likelihood score for one fighter.

        score = max over both wrists of
                  extension_ratio × wrist_velocity × direction_factor

    • extension_ratio: distance(wrist, shoulder-center) / torso_height.
    • wrist_velocity: magnitude of body-relative wrist displacement since
      the previous frame (shoulder motion subtracted so the fighter's own
      footwork doesn't register as wrist velocity). Kept in normalized
      [0,1] frame coords — NOT divided by torso_height. Empirically this
      gives a more consistent distribution across clips than torso-
      normalization, which amplified inter-clip score variance 3× by
      dividing by a small, noisy quantity.
    • direction_factor: how aligned the wrist velocity is with the vector
      from this fighter to the opponent (1 = straight toward, 0 = straight
      away, 0.5 = perpendicular or unknown). Defensive/retraction motion
      has direction_factor near 0, driving the score low.

    Returns 0.0 when required keypoints are missing.
    """
    sc = _shoulder_center(kps)
    th = _torso_height(kps)
    if sc is None or th is None or th <= 1e-6:
        return 0.0
    prev_sc = _shoulder_center(prev_kps) if prev_kps else None

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
        # Body-relative wrist displacement (subtract shoulder motion).
        cur_rel  = (w[0]  - sc[0],      w[1]  - sc[1])
        prev_rel = (pw[0] - prev_sc[0], pw[1] - prev_sc[1])
        vx = cur_rel[0] - prev_rel[0]
        vy = cur_rel[1] - prev_rel[1]
        vmag = math.hypot(vx, vy)
        if vmag < 1e-6:
            continue
        # Extension fraction (torso-normalized).
        ext = _dist(w, sc) / th
        # Directional factor: cosine of angle between velocity and opponent
        # direction, remapped to [0, 1].
        if tmag < 1e-6:
            dir_factor = 0.5
        else:
            cos_sim = (vx * tx + vy * ty) / (vmag * tmag)
            dir_factor = max(0.0, min(1.0, (cos_sim + 1.0) / 2.0))
        score = ext * vmag * dir_factor
        if score > best:
            best = score
    return best


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
        my_punch_score[i] = _punch_score(my_kps, prev_my, me_to_op)
        op_punch_score[i] = _punch_score(op_kps, prev_op, op_to_me)
        # Convert normalized keypoints to image pixels (head position used
        # only as a relative signal, not in arena coords).
        def _hp(kps):
            h = _head_pos(kps)
            return None if h is None else (h[0] * fw, h[1] * fh)
        my_head_img[i] = _hp(my_kps)
        op_head_img[i] = _hp(op_kps)

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

    # ── Metric 2: Effective Aggression (mixed) ─────────────────────────────
    # Per second: (% frames with closing×punching) + partial credit for
    # closing alone. Then normalize so ME + OP ≈ 100 at each second.
    my_agg_sec = [50] * duration_s
    op_agg_sec = [50] * duration_s
    for s in range(duration_s):
        idxs = by_sec.get(s, [])
        if not idxs:
            continue
        me_score = 0.0
        op_score = 0.0
        for i in idxs:
            # Arena-dependent: closing contributes weighted by tier.
            if my_closing[i] > ADVANCE_EPS_SCALE:
                me_score += 0.6 * tier_weight
            if op_closing[i] > ADVANCE_EPS_SCALE:
                op_score += 0.6 * tier_weight
            # Body-relative: punching contributes at full weight.
            if my_punch[i]:
                me_score += 1.0 * (1.4 if my_closing[i] > ADVANCE_EPS_SCALE else 1.0)
            if op_punch[i]:
                op_score += 1.0 * (1.4 if op_closing[i] > ADVANCE_EPS_SCALE else 1.0)
        tot = me_score + op_score
        if tot > 1e-6:
            my_agg_sec[s] = round(100 * me_score / tot)
            op_agg_sec[s] = 100 - my_agg_sec[s]

    # ── Metric 3: Work Rate (mixed) ────────────────────────────────────────
    # Path length per second (in scales) + punches per second. Both sides
    # normalized against per-second max so the score is relative, not absolute.
    my_path = [0.0] * n
    op_path = [0.0] * n
    for i in range(1, n):
        if i < len(my_traj_ref) and my_traj_ref[i] is not None and my_traj_ref[i - 1] is not None:
            my_path[i] = _dist(my_traj_ref[i], my_traj_ref[i - 1]) / median_scale
        if i < len(op_traj_ref) and op_traj_ref[i] is not None and op_traj_ref[i - 1] is not None:
            op_path[i] = _dist(op_traj_ref[i], op_traj_ref[i - 1]) / median_scale

    my_work_sec = [50] * duration_s
    op_work_sec = [50] * duration_s
    for s in range(duration_s):
        idxs = by_sec.get(s, [])
        if not idxs:
            continue
        # Arena component: path length in scales (tier-weighted).
        me_mv = sum(my_path[i] for i in idxs) * tier_weight
        op_mv = sum(op_path[i] for i in idxs) * tier_weight
        # Body component: punch count.
        me_punches = sum(1 for i in idxs if my_punch[i])
        op_punches = sum(1 for i in idxs if op_punch[i])
        me = me_mv + me_punches
        op = op_mv + op_punches
        tot = me + op
        if tot > 1e-6:
            my_work_sec[s] = round(100 * me / tot)
            op_work_sec[s] = 100 - my_work_sec[s]

    # ── Metric 4: Defense (mixed) ──────────────────────────────────────────
    # Arena component: fraction of time opponent was OUT of effective range.
    # Body component: head-position variance (proxy for head movement).
    my_def_sec = [50] * duration_s
    op_def_sec = [50] * duration_s
    for s in range(duration_s):
        idxs = by_sec.get(s, [])
        if not idxs:
            continue
        out_of_range_me = sum(1 for i in idxs
                              if my_to_op[i] is not None and my_to_op[i] > RANGE_TRIGGER_SCALE)
        out_of_range_op = out_of_range_me   # symmetric distance
        # Head-mobility variance within the second (body-relative).
        my_heads = [my_head_img[i] for i in idxs if my_head_img[i] is not None]
        op_heads = [op_head_img[i] for i in idxs if op_head_img[i] is not None]
        def _var(pts):
            if len(pts) < 2:
                return 0.0
            arr = np.array(pts, dtype=np.float32)
            return float(arr.std(axis=0).mean()) / max(median_scale, 1.0)
        me_mob = _var(my_heads)
        op_mob = _var(op_heads)
        # Score: out-of-range contributes when arena tier is good, mobility
        # contributes at full weight regardless.
        me = (out_of_range_me / max(len(idxs), 1)) * tier_weight + me_mob
        op = (out_of_range_op / max(len(idxs), 1)) * tier_weight + op_mob
        tot = me + op
        if tot > 1e-6:
            my_def_sec[s] = round(100 * me / tot)
            op_def_sec[s] = 100 - my_def_sec[s]

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
    my_work      = _apply_fighter_visibility(_avg(my_work_sec), my_vis)
    op_work      = _apply_fighter_visibility(_avg(op_work_sec), op_vis)
    my_def       = _apply_fighter_visibility(_avg(my_def_sec), my_vis)
    op_def       = _apply_fighter_visibility(_avg(op_def_sec), op_vis)
    my_guard     = _avg(my_guard_sec)   # body-relative, never downweighted
    op_guard     = _avg(op_guard_sec)

    # Peak-detect on the directional score — one peak per real punch.
    my_peaks = _detect_punch_peaks(my_punch_score, fps_pose)
    op_peaks = _detect_punch_peaks(op_punch_score, fps_pose)
    my_punches_total = len(my_peaks)
    op_punches_total = len(op_peaks)

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
        "my_work_rate":    my_work,
        "op_work_rate":    op_work,
        "my_defense":      my_def,
        "op_defense":      op_def,
        "my_guard":        my_guard,
        "op_guard":        op_guard,
        "my_punches":      my_punches_total,
        "op_punches":      op_punches_total,
        "series": {
            "my_ring":       my_ring_sec,
            "op_ring":       op_ring_sec,
            "my_aggression": my_agg_sec,
            "op_aggression": op_agg_sec,
            "my_work_rate":  my_work_sec,
            "op_work_rate":  op_work_sec,
            "my_defense":    my_def_sec,
            "op_defense":    op_def_sec,
            "my_guard":      my_guard_sec,
            "op_guard":      op_guard_sec,
        },
    }


def compute_and_save(session_dir: Path) -> dict:
    result = compute(session_dir)
    (session_dir / "arena_metrics.json").write_text(
        json.dumps(result, separators=(",", ":")))
    return result
