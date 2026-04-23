"""
Boxing Sparring Analyser — Flask backend  v4
Tracking: Kalman filter + IoU matching + coasting + appearance re-ID
          Optional SAM2 upgrade when sam2_checkpoints/ is present.
"""
import os
import time
import re
import json
import uuid
import math
import bisect
import base64
import threading
import subprocess
import tempfile
import shutil
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import (
    Flask, request, jsonify, render_template,
    send_file, redirect, url_for, Response,
)

import pipeline_cache
import diagnostics
import screenshots
import arena_detector
import metrics as arena_metrics_mod

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024 * 1024

BASE = Path(__file__).parent
SESS = BASE / "sessions_data"
SESS.mkdir(exist_ok=True)

# ─── Session helpers ──────────────────────────────────────────────────────────

def sess_dir(sid):  return SESS / sid
def meta_file(sid): return sess_dir(sid) / "meta.json"

def read_meta(sid):
    p = meta_file(sid)
    return json.loads(p.read_text()) if p.exists() else None

def write_meta(sid, meta):
    sess_dir(sid).mkdir(parents=True, exist_ok=True)
    meta_file(sid).write_text(json.dumps(meta, indent=2))

def list_sessions():
    out = []
    if not SESS.exists():
        return out
    for d in SESS.iterdir():
        if d.is_dir():
            m = read_meta(d.name)
            if m:
                out.append(m)
    # Newest upload first
    out.sort(key=lambda m: m.get("upload_time", ""), reverse=True)
    return out


# ─── YOLO (lazy-loaded) ───────────────────────────────────────────────────────

_model = None
_lock  = threading.Lock()

def get_model():
    global _model
    with _lock:
        if _model is None:
            from ultralytics import YOLO
            _model = YOLO("yolov8s-pose.pt")   # small model — better accuracy in crowds
    return _model


# ─── COCO skeleton + colour constants ────────────────────────────────────────

COCO_SKELETON = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]

PICK_PALETTE_BGR = [(0,140,255),(255,80,0),(0,200,0),(0,0,210)]
PICK_PALETTE_CSS = ["#ff8c00","#0050ff","#00c800","#d20000"]

C_NEUTRAL  = (0,  200,   0)
C_ATTACK   = (0,    0, 220)
C_RETREAT  = (220,  50,   0)
C_SKELETON = (150, 150, 150)
C_KP       = (80,   80, 200)
C_GUARD_UP = (255, 255, 255)
C_GUARD_DN = (0,  215, 255)
C_COAST    = (160, 160, 160)   # grey — coasting, no live detection


# ─── Safe metadata helper ─────────────────────────────────────────────────────

def _safe(val, fallback):
    """Guard OpenCV cap.get() values that can return NaN / Inf / 0."""
    try:
        v = float(val)
        return fallback if (math.isnan(v) or math.isinf(v) or v <= 0) else v
    except Exception:
        return fallback


# ─── Geometry helpers ─────────────────────────────────────────────────────────

def _cx_cy(box):
    return ((box[0]+box[2])/2.0, (box[1]+box[3])/2.0)

def _dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def _kp(kps, i, conf_thresh=0.40):
    """Return (x, y) for keypoint i if confidence is sufficient, else None.
    YOLO pose tensors are shaped [N, 17, 3] with columns (x, y, conf).
    Filtering by confidence stops gloves/occlusions being mistaken for
    shoulders — a common failure mode in boxing that corrupts the torso crop.
    """
    if kps is None or i >= len(kps): return None
    kp = kps[i]
    x, y = float(kp[0]), float(kp[1])
    if x <= 0 and y <= 0: return None          # undetected
    if len(kp) >= 3 and float(kp[2]) < conf_thresh:
        return None                             # low-confidence detection
    return (x, y)

def _ring_score(box, fw, fh):
    """
    Score a detection as 'likely an in-ring boxer'.
    Rewards large box area + centrality along the primary axis;
    penalises outer 15% edges.  Portrait-aware: uses y-centrality when fh > fw*1.2.
    """
    area = (box[2]-box[0]) * (box[3]-box[1])
    if fh > fw * 1.2:                              # portrait
        frac     = (box[1]+box[3]) / (2.0*fh)
    else:                                          # landscape / square
        frac     = (box[0]+box[2]) / (2.0*fw)
    edge_pen = max(0.05, 1.0 - max(0.0, abs(frac-0.5) - 0.25) * 5.0)
    return area * edge_pen


# ─── Appearance helpers (D: re-ID) ────────────────────────────────────────────

def _hsv_hist(roi):
    """
    Compute a 3-channel HSV histogram (H×S×V) for a BGR ROI and return it
    as a normalised flat float list.

    Bin layout: H=12 bins (30° each), S=8 bins, V=6 bins → 576 values.
    Adding V is critical for achromatic colours (grey, white, black):
    H+S alone maps all grey shades to the same low-saturation bin regardless
    of brightness, making a dark-grey vest and a white T-shirt indistinguishable.
    V separates them clearly and costs almost nothing extra at runtime.
    """
    hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None,
                        [12, 8, 6],
                        [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.flatten().tolist()


def _extract_hist(frame, box):
    """
    HSV histogram of the boxer's torso region (20–72 % of bbox height).
    Returns a flat float list (JSON-serialisable) or None if the ROI is empty.
    """
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    bh  = y2 - y1
    ty1 = max(0, int(y1 + bh * 0.20))
    ty2 = max(ty1 + 1, int(y1 + bh * 0.72))
    roi = frame[ty1:ty2, x1:x2]
    if roi.size == 0:
        return None
    return _hsv_hist(roi)


def _extract_hist_body(frame, box, kps):
    """
    Keypoint-guided torso histogram.

    Uses shoulder (5,6) + hip (11,12) keypoints to crop the body region,
    which removes background / ring-rope pixels that bleed inside the bbox.

    Safety rules that prevent the boxing-glove problem:
      1. Only keypoints with confidence ≥ 0.40 are accepted (see _kp).
      2. The resulting crop is clamped to the bounding box — keypoints that
         land outside the box (e.g. a shoulder point mis-projected onto a
         raised glove above the head) are therefore silently discarded.
      3. The clamped crop must cover ≥ 30 % of the bbox width and
         ≥ 20 % of the bbox height to be trusted.  Crops that fail this
         check fall back to the reliable 20–72 % height strip.

    Falls back to _extract_hist (bbox strip) when < 3 confident keypoints
    are inside the box or the clamped crop is too small.
    """
    bx1, by1, bx2, by2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    bw = bx2 - bx1
    bh = by2 - by1

    if kps is not None and bw > 0 and bh > 0:
        # Only keep keypoints that lie inside (or just inside) the bbox so
        # that mis-projected glove/wrist points are automatically ignored.
        body_pts = []
        for ki in (5, 6, 11, 12):
            p = _kp(kps, ki)
            if p is None: continue
            px, py = p
            if bx1 - bw*0.10 <= px <= bx2 + bw*0.10 and by1 <= py <= by2 + bh*0.10:
                body_pts.append(p)

        if len(body_pts) >= 3:
            xs     = [p[0] for p in body_pts]
            ys     = [p[1] for p in body_pts]
            w_span = max(xs) - min(xs)
            h_span = max(ys) - min(ys)
            pad_x  = max(w_span * 0.28, 15)

            # Clamp entirely within the bounding box
            x1 = max(bx1, int(min(xs) - pad_x))
            x2 = min(bx2, int(max(xs) + pad_x))
            y1 = max(by1, int(min(ys) - h_span * 0.12))
            y2 = min(by2, int(max(ys) + h_span * 0.40))

            # Sanity check: crop must be a meaningful portion of the bbox
            if (x2 - x1 >= bw * 0.30 and y2 - y1 >= bh * 0.20
                    and x2 > x1 + 10 and y2 > y1 + 10):
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    return _hsv_hist(roi)

    return _extract_hist(frame, box)


def _hist_sim(h1, h2):
    """
    Appearance similarity in [0, 1].  1 = identical clothing colours, 0 = completely different.
    Uses Bhattacharyya distance (lighting-robust via H + S channels only).
    """
    if h1 is None or h2 is None:
        return 0.0
    a = np.array(h1, dtype=np.float32)
    b = np.array(h2, dtype=np.float32)
    dist = cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA)
    return float(max(0.0, 1.0 - dist))


# ─── IoU helper (B: IoU matching) ────────────────────────────────────────────

def _box_iou(b1, b2):
    """Intersection-over-Union for two [x1,y1,x2,y2] boxes."""
    xi1 = max(b1[0], b2[0]); yi1 = max(b1[1], b2[1])
    xi2 = min(b1[2], b2[2]); yi2 = min(b1[3], b2[3])
    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    if inter == 0:
        return 0.0
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


# ─── SAM2 optional tracker ───────────────────────────────────────────────────
# When sam2_venv/ and sam2_checkpoints/sam2.1_hiera_tiny.pt both exist, the
# analysis pipeline offloads tracking to SAM2 via subprocess.  The Kalman
# tracker is kept as the fallback if SAM2 is not installed.

_SAM2_SCRIPT = BASE / "sam2_tracker.py"
_SAM2_VENV   = BASE / "sam2_venv" / "bin" / "python"
_SAM2_CKPT   = BASE / "sam2_checkpoints" / "sam2.1_hiera_tiny.pt"
_SAM2_SMALL  = BASE / "sam2_checkpoints" / "sam2.1_hiera_small.pt"

def sam2_available() -> bool:
    """True if the SAM2 venv and at least the tiny checkpoint both exist."""
    return _SAM2_VENV.exists() and (_SAM2_CKPT.exists() or _SAM2_SMALL.exists())


def run_sam2_tracker(video_path: str, my_pt, op_pt):
    """
    Call sam2_tracker.py in the SAM2 venv.  Returns the parsed JSON dict on
    success, or None if SAM2 is not available / the call fails.

    my_pt / op_pt: (x, y) centre of each boxer on frame 0 (from picker).

    Returns a dict with keys:
      ok, model, device, frame_count, fps_effective, boxes
    or None on any failure.  Also returns a "diag" key with human-readable
    status for meta storage.
    """
    if not sam2_available():
        return None

    # TODO(production): restore "small" if _SAM2_SMALL.exists() else "tiny"
    model = "tiny"
    out_path = Path(tempfile.mktemp(suffix=".json"))
    try:
        cmd = [
            str(_SAM2_VENV), str(_SAM2_SCRIPT),
            "--video",   video_path,
            "--my_pt",   f"{my_pt[0]},{my_pt[1]}",
            "--op_pt",   f"{op_pt[0]},{op_pt[1]}",
            "--model",   model,
            "--max_dim", "1024",   # downscale to prevent OOM on high-res videos
            "--out",     str(out_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if not out_path.exists():
            # Subprocess ran but didn't write output — capture stderr for diagnostics
            stderr_snippet = (proc.stderr or "")[-500:]
            print(f"[SAM2] no output file; stderr: {stderr_snippet}")
            return None
        result = json.loads(out_path.read_text())
        if not result.get("ok"):
            print(f"[SAM2] tracker returned error: {result.get('error')}")
            return None
        return result
    except subprocess.TimeoutExpired:
        print("[SAM2] subprocess timed out (600s)")
        return None
    except Exception as e:
        print(f"[SAM2] exception: {e}")
        return None
    finally:
        out_path.unlink(missing_ok=True)


# ─── Kalman box tracker (A: Kalman filter) ───────────────────────────────────

class KalmanBoxTracker:
    """
    Constant-velocity Kalman filter tracking a single boxer's bounding box.
    Pure numpy — no extra dependencies.

    State [8]:       cx, cy, w, h,  vx, vy, vw, vh
    Observation [4]: cx, cy, w, h
    """
    # Transition matrix: position += velocity each frame
    _F = np.array([
        [1,0,0,0, 1,0,0,0],
        [0,1,0,0, 0,1,0,0],
        [0,0,1,0, 0,0,1,0],
        [0,0,0,1, 0,0,0,1],
        [0,0,0,0, 1,0,0,0],
        [0,0,0,0, 0,1,0,0],
        [0,0,0,0, 0,0,1,0],
        [0,0,0,0, 0,0,0,1],
    ], dtype=np.float64)
    _H = np.eye(4, 8, dtype=np.float64)        # observe [cx, cy, w, h]
    _R = np.diag([25., 25., 100., 100.])        # observation noise  (px²)
    _Q = np.diag([9.,  9.,  4.,  4.,            # process noise
                  16., 16., 4.,  4.])

    def __init__(self, box, hist=None):
        cx = (box[0]+box[2]) / 2.0
        cy = (box[1]+box[3]) / 2.0
        w  = float(max(box[2]-box[0], 10))
        h  = float(max(box[3]-box[1], 10))
        self.x = np.array([cx, cy, w, h, 0., 0., 0., 0.], dtype=np.float64)
        self.P = np.diag([100., 100., 100., 100., 500., 500., 100., 100.])
        self.coast = 0      # frames without a matched detection
        self.hist  = hist   # flattened HSV histogram for appearance re-ID

    def predict(self):
        """Propagate state one frame forward. Returns predicted [x1,y1,x2,y2]."""
        self.x = self._F @ self.x
        self.P = self._F @ self.P @ self._F.T + self._Q
        return self.to_box()

    def update(self, box):
        """Fuse a matched detection into the state and reset coast counter."""
        cx = (box[0]+box[2]) / 2.0
        cy = (box[1]+box[3]) / 2.0
        w  = float(max(box[2]-box[0], 10))
        h  = float(max(box[3]-box[1], 10))
        z  = np.array([cx, cy, w, h], dtype=np.float64)
        S  = self._H @ self.P @ self._H.T + self._R
        K  = self.P @ self._H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self._H @ self.x)
        self.P = (np.eye(8) - K @ self._H) @ self.P
        self.coast = 0

    def miss(self):
        """No detection matched this frame — advance coast counter.
        After 3 misses, progressively damp the stored velocity so the
        Kalman prediction stays anchored near the last confirmed position
        instead of extrapolating the boxer off-screen into crowd/corner areas.
        """
        self.coast += 1
        if self.coast >= 3:
            self.x[4:8] *= 0.85   # decay vx, vy, vw, vh toward zero

    def to_box(self):
        cx, cy, w, h = self.x[:4]
        w = max(w, 10.); h = max(h, 10.)
        return [int(cx-w/2), int(cy-h/2), int(cx+w/2), int(cy+h/2)]


# ─── Analysis tuning constants ───────────────────────────────────────────────

ANALYSIS_STRIDE   = 3      # run YOLO every Nth frame; Kalman fills the rest
PUNCH_THRESH_FRAC = 0.21   # wrist displacement > 21% of boxer height per stride = punch
                           # (equivalent to the old 7%-of-frame-width threshold, since
                           # typical boxer height ≈ 33% of frame width: 0.07/0.33 ≈ 0.21)
CONF_THRESH       = 0.35   # minimum tracker confidence to count toward metrics


# ─── Punch / rope helpers ─────────────────────────────────────────────────────

def _wrist_speed(kps, prev_kps):
    """Max wrist displacement relative to the boxer's own body centre.

    Subtracting shoulder displacement removes footwork and body sway — a
    boxer who shuffles laterally moves their wrists with them, which should
    not register as punch activity.  Only genuine wrist extension (punches,
    parries) contributes to the score.
    """
    if kps is None or prev_kps is None:
        return 0.0

    # Body reference: average of available shoulder keypoints
    def _body_centre(k):
        pts = [_kp(k, i) for i in (5, 6) if _kp(k, i) is not None]
        if not pts: return None
        return (sum(p[0] for p in pts) / len(pts),
                sum(p[1] for p in pts) / len(pts))

    bc_curr = _body_centre(kps)
    bc_prev = _body_centre(prev_kps)
    body_dx = body_dy = 0.0
    if bc_curr and bc_prev:
        body_dx = bc_curr[0] - bc_prev[0]
        body_dy = bc_curr[1] - bc_prev[1]

    best = 0.0
    for ki in (9, 10):
        w  = _kp(kps,      ki)
        pw = _kp(prev_kps, ki)
        if w and pw:
            rel_dx = (w[0] - pw[0]) - body_dx
            rel_dy = (w[1] - pw[1]) - body_dy
            best = max(best, math.hypot(rel_dx, rel_dy))
    return best


def _wrist_speed_lr(kps, prev_kps):
    """Body-relative speed of each wrist separately — returns (left_speed, right_speed).

    Uses the same shoulder-subtraction as _wrist_speed so body sway is removed.
    Keypoint 9 = left wrist, 10 = right wrist (from YOLO/COCO convention).
    """
    if kps is None or prev_kps is None:
        return 0.0, 0.0

    def _body_centre(k):
        pts = [_kp(k, i) for i in (5, 6) if _kp(k, i) is not None]
        if not pts: return None
        return (sum(p[0] for p in pts) / len(pts),
                sum(p[1] for p in pts) / len(pts))

    bc_curr = _body_centre(kps)
    bc_prev = _body_centre(prev_kps)
    body_dx = body_dy = 0.0
    if bc_curr and bc_prev:
        body_dx = bc_curr[0] - bc_prev[0]
        body_dy = bc_curr[1] - bc_prev[1]

    speeds = []
    for ki in (9, 10):   # 9 = left wrist, 10 = right wrist
        w  = _kp(kps,      ki)
        pw = _kp(prev_kps, ki)
        if w and pw:
            rel_dx = (w[0] - pw[0]) - body_dx
            rel_dy = (w[1] - pw[1]) - body_dy
            speeds.append(math.hypot(rel_dx, rel_dy))
        else:
            speeds.append(0.0)
    return speeds[0], speeds[1]   # (left_speed, right_speed)


def _is_on_ropes(box, fw, fh=None):
    """True when a boxer's centre is within the outer 18% of the primary axis.
    Landscape: horizontal edges (left/right ropes).
    Portrait:  vertical edges (top/bottom — camera shows depth of ring)."""
    if fh is not None and fh > fw * 1.2:   # portrait
        cy = (box[1] + box[3]) / 2.0
        return cy < fh * 0.18 or cy > fh * 0.82
    cx = (box[0] + box[2]) / 2.0
    return cx < fw * 0.18 or cx > fw * 0.82


def _size_score(cand_box, expected_h):
    """
    Multiplicative penalty for candidates that are much smaller than the expected
    boxer height.  Background people are farther from the camera → shorter bbox.
    Returns 1.0 when expected_h is unknown or the size is plausible.
    """
    if expected_h is None or expected_h <= 0:
        return 1.0
    ratio = (cand_box[3] - cand_box[1]) / expected_h
    if ratio < 0.55:            # tiny — almost certainly background
        return max(0.05, ratio / 0.55)
    if ratio > 1.80:            # huge — probably merged / bad detection
        return max(0.30, 1.80 / ratio)
    return 1.0


# ─── Picker frame sampling ───────────────────────────────────────────────────

PICKER_SAMPLE_FRACS = [0.10, 0.22, 0.35, 0.48, 0.61, 0.75]

def _picker_score(raw, fw, fh):
    """
    Score a candidate frame for picker usefulness.
    Best frame = two well-separated, large, central detections with minimal overlap.
    Returns 0 if fewer than 2 detections are present.
    """
    if len(raw) < 2:
        return 0.0
    top2    = sorted(raw, key=lambda b: _ring_score(b, fw, fh), reverse=True)[:2]
    b1, b2  = top2
    sep     = _dist(_cx_cy(b1), _cx_cy(b2)) / max(fw, fh)   # normalised separation
    overlap = _box_iou(b1, b2)                               # penalise clinch / overlap
    quality = (_ring_score(b1, fw, fh) + _ring_score(b2, fw, fh)) / (fw * fh * 2)
    return sep * (1.0 - overlap) * quality


# ─── Tracking constants ───────────────────────────────────────────────────────

MAX_COAST  = 60     # frames to keep coasting on prediction alone (~2 s at 30 fps)
REID_AFTER = 15     # start trying histogram re-ID after this many coast frames
REID_MIN   = 0.65   # minimum Bhattacharyya similarity to accept a re-ID match
                    # (3-channel H×S×V histogram is more discriminative than the
                    # old 2-channel H×S, so a slightly lower threshold still
                    # rejects mismatches while achromatic grey can pass)
MIN_SCORE  = 0.04   # minimum IoU-or-distance score to accept any assignment


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", sessions=list_sessions())


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("video")
    if not f:
        return "No file received", 400

    sid  = uuid.uuid4().hex[:8]
    sess_dir(sid).mkdir(parents=True, exist_ok=True)

    ext   = Path(f.filename).suffix.lower() or ".mp4"
    vpath = sess_dir(sid) / f"original{ext}"
    f.save(str(vpath))

    # Sample multiple frames to find one where both boxers are clearly visible
    cap     = cv2.VideoCapture(str(vpath))
    fps_r   = _safe(cap.get(cv2.CAP_PROP_FPS),           30.0)
    total_f = int(_safe(cap.get(cv2.CAP_PROP_FRAME_COUNT), 0))

    model    = get_model()
    min_area_frac = 0.01   # minimum box area as fraction of frame area

    def _detect_at(frac):
        """Read frame at frac ∈ [0,1], run YOLO, return (frame, boxes, kps_list, fw, fh) or None."""
        target = max(0, int(total_f * frac)) if total_f > 0 else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frm = cap.read()
        if not ok:
            return None
        fh_, fw_ = frm.shape[:2]
        min_a = fw_ * fh_ * min_area_frac
        res   = model(frm, verbose=False)[0]
        boxes = []
        kps_list = []
        if res.boxes is not None:
            for idx, box in enumerate(res.boxes):
                if int(box.cls[0]) == 0:
                    x1,y1,x2,y2 = [int(v) for v in box.xyxy[0].tolist()]
                    if (x2-x1)*(y2-y1) >= min_a:
                        boxes.append([x1,y1,x2,y2])
                        kps = (res.keypoints.data[idx].cpu().numpy()
                               if res.keypoints is not None else None)
                        kps_list.append(kps)
        # Sort by ring score; keep paired order
        scored = sorted(zip(boxes, kps_list),
                        key=lambda p: _ring_score(p[0], fw_, fh_), reverse=True)[:4]
        boxes    = [p[0] for p in scored]
        kps_list = [p[1] for p in scored]
        return frm, boxes, kps_list, fw_, fh_

    # Try each candidate fraction; score by two-boxer separation
    candidates = []
    first_frame = None
    for frac in PICKER_SAMPLE_FRACS:
        result = _detect_at(frac)
        if result is None:
            continue
        frm, boxes, kps_list, fw_, fh_ = result
        if first_frame is None:
            first_frame = frm
        score = _picker_score(boxes, fw_, fh_)
        # Sort left→right while keeping boxes and keypoints aligned
        paired = sorted(zip(boxes, kps_list), key=lambda p: (p[0][0]+p[0][2])//2)
        sorted_boxes    = [p[0] for p in paired]
        sorted_kps_list = [p[1] for p in paired]
        # Use the same histogram method as tracking so initial reference
        # histograms are directly comparable to per-frame tracking histograms.
        candidates.append(dict(
            frame_frac    = frac,
            detected_boxes= sorted_boxes,
            box_hists     = [_extract_hist_body(frm, b, k)
                             for b, k in zip(sorted_boxes, sorted_kps_list)],
            score         = score,
        ))
    cap.release()

    # Sort candidates: prefer frames with 2+ well-separated boxers
    candidates.sort(key=lambda c: c["score"], reverse=True)

    # If nothing worked at all, fall back to a blank candidate
    if not candidates:
        if first_frame is None:
            return "Cannot read video file", 400
        candidates = [dict(frame_frac=0.30, detected_boxes=[],
                           box_hists=[], score=0.0)]

    best      = candidates[0]
    raw       = best["detected_boxes"]
    box_hists = best["box_hists"]

    # Regenerate preview image for the best candidate frame
    cap2 = cv2.VideoCapture(str(vpath))
    cap2.set(cv2.CAP_PROP_POS_FRAMES,
             max(0, int(total_f * best["frame_frac"])) if total_f > 0 else 0)
    ok, frame = cap2.read()
    cap2.release()
    if not ok:
        return "Cannot read video file", 400

    fh, fw = frame.shape[:2]

    preview = frame.copy()
    for i, (x1,y1,x2,y2) in enumerate(raw):
        col = PICK_PALETTE_BGR[i % len(PICK_PALETTE_BGR)]
        cv2.rectangle(preview, (x1,y1), (x2,y2), col, 3)
        cv2.putText(preview, f"Boxer {i+1}",
                    (x1+4, max(y1-14, 28)),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, col, 2)
    cv2.imwrite(str(sess_dir(sid)/"preview.jpg"), preview)
    cv2.imwrite(str(sess_dir(sid)/"first_frame.jpg"), frame)

    meta = dict(
        id               = sid,
        filename         = f.filename,
        video_ext        = ext,
        upload_time      = datetime.utcnow().isoformat(),
        status           = "picking",
        boxer_index      = None,
        op_index         = None,
        detected_boxes   = raw,
        box_hists        = box_hists,
        palette_css      = PICK_PALETTE_CSS[:len(raw)],
        frame_size       = [fw, fh],
        fps              = fps_r,
        total_frames     = total_f,
        progress         = 0,
        metrics          = None,
        annotated_ready  = False,
        error            = None,
        my_ref_center    = None,
        op_ref_center    = None,
        my_hist          = None,
        op_hist          = None,
        picker_candidates= candidates,
        picker_cand_idx  = 0,
    )
    write_meta(sid, meta)
    return redirect(url_for("pick", sid=sid))


@app.route("/upload_url", methods=["POST"])
def upload_url():
    """Download a video from a URL (YouTube, Vimeo, etc.) using yt-dlp,
    then run the same boxer-detection / picker setup as a normal file upload."""
    data = request.get_json(silent=True) or {}
    url  = (data.get("url") or "").strip()
    if not url:
        return jsonify(error="No URL provided"), 400

    try:
        import yt_dlp
    except ImportError:
        return jsonify(error="yt-dlp is not installed — run: pip install yt-dlp"), 500

    # YouTube currently forces SABR streaming which yt-dlp cannot download.
    # Show a clear message rather than a confusing 403 error.
    _yt_hosts = ("youtube.com", "youtu.be", "www.youtube.com", "m.youtube.com")
    if any(h in url for h in _yt_hosts):
        return jsonify(error=(
            "YouTube downloads are currently blocked by YouTube's SABR streaming "
            "protection and cannot be automated. Please download the video manually "
            "(any browser YouTube downloader extension works) and upload the file instead."
        )), 400

    sid = uuid.uuid4().hex[:8]
    sess_dir(sid).mkdir(parents=True, exist_ok=True)

    # ── Download ──────────────────────────────────────────────────────────────
    # Cap at 1080p; merge video+audio into mp4 so ffmpeg can extract audio later.
    # YouTube requires browser cookies to avoid 403 errors — try Safari first
    # (default macOS browser), then Chrome, then fall back to no cookies.
    out_tmpl = str(sess_dir(sid) / "original.%(ext)s")
    base_opts = {
        "format":              "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
        "outtmpl":             out_tmpl,
        "merge_output_format": "mp4",
        "quiet":               True,
        "no_warnings":         True,
        "noplaylist":          True,
    }

    # YouTube blocks direct downloads in 2024+ unless you use a mobile client API
    # or supply browser cookies.  Try strategies in order — stop at first success.
    # We only attempt Safari (not Chrome/Firefox) to avoid OS keychain permission popups.
    strategies = [
        # 1. iOS client — bypasses most 403s without needing cookies
        {"extractor_args": {"youtube": {"player_client": ["ios"]}}},
        # 2. iOS client + Safari cookies (for age-restricted / members-only)
        {"extractor_args": {"youtube": {"player_client": ["ios"]}},
         "cookiesfrombrowser": ("safari",)},
        # 3. Web client + Safari cookies — fallback for non-YouTube sites
        {"cookiesfrombrowser": ("safari",)},
        # 4. Plain web client — last resort (often 403 on YouTube)
        {},
    ]

    info = None
    last_error = None
    for extra in strategies:
        opts = {**base_opts, **extra}
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
            break   # success
        except Exception as e:
            last_error = e
            for f in sess_dir(sid).glob("original.*"):
                try: f.unlink()
                except Exception: pass

    if info is None:
        return jsonify(error=f"Download failed: {last_error}"), 400

    title    = info.get("title", "video")
    filename = f"{title[:60]}.mp4"

    # yt-dlp always produces .mp4 when merge_output_format=mp4
    vpath = sess_dir(sid) / "original.mp4"
    if not vpath.exists():
        # Fallback: find whatever file was written
        candidates_f = list(sess_dir(sid).glob("original.*"))
        if not candidates_f:
            return jsonify(error="Download produced no output file"), 500
        vpath = candidates_f[0]

    ext = vpath.suffix.lower()

    # ── Boxer detection (identical to /upload) ────────────────────────────────
    cap     = cv2.VideoCapture(str(vpath))
    fps_r   = _safe(cap.get(cv2.CAP_PROP_FPS),           30.0)
    total_f = int(_safe(cap.get(cv2.CAP_PROP_FRAME_COUNT), 0))

    model         = get_model()
    min_area_frac = 0.01

    def _detect_at(frac):
        target = max(0, int(total_f * frac)) if total_f > 0 else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frm = cap.read()
        if not ok:
            return None
        fh_, fw_ = frm.shape[:2]
        min_a = fw_ * fh_ * min_area_frac
        res   = model(frm, verbose=False)[0]
        boxes, kps_list = [], []
        if res.boxes is not None:
            for idx, box in enumerate(res.boxes):
                if int(box.cls[0]) == 0:
                    x1,y1,x2,y2 = [int(v) for v in box.xyxy[0].tolist()]
                    if (x2-x1)*(y2-y1) >= min_a:
                        boxes.append([x1,y1,x2,y2])
                        kps = (res.keypoints.data[idx].cpu().numpy()
                               if res.keypoints is not None else None)
                        kps_list.append(kps)
        scored = sorted(zip(boxes, kps_list),
                        key=lambda p: _ring_score(p[0], fw_, fh_), reverse=True)[:4]
        return frm, [p[0] for p in scored], [p[1] for p in scored], fw_, fh_

    picker_candidates = []
    first_frame = None
    for frac in PICKER_SAMPLE_FRACS:
        result = _detect_at(frac)
        if result is None:
            continue
        frm, boxes, kps_list, fw_, fh_ = result
        if first_frame is None:
            first_frame = frm
        score  = _picker_score(boxes, fw_, fh_)
        paired = sorted(zip(boxes, kps_list), key=lambda p: (p[0][0]+p[0][2])//2)
        picker_candidates.append(dict(
            frame_frac     = frac,
            detected_boxes = [p[0] for p in paired],
            box_hists      = [_extract_hist_body(frm, b, k)
                              for b, k in zip([p[0] for p in paired],
                                              [p[1] for p in paired])],
            score          = score,
        ))
    cap.release()

    picker_candidates.sort(key=lambda c: c["score"], reverse=True)
    if not picker_candidates:
        if first_frame is None:
            return jsonify(error="Cannot read downloaded video"), 500
        picker_candidates = [dict(frame_frac=0.30, detected_boxes=[],
                                  box_hists=[], score=0.0)]

    best      = picker_candidates[0]
    raw       = best["detected_boxes"]
    box_hists = best["box_hists"]

    cap2 = cv2.VideoCapture(str(vpath))
    cap2.set(cv2.CAP_PROP_POS_FRAMES,
             max(0, int(total_f * best["frame_frac"])) if total_f > 0 else 0)
    ok, frame = cap2.read()
    cap2.release()
    if not ok:
        return jsonify(error="Cannot read downloaded video frame"), 500

    fh, fw = frame.shape[:2]
    preview = frame.copy()
    for i, (x1,y1,x2,y2) in enumerate(raw):
        col = PICK_PALETTE_BGR[i % len(PICK_PALETTE_BGR)]
        cv2.rectangle(preview, (x1,y1), (x2,y2), col, 3)
        cv2.putText(preview, f"Boxer {i+1}", (x1+4, max(y1-14, 28)),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, col, 2)
    cv2.imwrite(str(sess_dir(sid)/"preview.jpg"), preview)
    cv2.imwrite(str(sess_dir(sid)/"first_frame.jpg"), frame)

    meta = dict(
        id               = sid,
        filename         = filename,
        video_ext        = ext,
        upload_time      = datetime.utcnow().isoformat(),
        status           = "picking",
        boxer_index      = None,
        op_index         = None,
        detected_boxes   = raw,
        box_hists        = box_hists,
        palette_css      = PICK_PALETTE_CSS[:len(raw)],
        frame_size       = [fw, fh],
        fps              = fps_r,
        total_frames     = total_f,
        progress         = 0,
        metrics          = None,
        annotated_ready  = False,
        error            = None,
        my_ref_center    = None,
        op_ref_center    = None,
        my_hist          = None,
        op_hist          = None,
        picker_candidates= picker_candidates,
        picker_cand_idx  = 0,
        source_url       = url,
    )
    write_meta(sid, meta)
    return jsonify(sid=sid)


@app.route("/pick/<sid>")
def pick(sid):
    meta = read_meta(sid)
    if meta is None: return "Session not found", 404
    n            = len(meta.get("detected_boxes", []))
    cand_idx     = meta.get("picker_cand_idx", 0)
    total_cands  = len(meta.get("picker_candidates", []))
    frame_pct    = int(meta.get("picker_candidates", [{}])[cand_idx].get("frame_frac", 0.30) * 100)
    return render_template("pick.html", meta=meta, n=n,
                           cand_idx=cand_idx, total_cands=total_cands,
                           frame_pct=frame_pct)


@app.route("/pick/<sid>/reframe", methods=["POST"])
def reframe(sid):
    """Cycle to the next pre-scored candidate frame."""
    meta = read_meta(sid)
    if meta is None: return "Session not found", 404

    picker_cands = meta.get("picker_candidates", [])
    if len(picker_cands) <= 1:
        return redirect(url_for("pick", sid=sid))

    idx  = (meta.get("picker_cand_idx", 0) + 1) % len(picker_cands)
    cand = picker_cands[idx]

    raw       = cand["detected_boxes"]
    box_hists = cand["box_hists"]

    meta["picker_cand_idx"] = idx
    meta["detected_boxes"]  = raw
    meta["box_hists"]       = box_hists
    meta["palette_css"]     = PICK_PALETTE_CSS[:len(raw)]

    # Re-render preview.jpg from the stored frame fraction
    vpath   = str(sess_dir(sid) / f"original{meta['video_ext']}")
    total_f = int(meta.get("total_frames") or 0)
    cap     = cv2.VideoCapture(vpath)
    target  = max(0, int(total_f * cand["frame_frac"])) if total_f > 0 else 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, frame = cap.read()
    cap.release()

    if ok:
        preview = frame.copy()
        for i, (x1,y1,x2,y2) in enumerate(raw):
            col = PICK_PALETTE_BGR[i % len(PICK_PALETTE_BGR)]
            cv2.rectangle(preview, (x1,y1), (x2,y2), col, 3)
            cv2.putText(preview, f"Boxer {i+1}",
                        (x1+4, max(y1-14, 28)),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, col, 2)
        cv2.imwrite(str(sess_dir(sid)/"preview.jpg"), preview)

    write_meta(sid, meta)
    return redirect(url_for("pick", sid=sid))


@app.route("/pick/<sid>", methods=["POST"])
def choose(sid):
    meta = read_meta(sid)
    if meta is None: return "Session not found", 404

    my_idx  = int(request.form.get("my_idx",  0))
    op_idx  = int(request.form.get("op_idx", -1))
    ref_idx = int(request.form.get("ref_idx", -1))
    boxes   = meta.get("detected_boxes", [])
    hists   = meta.get("box_hists", [])

    def bx_center(b):
        return [(b[0]+b[2])//2, (b[1]+b[3])//2]

    meta["boxer_index"]   = my_idx
    meta["op_index"]      = op_idx
    meta["ref_index"]     = ref_idx
    meta["my_ref_center"] = bx_center(boxes[my_idx])  if my_idx  < len(boxes) else None
    meta["op_ref_center"] = bx_center(boxes[op_idx])  if 0 <= op_idx  < len(boxes) else None
    meta["my_hist"]       = hists[my_idx]  if my_idx  < len(hists) else None
    meta["op_hist"]       = hists[op_idx]  if 0 <= op_idx  < len(hists) else None
    meta["ref_hist"]      = hists[ref_idx] if 0 <= ref_idx < len(hists) else None
    meta["ref_box_init"]  = boxes[ref_idx] if 0 <= ref_idx < len(boxes) else None
    meta["status"]        = "analysing"
    write_meta(sid, meta)

    threading.Thread(target=analyse, args=(sid,), daemon=True).start()
    return redirect(url_for("session", sid=sid))


@app.route("/session/<sid>")
def session(sid):
    meta = read_meta(sid)
    if meta is None: return "Session not found", 404
    return render_template("session.html", meta=meta)


def _read_sam2_progress(sid):
    """Return live SAM2 progress dict from the progress JSON file, or None."""
    p = sess_dir(sid) / "sam2_test_progress.json"
    try:
        if p.exists():
            return json.loads(p.read_text())
    except Exception:
        pass
    return None


# ── SAM2 pose enrichment + metrics ────────────────────────────────────────────
# SAM2 = identity anchor (who is who).  YOLO-pose = pose/punch detection.
# They run at DIFFERENT frame rates and are bridged by interpolated bboxes.
# Post-clinch colour histogram catches SAM2 identity swaps.

# TODO(production): restore TARGET_POSE_FPS = 20
TARGET_POSE_FPS = 15   # TEMP: slightly lower for M2 Air comfort


def _run_pose_enrichment(sid):
    """
    After SAM2 tracking, run YOLO-pose at TARGET_POSE_FPS (decoupled from SAM2
    fps) and match keypoints to SAM2 identities via interpolated bboxes.
    Post-clinch colour histogram corrects potential SAM2 label swaps.
    Writes sam2_enriched.json.
    """
    import bisect
    import numpy as np

    meta = read_meta(sid)
    if meta is None:
        return False

    track_path = sess_dir(sid) / "sam2_track.json"
    if not track_path.exists():
        print(f"[enrichment] No sam2_track.json for {sid}")
        return False

    try:
        sam2_frames = json.loads(track_path.read_text())
    except Exception as e:
        print(f"[enrichment] Failed to read track JSON: {e}")
        return False

    for candidate in ("lab_compressed.mp4", "compressed.mp4"):
        _c = sess_dir(sid) / candidate
        if _c.exists():
            vpath = str(_c)
            break
    else:
        vpath = str(sess_dir(sid) / f"original{meta['video_ext']}")

    cap = cv2.VideoCapture(vpath)
    fps_orig = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    stride_pose = max(1, round(fps_orig / TARGET_POSE_FPS))
    fps_pose = fps_orig / stride_pose

    sam2_scale = meta.get("sam2_test_scale", 1.0) or 1.0
    sam2_stride = meta.get("lab_seed_stride") or max(1, round(fps_orig / 3.0))
    inv_scale = 1.0 / sam2_scale if sam2_scale > 0 else 1.0

    # ── Build sorted SAM2 bbox timeline for interpolation ─────────────────
    sam2_by_raw: dict = {}
    for sf in sam2_frames:
        sam2_by_raw[sf["raw_fi"]] = sf
    sam2_raw_fis = sorted(sam2_by_raw.keys())

    def _lerp_bbox(b1, b2, t):
        """Linear interpolation between two bboxes.  t in [0, 1]."""
        if b1 is None:
            return b2
        if b2 is None:
            return b1
        return [b1[i] + (b2[i] - b1[i]) * t for i in range(4)]

    def _interp_sam2(raw_fi):
        """Return interpolated (my_bbox, op_bbox) at raw_fi in video coords."""
        idx = bisect.bisect_left(sam2_raw_fis, raw_fi)

        # Exact match
        if idx < len(sam2_raw_fis) and sam2_raw_fis[idx] == raw_fi:
            sf = sam2_by_raw[raw_fi]
            my_s = sf.get("my_bbox")
            op_s = sf.get("op_bbox")
            my_b = ([my_s[i] * inv_scale for i in range(4)] if my_s else None)
            op_b = ([op_s[i] * inv_scale for i in range(4)] if op_s else None)
            return my_b, op_b

        # Edge cases
        if idx == 0:
            sf = sam2_by_raw[sam2_raw_fis[0]]
            my_s = sf.get("my_bbox")
            op_s = sf.get("op_bbox")
            return (([my_s[i] * inv_scale for i in range(4)] if my_s else None),
                    ([op_s[i] * inv_scale for i in range(4)] if op_s else None))
        if idx >= len(sam2_raw_fis):
            sf = sam2_by_raw[sam2_raw_fis[-1]]
            my_s = sf.get("my_bbox")
            op_s = sf.get("op_bbox")
            return (([my_s[i] * inv_scale for i in range(4)] if my_s else None),
                    ([op_s[i] * inv_scale for i in range(4)] if op_s else None))

        # Interpolate between bracketing SAM2 frames
        fi_a = sam2_raw_fis[idx - 1]
        fi_b = sam2_raw_fis[idx]
        t = (raw_fi - fi_a) / max(fi_b - fi_a, 1)
        sf_a = sam2_by_raw[fi_a]
        sf_b = sam2_by_raw[fi_b]

        my_a = sf_a.get("my_bbox")
        my_b = sf_b.get("my_bbox")
        op_a = sf_a.get("op_bbox")
        op_b = sf_b.get("op_bbox")
        my_a_v = [my_a[i] * inv_scale for i in range(4)] if my_a else None
        my_b_v = [my_b[i] * inv_scale for i in range(4)] if my_b else None
        op_a_v = [op_a[i] * inv_scale for i in range(4)] if op_a else None
        op_b_v = [op_b[i] * inv_scale for i in range(4)] if op_b else None

        return _lerp_bbox(my_a_v, my_b_v, t), _lerp_bbox(op_a_v, op_b_v, t)

    def _iou_enrich(a, b):
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        aa = max(1, (a[2] - a[0]) * (a[3] - a[1]))
        ab = max(1, (b[2] - b[0]) * (b[3] - b[1]))
        return inter / (aa + ab - inter)

    # ── Colour histogram helpers for post-clinch identity correction ──────
    def _torso_hsv_hist(frame_bgr, bbox):
        """Extract HSV hue histogram from the torso region of a bbox."""
        if bbox is None:
            return None
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(fw, int(bbox[2]))
        y2 = min(fh, int(bbox[3]))
        # Torso = middle 60% vertically, middle 60% horizontally
        h = y2 - y1
        w = x2 - x1
        if h < 10 or w < 10:
            return None
        ty1 = y1 + int(h * 0.15)
        ty2 = y1 + int(h * 0.55)
        tx1 = x1 + int(w * 0.20)
        tx2 = x1 + int(w * 0.80)
        crop = frame_bgr[ty1:ty2, tx1:tx2]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [18, 8],
                            [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    def _hist_corr(h1, h2):
        """Correlation between two histograms. 1.0 = identical."""
        if h1 is None or h2 is None:
            return 0.0
        return float(cv2.compareHist(
            h1.astype(np.float32), h2.astype(np.float32),
            cv2.HISTCMP_CORREL))

    yolo_model = get_model()
    enriched_frames = []

    fi = 0
    _max_fi = int(fps_orig * 300)  # 5 min cap
    pose_count = 0

    # Clinch state tracking for colour-histogram identity correction
    in_clinch = False
    pre_clinch_my_hist = None
    pre_clinch_op_hist = None
    identity_swapped = False    # persistent flag if SAM2 labels are swapped
    swap_events = []            # [{time_s, reason}] for diagnostics

    # Anchor histograms captured early (first confident both-visible frame).
    # Used for a periodic identity check every REID_PERIOD_S so we catch
    # drift that happens during long clinches, not just on clinch exit.
    ref_my_hist = None
    ref_op_hist = None
    REID_PERIOD_S = 1.5
    REID_SWAP_MARGIN = 0.05
    last_reid_t = -1e9

    while True:
        ok, frame = cap.read()
        if not ok or fi > _max_fi:
            break

        if fi % stride_pose == 0:
            # Get interpolated SAM2 bboxes for this exact frame
            my_orig, op_orig = _interp_sam2(fi)

            # Apply persistent swap correction
            if identity_swapped:
                my_orig, op_orig = op_orig, my_orig

            # Run YOLO-pose
            res = yolo_model(frame, verbose=False)[0]

            yolo_dets = []
            if res.boxes is not None:
                for i_det, b in enumerate(res.boxes):
                    if int(b.cls[0]) != 0:
                        continue
                    conf = float(b.conf[0])
                    if conf < 0.30:
                        continue
                    box = [float(v) for v in b.xyxy[0].tolist()]
                    kps = None
                    if (res.keypoints is not None
                            and i_det < len(res.keypoints)):
                        kd = res.keypoints[i_det].data[0]   # [17, 3]
                        kps = [[float(kd[k][0]), float(kd[k][1]),
                                float(kd[k][2])] for k in range(17)]
                    yolo_dets.append({"box": box, "kps": kps, "conf": conf})

            # Match YOLO detections to (interpolated) SAM2 identities by IoU
            my_kps = None
            op_kps = None
            my_yolo_box = None
            op_yolo_box = None
            used_idx = -1

            if my_orig and yolo_dets:
                best_iou, best_j = 0.15, -1
                for j, yd in enumerate(yolo_dets):
                    iou = _iou_enrich(my_orig, yd["box"])
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                if best_j >= 0:
                    my_kps = yolo_dets[best_j]["kps"]
                    my_yolo_box = yolo_dets[best_j]["box"]
                    used_idx = best_j

            if op_orig and yolo_dets:
                best_iou, best_j = 0.15, -1
                for j, yd in enumerate(yolo_dets):
                    if j == used_idx:
                        continue
                    iou = _iou_enrich(op_orig, yd["box"])
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                if best_j >= 0:
                    op_kps = yolo_dets[best_j]["kps"]
                    op_yolo_box = yolo_dets[best_j]["box"]

            # Clinch detection: bbox centres within ¼ combined width
            clinch = False
            if my_orig and op_orig:
                mcx = (my_orig[0] + my_orig[2]) / 2
                mcy = (my_orig[1] + my_orig[3]) / 2
                ocx = (op_orig[0] + op_orig[2]) / 2
                ocy = (op_orig[1] + op_orig[3]) / 2
                mw  = my_orig[2] - my_orig[0]
                ow  = op_orig[2] - op_orig[0]
                clinch = math.hypot(mcx - ocx, mcy - ocy) < (mw + ow) / 4

            # ── Anchor reference histograms on the first clean both-visible
            #    frame (not in clinch, both YOLO boxes present). ────────────
            t_now = fi / fps_orig
            if (ref_my_hist is None and ref_op_hist is None
                    and not clinch
                    and my_yolo_box is not None and op_yolo_box is not None):
                cand_my = _torso_hsv_hist(frame, my_yolo_box)
                cand_op = _torso_hsv_hist(frame, op_yolo_box)
                if cand_my is not None and cand_op is not None:
                    ref_my_hist, ref_op_hist = cand_my, cand_op
                    print(f"[enrichment] Anchored reference histograms at "
                          f"t={t_now:.2f}s")

            # ── Post-clinch colour histogram identity correction ──────────
            if clinch and not in_clinch:
                # Entering clinch — snapshot colour histograms
                pre_clinch_my_hist = _torso_hsv_hist(frame, my_yolo_box or my_orig)
                pre_clinch_op_hist = _torso_hsv_hist(frame, op_yolo_box or op_orig)
                in_clinch = True
            elif not clinch and in_clinch:
                # Exiting clinch — check if identities swapped
                in_clinch = False
                if pre_clinch_my_hist is not None and pre_clinch_op_hist is not None:
                    post_my_hist = _torso_hsv_hist(frame, my_yolo_box or my_orig)
                    post_op_hist = _torso_hsv_hist(frame, op_yolo_box or op_orig)
                    if post_my_hist is not None and post_op_hist is not None:
                        # Compare: does current "my" look more like pre-clinch "op"?
                        same_corr = (_hist_corr(pre_clinch_my_hist, post_my_hist)
                                     + _hist_corr(pre_clinch_op_hist, post_op_hist))
                        swap_corr = (_hist_corr(pre_clinch_my_hist, post_op_hist)
                                     + _hist_corr(pre_clinch_op_hist, post_my_hist))
                        if swap_corr > same_corr + 0.15:
                            identity_swapped = not identity_swapped
                            # Flip this frame's assignments
                            my_orig, op_orig = op_orig, my_orig
                            my_kps, op_kps = op_kps, my_kps
                            my_yolo_box, op_yolo_box = op_yolo_box, my_yolo_box
                            swap_events.append({
                                "time_s": round(t_now, 2),
                                "reason": "clinch_exit",
                                "same":   round(same_corr, 3),
                                "swap":   round(swap_corr, 3),
                            })
                            print(f"[enrichment] Clinch-exit swap at "
                                  f"t={t_now:.2f}s "
                                  f"(same={same_corr:.2f} swap={swap_corr:.2f})")

            # ── Periodic reference-anchored re-ID (catches drift that the
            #    clinch-exit check misses — especially long clinches where
            #    SAM2 flipped mid-clinch but the clinch-exit histograms look
            #    consistent with the *flipped* pre-clinch snapshot). ───────
            if (ref_my_hist is not None and ref_op_hist is not None
                    and not clinch
                    and (t_now - last_reid_t) >= REID_PERIOD_S
                    and my_yolo_box is not None and op_yolo_box is not None):
                last_reid_t = t_now
                cur_my = _torso_hsv_hist(frame, my_yolo_box)
                cur_op = _torso_hsv_hist(frame, op_yolo_box)
                if cur_my is not None and cur_op is not None:
                    same = (_hist_corr(ref_my_hist, cur_my)
                            + _hist_corr(ref_op_hist, cur_op))
                    swap = (_hist_corr(ref_my_hist, cur_op)
                            + _hist_corr(ref_op_hist, cur_my))
                    if swap > same + REID_SWAP_MARGIN:
                        identity_swapped = not identity_swapped
                        my_orig, op_orig = op_orig, my_orig
                        my_kps, op_kps = op_kps, my_kps
                        my_yolo_box, op_yolo_box = op_yolo_box, my_yolo_box
                        swap_events.append({
                            "time_s": round(t_now, 2),
                            "reason": "periodic_reid",
                            "same":   round(same, 3),
                            "swap":   round(swap, 3),
                        })
                        print(f"[enrichment] Periodic re-ID swap at "
                              f"t={t_now:.2f}s "
                              f"(same={same:.2f} swap={swap:.2f})")

            # Normalise to [0, 1]
            def _nb(b):
                return [b[0] / fw, b[1] / fh, b[2] / fw, b[3] / fh] if b else None
            def _nk(kps):
                return ([[k[0] / fw, k[1] / fh, k[2]] for k in kps]
                        if kps else None)

            enriched_frames.append({
                "fi":      pose_count,
                "raw_fi":  fi,
                "time_s":  round(fi / fps_orig, 4),
                "my_bbox": _nb(my_orig),
                "op_bbox": _nb(op_orig),
                "my_kps":  _nk(my_kps),
                "op_kps":  _nk(op_kps),
                "clinch":  clinch,
            })
            pose_count += 1

        fi += 1

    cap.release()

    enriched = {
        "fps_pose":    round(fps_pose, 2),
        "fps_orig":    round(fps_orig, 2),
        "stride_pose": stride_pose,
        "stride_sam2": sam2_stride,
        "frame_w":     fw,
        "frame_h":     fh,
        "identity_swapped": identity_swapped,
        "swap_events":      swap_events,
        "frames":      enriched_frames,
    }
    out_path = sess_dir(sid) / "sam2_enriched.json"
    out_path.write_text(json.dumps(enriched))
    print(f"[enrichment] Wrote {len(enriched_frames)} enriched frames → {out_path}"
          f"  (fps_pose={fps_pose:.1f}, swapped={identity_swapped})")

    # Emit baseline diagnostics bundle (fighter boxes + pose skeleton).
    # Later stages append to this bundle as they are added.
    try:
        diagnostics.refresh_from_enriched(sess_dir(sid))
    except Exception as _e:
        print(f"[enrichment] Diagnostics build failed (non-fatal): {_e}")

    # Baseline screenshots rendered later (after metrics) so target zones and
    # landed-hit red flashes can be drawn.

    # Arena detection — motion-compensated, built from SAM2-tracked feet.
    # Session-scoped because it depends on the user's picked identities.
    try:
        arena_out = sess_dir(sid) / "arena.json"
        arena = arena_detector.detect_and_save(vpath, enriched, arena_out)
        if arena.get("ok"):
            # Inject fps_pose so screenshot rendering can resolve timestamps.
            arena["fps_pose"] = enriched.get("fps_pose", 15.0)
            arena_out.write_text(json.dumps(arena, separators=(",", ":")))
            shots_dir = sess_dir(sid) / "screenshots" / "arena"
            n = screenshots.render_arena(vpath, arena, shots_dir, n=6)
            heat_dir = sess_dir(sid) / "screenshots" / "arena_heatmap"
            nh = screenshots.render_arena_heatmap(vpath, arena, heat_dir)
            print(f"[enrichment] arena tier={arena['confidence_tier']} "
                  f"visibility={arena['visibility_score']:.2f} "
                  f"(ME={arena.get('my_visibility_score',0):.2f} "
                  f"OP={arena.get('op_visibility_score',0):.2f}) "
                  f"n_pts={arena['n_foot_points']} "
                  f"span={arena.get('span_in_scales', 0):.2f} scales — "
                  f"wrote {n} arena + {nh} heatmap screenshots")

            # Compute the five arena-aware metrics, weighted by tier.
            arena_metrics_result = None
            try:
                m = arena_metrics_mod.compute_and_save(sess_dir(sid))
                if m.get("ok"):
                    arena_metrics_result = m
                    meta = read_meta(sid) or {}
                    meta["arena_metrics"] = m
                    write_meta(sid, meta)
                    print(f"[enrichment] metrics  ring={m['my_ring']}/{m['op_ring']}"
                          f"  aggr={m['my_aggression']}/{m['op_aggression']}"
                          f"  mov={m['my_movement']}/{m['op_movement']}"
                          f"  vol={m['my_volume']}/{m['op_volume']}"
                          f"  def={m['my_defense']}/{m['op_defense']}"
                          f"  guard={m['my_guard']}/{m['op_guard']}"
                          f"  punches={m['my_punches']}/{m['op_punches']}"
                          f"  landed me={m['my_landed_head']}H+{m['my_landed_body']}B"
                          f"  op={m['op_landed_head']}H+{m['op_landed_body']}B")
            except Exception as _me:
                print(f"[enrichment] Arena metrics failed (non-fatal): {_me}")

            # Baseline screenshots — now with target zones + landed-hit flashes.
            try:
                shots_dir = sess_dir(sid) / "screenshots" / "baseline"
                n = screenshots.render_baseline(vpath, enriched, shots_dir,
                                                n=12,
                                                arena_metrics=arena_metrics_result)
                print(f"[enrichment] wrote {n} baseline screenshots to {shots_dir}")
            except Exception as _e:
                print(f"[enrichment] Baseline screenshots failed (non-fatal): {_e}")

            # Punch-peak screenshots — one frame per landed-hit candidate so
            # the user can eyeball every verdict in order. Much better UX than
            # hunting for red flashes in live video.
            if arena_metrics_result:
                try:
                    peaks_dir = sess_dir(sid) / "screenshots" / "punch_peaks"
                    n = screenshots.render_punch_peaks(
                        vpath, enriched, arena_metrics_result, peaks_dir)
                    print(f"[enrichment] wrote {n} punch-peak screenshots to {peaks_dir}")
                except Exception as _e:
                    print(f"[enrichment] Punch-peak screenshots failed (non-fatal): {_e}")
        else:
            print(f"[enrichment] arena detection failed: {arena.get('error')}")
    except Exception as _e:
        print(f"[enrichment] Arena detection failed (non-fatal): {_e}")

    return True




def _compute_sam2_metrics(sid):
    """
    Compute boxing metrics from SAM2-tracked + YOLO-pose enriched data.
    v3: heuristic punch detection,
        punch-driven aggression, direction-weighted pace, context-split guard.
    Stores result in meta['sam2_metrics'].
    """
    enriched_path = sess_dir(sid) / "sam2_enriched.json"
    if not enriched_path.exists():
        return False

    data = json.loads(enriched_path.read_text())
    frames = data["frames"]
    fps_pose = data["fps_pose"]

    if len(frames) < 4:
        return False

    duration_s = int(frames[-1]["time_s"]) + 1
    KP_CONF = 0.35   # slightly more lenient for higher fps

    # ── Helper functions ─────────────────────────────────────────────────────

    def _sh_centroid(kps):
        """Shoulder centroid from keypoints 5 (left shoulder) and 6 (right)."""
        if not kps:
            return None
        pts = [(kps[i][0], kps[i][1])
               for i in (5, 6) if kps[i][2] >= KP_CONF]
        if not pts:
            return None
        return (sum(p[0] for p in pts) / len(pts),
                sum(p[1] for p in pts) / len(pts))

    def _bcenter(bbox):
        if not bbox:
            return None
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    def _bbox_h(bbox):
        if not bbox:
            return 0.3
        return bbox[3] - bbox[1]

    # ── Punch detection v2: weighted score (acceleration > extension) ────
    # Extension ratio = distance(wrist, shoulder) / torso_height
    # Wrist acceleration = change in body-relative wrist position between frames
    # Weighted score: 0.30 * ext_norm + 0.70 * accel_norm > threshold
    # Acceleration weighted higher because it catches all punch types (hooks,
    # uppercuts) while extension alone only catches straights.
    # Consecutive punch frames are deduplicated into single punches.

    def _wrist_positions(kps, sh):
        """Return body-relative wrist positions [(rx, ry), ...] for indices 9,10."""
        if not kps or not sh:
            return [None, None]
        out = []
        for wi in (9, 10):
            if kps[wi][2] >= KP_CONF:
                out.append((kps[wi][0] - sh[0], kps[wi][1] - sh[1]))
            else:
                out.append(None)
        return out

    def _wrist_extension(kps, sh, torso_h):
        """Max wrist extension ratio (dist from shoulder / torso height)."""
        if not kps or not sh or torso_h < 0.01:
            return 0.0
        mx = 0.0
        for wi in (9, 10):
            if kps[wi][2] >= KP_CONF:
                dx = kps[wi][0] - sh[0]
                dy = kps[wi][1] - sh[1]
                mx = max(mx, math.hypot(dx, dy) / torso_h)
        return mx

    def _wrist_accel(wrists_cur, wrists_prev):
        """Max wrist acceleration (change in body-relative position)."""
        mx = 0.0
        for wc, wp in zip(wrists_cur, wrists_prev):
            if wc is not None and wp is not None:
                mx = max(mx, math.hypot(wc[0] - wp[0], wc[1] - wp[1]))
        return mx

    # Median bbox height for normalisation
    all_h = []
    for f in frames:
        for key in ("my_bbox", "op_bbox"):
            b = f.get(key)
            if b:
                all_h.append(b[3] - b[1])
    median_h = sorted(all_h)[len(all_h) // 2] if all_h else 0.3

    # Collect p90 of extension and acceleration for score normalisation
    _all_ext = []
    _all_acc = []
    for _i, _f in enumerate(frames):
        _prev = frames[_i - 1] if _i > 0 else None
        for _pfx in ("my", "op"):
            _kps = _f.get(f"{_pfx}_kps")
            _sh  = _sh_centroid(_kps)
            _th  = _bbox_h(_f.get(f"{_pfx}_bbox"))
            _pk  = _prev.get(f"{_pfx}_kps") if _prev else None
            _psh = _sh_centroid(_pk) if _prev else None
            _w   = _wrist_positions(_kps, _sh)
            _wp  = _wrist_positions(_pk, _psh) if _prev else [None, None]
            _e   = _wrist_extension(_kps, _sh, _th)
            _a   = _wrist_accel(_w, _wp)
            if _e > 0:
                _all_ext.append(_e)
            if _a > 0:
                _all_acc.append(_a)
    _all_ext.sort()
    _all_acc.sort()
    ext_p90 = _all_ext[int(len(_all_ext) * 0.90)] if _all_ext else 0.30
    acc_p90 = _all_acc[int(len(_all_acc) * 0.90)] if _all_acc else 0.10

    # Punch scoring: weighted combination (acceleration has 70% weight)
    PUNCH_W_EXT  = 0.30
    PUNCH_W_ACC  = 0.70
    PUNCH_SCORE_THRESH = 1.10
    PUNCH_ACCEL_FLOOR  = median_h * 0.03   # minimum accel to filter noise

    def _is_punch(ext, accel):
        if accel < PUNCH_ACCEL_FLOOR:
            return False
        score = (PUNCH_W_EXT * (ext / max(ext_p90, 0.01))
                 + PUNCH_W_ACC * (accel / max(acc_p90, 0.001)))
        return score > PUNCH_SCORE_THRESH

    # Dedup: adaptive gap based on fps (~0.35s between separate punches)
    PUNCH_DEDUP_GAP = max(2, round(fps_pose * 0.35))

    # ── Guard v2: context-split (offensive vs defensive) ──────────────────
    def _guard_score_v2(kps, is_punching):
        """
        Context-split guard scoring.
        When NOT punching: both hands should be up → score both wrists.
        When punching: only score the non-punching (higher, more retracted) hand.
        Returns (guard_score, is_defensive_frame).
        """
        if not kps:
            return 0.5, True
        sh_pts = [kps[i] for i in (5, 6) if kps[i][2] >= KP_CONF]
        if not sh_pts:
            return 0.5, True
        avg_sh_y = sum(p[1] for p in sh_pts) / len(sh_pts)
        slack = 0.08  # normalised coords

        wrist_up = {}
        for wi in (9, 10):
            if kps[wi][2] >= KP_CONF:
                wrist_up[wi] = kps[wi][1] <= avg_sh_y + slack

        if not wrist_up:
            return 0.5, not is_punching

        if not is_punching:
            # Defensive frame: score both hands
            up = sum(1 for v in wrist_up.values() if v)
            total = len(wrist_up)
            if total == 2:
                score = [0.0, 0.5, 1.0][up]
            else:
                score = 1.0 if up else 0.0
            return score, True
        else:
            # Offensive frame: only score the non-punching hand
            # The non-punching hand is the one closer to the shoulder (less extended)
            sh_c = (sum(p[0] for p in sh_pts) / len(sh_pts),
                    sum(p[1] for p in sh_pts) / len(sh_pts))
            best_guard_wrist = None
            min_dist = float('inf')
            for wi in wrist_up:
                dx = kps[wi][0] - sh_c[0]
                dy = kps[wi][1] - sh_c[1]
                d = math.hypot(dx, dy)
                if d < min_dist:
                    min_dist = d
                    best_guard_wrist = wi
            if best_guard_wrist is not None:
                return (1.0 if wrist_up[best_guard_wrist] else 0.0), False
            return 0.5, False

    # ── Per-frame signal extraction ──────────────────────────────────────────
    sigs = []
    for i, f in enumerate(frames):
        prev = frames[i - 1] if i > 0 else None
        s = {"sec": int(f["time_s"]), "time_s": f["time_s"],
             "clinch": f.get("clinch", False)}

        # ME
        my_kps = f.get("my_kps")
        my_sh  = _sh_centroid(my_kps)
        my_th  = _bbox_h(f.get("my_bbox"))
        p_my_kps = prev.get("my_kps") if prev else None
        p_my_sh  = _sh_centroid(p_my_kps) if prev else None

        my_wrists      = _wrist_positions(my_kps, my_sh)
        my_wrists_prev = _wrist_positions(p_my_kps, p_my_sh) if prev else [None, None]
        my_ext   = _wrist_extension(my_kps, my_sh, my_th)
        my_accel = _wrist_accel(my_wrists, my_wrists_prev)
        my_punch = _is_punch(my_ext, my_accel)

        my_bd = (math.hypot(my_sh[0] - p_my_sh[0], my_sh[1] - p_my_sh[1])
                 if my_sh and p_my_sh else 0.0)

        my_guard_val, my_guard_def = _guard_score_v2(my_kps, my_punch)
        s["my_punch"]     = my_punch
        s["my_ext"]       = my_ext
        s["my_accel"]     = my_accel
        s["my_guard"]     = my_guard_val
        s["my_guard_def"] = my_guard_def  # True = defensive frame
        s["my_bd"]        = my_bd

        my_c = _bcenter(f.get("my_bbox"))
        op_c = _bcenter(f.get("op_bbox"))
        s["my_cx"] = my_c[0] if my_c else None
        s["op_cx"] = op_c[0] if op_c else None

        # OP
        op_kps = f.get("op_kps")
        op_sh  = _sh_centroid(op_kps)
        op_th  = _bbox_h(f.get("op_bbox"))
        p_op_kps = prev.get("op_kps") if prev else None
        p_op_sh  = _sh_centroid(p_op_kps) if prev else None

        op_wrists      = _wrist_positions(op_kps, op_sh)
        op_wrists_prev = _wrist_positions(p_op_kps, p_op_sh) if prev else [None, None]
        op_ext   = _wrist_extension(op_kps, op_sh, op_th)
        op_accel = _wrist_accel(op_wrists, op_wrists_prev)
        op_punch = _is_punch(op_ext, op_accel)

        op_bd = (math.hypot(op_sh[0] - p_op_sh[0], op_sh[1] - p_op_sh[1])
                 if op_sh and p_op_sh else 0.0)

        op_guard_val, op_guard_def = _guard_score_v2(op_kps, op_punch)
        s["op_punch"]     = op_punch
        s["op_ext"]       = op_ext
        s["op_accel"]     = op_accel
        s["op_guard"]     = op_guard_val
        s["op_guard_def"] = op_guard_def
        s["op_bd"]        = op_bd

        # Forward/backward activity (raw, for pace)
        s["my_act"] = my_bd + 1.5 * my_accel
        s["op_act"] = op_bd + 1.5 * op_accel

        sigs.append(s)

    # Punch detection is heuristic-only. (Future pass: audio-based.)

    # ── Group by second ──────────────────────────────────────────────────────
    by_sec: dict = {}
    for s in sigs:
        by_sec.setdefault(s["sec"], []).append(s)

    # ── ADVANCE / RETREAT v2: windowed (1.5s), "who is closing distance" ────
    # For each second, look at a 1.5s window centered on it.
    # Compute: did the distance between fighters shrink?
    #          Who moved more toward the other?
    ADV_WINDOW = 1.5  # seconds

    def _windowed_advance(sec):
        """
        Returns (my_pressing, op_pressing) booleans for this second.
        'Pressing' = I moved toward my opponent AND the gap closed.
        """
        t_start = sec - ADV_WINDOW / 2
        t_end   = sec + ADV_WINDOW / 2
        # Collect positions in this window
        my_positions = []
        op_positions = []
        for sg in sigs:
            if t_start <= sg["time_s"] <= t_end:
                if sg["my_cx"] is not None and sg["op_cx"] is not None:
                    my_positions.append(sg["my_cx"])
                    op_positions.append(sg["op_cx"])

        if len(my_positions) < 3:
            return False, False

        n = len(my_positions)
        half = n // 2

        # Average positions in first half vs second half
        my_first = sum(my_positions[:half]) / half
        my_second = sum(my_positions[half:]) / (n - half)
        op_first = sum(op_positions[:half]) / half
        op_second = sum(op_positions[half:]) / (n - half)

        # Distance between fighters
        dist_first  = abs(my_first - op_first)
        dist_second = abs(my_second - op_second)
        gap_closed = dist_first - dist_second

        # How much did each fighter move toward the other?
        # "toward" = in the direction of the opponent
        toward_dir_my = 1.0 if op_first > my_first else -1.0
        toward_dir_op = 1.0 if my_first > op_first else -1.0
        my_toward = (my_second - my_first) * toward_dir_my
        op_toward = (op_second - op_first) * toward_dir_op

        # Threshold: meaningful movement > 0.5% of frame width
        move_thresh = 0.005

        my_pressing = my_toward > move_thresh and gap_closed > 0
        op_pressing = op_toward > move_thresh and gap_closed > 0

        return my_pressing, op_pressing

    # Pre-compute advance/retreat per second
    adv_per_sec: dict = {}
    for sec in range(duration_s):
        my_press, op_press = _windowed_advance(sec)
        adv_per_sec[sec] = {
            "my_adv": my_press,
            "op_adv": op_press,
            "my_ret": op_press and not my_press,  # opponent pressing, I'm not
            "op_ret": my_press and not op_press,
        }

    # ── PACE v2 (direction-weighted: forward+punching = full, retreat = 0.3x) ─
    sw = max(5, min(15, duration_s // 6))

    def _pace_series_v2(act_key, bd_key, punch_key, fighter):
        raw = []
        last = 0.001
        for s in range(duration_s):
            vals = []
            adv_info = adv_per_sec.get(s, {})
            is_pressing = adv_info.get(f"{fighter}_adv", False)
            is_retreating = adv_info.get(f"{fighter}_ret", False)

            for ws in range(max(0, s - sw // 2), min(duration_s, s + sw // 2 + 1)):
                for sig in by_sec.get(ws, []):
                    v = sig[act_key]
                    if v > 0:
                        # Direction weight: forward/punch = 1.0, retreat = 0.3
                        if is_retreating:
                            v *= 0.3
                        elif is_pressing or sig[punch_key]:
                            v *= 1.0
                        else:
                            v *= 0.6  # neutral
                        vals.append(v)
            if len(vals) >= 3:
                vals.sort()
                p80 = vals[int(len(vals) * 0.8)]
                raw.append(p80)
                last = p80
            else:
                raw.append(last)
        return raw

    my_pace_r = _pace_series_v2("my_act", "my_bd", "my_punch", "my")
    op_pace_r = _pace_series_v2("op_act", "op_bd", "op_punch", "op")
    peak = max(max(my_pace_r), max(op_pace_r), 0.001)
    my_pace = [max(1, round(v / peak * 100)) for v in my_pace_r]
    op_pace = [max(1, round(v / peak * 100)) for v in op_pace_r]

    # ── Per-frame advancing / retreating (like old YOLO) ──────────────────
    # For each frame: compute direction of movement relative to opponent.
    ADVANCE_THRESH = 0.003  # normalized movement threshold
    for i, s in enumerate(sigs):
        prev = sigs[i - 1] if i > 0 else None
        for pfx, opfx in [("my", "op"), ("op", "my")]:
            advancing = False
            retreating = False
            if prev and s[f"{pfx}_cx"] is not None and prev[f"{pfx}_cx"] is not None \
               and s[f"{opfx}_cx"] is not None:
                toward_op = 1.0 if s[f"{opfx}_cx"] > s[f"{pfx}_cx"] else -1.0
                dx = s[f"{pfx}_cx"] - prev[f"{pfx}_cx"]
                advancing  = dx * toward_op > ADVANCE_THRESH
                retreating = dx * toward_op < -ADVANCE_THRESH
            s[f"{pfx}_advancing"] = advancing
            s[f"{pfx}_retreating"] = retreating
        # On ropes check (outer 18%)
        s["my_cornered"] = False
        s["op_cornered"] = False
        if s["my_cx"] is not None:
            s["my_cornered"] = s["my_cx"] < 0.18 or s["my_cx"] > 0.82
        if s["op_cx"] is not None:
            s["op_cornered"] = s["op_cx"] < 0.18 or s["op_cx"] > 0.82

    # ── AGGRESSION v2 (punch-driven, like old YOLO) ────────────────────────
    # Aggressing = punching OR (advancing while opponent retreating/cornered
    #              and opponent not counter-punching)
    # Retreating = moving backward while opponent is punching
    def _agg_series_v2(punch_key, adv_key, op_cornered_key, op_ret_key, op_punch_key):
        out = []
        for s in range(duration_s):
            fs = by_sec.get(s, [])
            if not fs:
                out.append(0); continue
            agg_count = 0
            for f in fs:
                my_punch = f[punch_key]
                my_adv = f[adv_key]
                op_corn = f[op_cornered_key]
                op_ret = f[op_ret_key]
                op_pch = f[op_punch_key]
                if my_punch or (my_adv and (op_corn or (op_ret and not op_pch))):
                    agg_count += 1
            out.append(round(agg_count / len(fs) * 100))
        return out

    # Helper: interpolate segment-level values to smooth per-second series
    def _interp_series(seg_mids, seg_vals, dur):
        """Linearly interpolate segment values across per-second series."""
        out = []
        for s in range(dur):
            t = s + 0.5  # centre of this second
            if not seg_mids:
                out.append(0); continue
            if t <= seg_mids[0]:
                out.append(round(seg_vals[0]))
            elif t >= seg_mids[-1]:
                out.append(round(seg_vals[-1]))
            else:
                idx = bisect.bisect_right(seg_mids, t)
                lo, hi = idx - 1, idx
                span = seg_mids[hi] - seg_mids[lo]
                if span > 0:
                    frac = (t - seg_mids[lo]) / span
                    v = seg_vals[lo] + frac * (seg_vals[hi] - seg_vals[lo])
                else:
                    v = seg_vals[lo]
                out.append(round(v))
        return out

    # Always compute heuristic aggression (used as fallback for uncovered seconds)
    my_agg_h = _agg_series_v2("my_punch", "my_advancing", "op_cornered",
                               "op_retreating", "op_punch")
    op_agg_h = _agg_series_v2("op_punch", "op_advancing", "my_cornered",
                               "my_retreating", "my_punch")

    my_agg = my_agg_h
    op_agg = op_agg_h

    # ── GUARD v2 (context-split: 70% defensive discipline, 30% offensive) ──
    def _guard_series_v2(guard_key, guard_def_key):
        out = []
        for s in range(duration_s):
            fs = by_sec.get(s, [])
            if not fs:
                out.append(50); continue
            def_scores = [f[guard_key] for f in fs if f[guard_def_key]]
            off_scores = [f[guard_key] for f in fs if not f[guard_def_key]]
            def_avg = sum(def_scores) / len(def_scores) if def_scores else 0.5
            off_avg = sum(off_scores) / len(off_scores) if off_scores else 0.5
            combined = 0.70 * def_avg + 0.30 * off_avg
            out.append(round(combined * 100))
        return out

    my_guard = _guard_series_v2("my_guard", "my_guard_def")
    op_guard = _guard_series_v2("op_guard", "op_guard_def")

    # ── RING GENERALSHIP v2 ─────────────────────────────────────────────────
    # Always compute heuristic ring control
    my_ring_h, op_ring_h = [], []
    for s in range(duration_s):
        adv = adv_per_sec.get(s, {})
        fs = by_sec.get(s, [])
        cli = sum(1 for f in fs if f["clinch"]) / max(len(fs), 1) if fs else 0

        m_rg = (0.60 * (1.0 if adv.get("my_adv") else 0.0)
                + 0.25 * (0.0 if adv.get("my_ret") else 1.0)
                + 0.15 * (1 - cli))
        o_rg = (0.60 * (1.0 if adv.get("op_adv") else 0.0)
                + 0.25 * (0.0 if adv.get("op_ret") else 1.0)
                + 0.15 * (1 - cli))
        tot = m_rg + o_rg
        if tot > 0:
            my_ring_h.append(round(m_rg / tot * 100))
            op_ring_h.append(100 - round(m_rg / tot * 100))
        else:
            my_ring_h.append(50); op_ring_h.append(50)

    my_ring = my_ring_h
    op_ring = op_ring_h

    # ── CONTROL (composite: 35% agg + 35% ring + 20% pace + 10% guard) ──────
    my_ctrl, op_ctrl = [], []
    for s in range(duration_s):
        m = (0.35 * my_agg[s] + 0.35 * my_ring[s]
             + 0.20 * my_pace[s] + 0.10 * my_guard[s]) / 100
        o = (0.35 * op_agg[s] + 0.35 * op_ring[s]
             + 0.20 * op_pace[s] + 0.10 * op_guard[s]) / 100
        tot = m + o
        if tot > 0:
            my_ctrl.append(round(m / tot * 100))
            op_ctrl.append(100 - round(m / tot * 100))
        else:
            my_ctrl.append(50); op_ctrl.append(50)

    # ── Summary scores ───────────────────────────────────────────────────────
    def _avg(lst):
        return round(sum(lst) / max(len(lst), 1))

    # Clinch counted from per-frame signals.
    clinch_secs = 0
    for s in range(duration_s):
        fs = by_sec.get(s, [])
        if not fs:
            continue
        if any(f.get("clinch", False) for f in fs):
            clinch_secs += 1

    # ── Punch counts ───────────────────────────────────────────────────────
    def _dedup_punches(key):
        punch_fis = [i for i, s in enumerate(sigs) if s[key]]
        if not punch_fis:
            return 0
        count = 1
        for j in range(1, len(punch_fis)):
            if punch_fis[j] - punch_fis[j - 1] > PUNCH_DEDUP_GAP:
                count += 1
        return count
    my_punch_total = _dedup_punches("my_punch")
    op_punch_total = _dedup_punches("op_punch")
    my_landed = op_landed = 0
    my_head_landed = op_head_landed = 0
    my_body_landed = op_body_landed = 0

    metrics = {
        "my_pace":           my_pace,
        "op_pace":           op_pace,
        "my_aggression_sec": my_agg,
        "op_aggression_sec": op_agg,
        "my_guard_sec":      my_guard,
        "op_guard_sec":      op_guard,
        "my_control_sec":    my_ctrl,
        "op_control_sec":    op_ctrl,
        "my_ring_sec":       my_ring,
        "op_ring_sec":       op_ring,
        "my_aggression":     _avg(my_agg),
        "op_aggression":     _avg(op_agg),
        "my_guard":          _avg(my_guard),
        "op_guard":          _avg(op_guard),
        "my_control":        _avg(my_ctrl),
        "op_control":        _avg(op_ctrl),
        "my_pace_score":     _avg(my_pace),
        "op_pace_score":     _avg(op_pace),
        "my_ring_gen":       _avg(my_ring),
        "op_ring_gen":       _avg(op_ring),
        "my_punches":        my_punch_total,
        "op_punches":        op_punch_total,
        "my_landed":         my_landed,
        "op_landed":         op_landed,
        "my_head_landed":    my_head_landed,
        "op_head_landed":    op_head_landed,
        "my_body_landed":    my_body_landed,
        "op_body_landed":    op_body_landed,
        "clinch_seconds":    clinch_secs,
        "duration_s":        duration_s,
        "frames_enriched":   len(frames),
        "fps_pose":          fps_pose,
    }

    meta = read_meta(sid)
    meta["sam2_metrics"] = metrics
    write_meta(sid, meta)

    print(f"[metrics] SAM2 metrics v2 for {sid}: {duration_s}s  "
          f"ctrl={metrics['my_control']}/{metrics['op_control']}  "
          f"agg={metrics['my_aggression']}/{metrics['op_aggression']}  "
          f"guard={metrics['my_guard']}/{metrics['op_guard']}  "
          f"punches={my_punch_total}/{op_punch_total}  "
          f"ring={metrics['my_ring_gen']}/{metrics['op_ring_gen']}")
    return True


def _run_sam2_subprocess(sid):
    """
    Background thread for the lab pipeline.
    1. Pre-computes YOLO detections on every SAM2-sampled frame (for re-anchoring).
    2. Launches sam2_visualizer.py with the user-confirmed seed frame index.
    """
    meta = read_meta(sid)
    if meta is None:
        return

    for candidate in ("lab_compressed.mp4", "compressed.mp4"):
        _c = sess_dir(sid) / candidate
        if _c.exists():
            vpath = str(_c)
            break
    else:
        vpath = str(sess_dir(sid) / f"original{meta['video_ext']}")

    out_video    = str(sess_dir(sid) / "sam2_test.mp4")
    out_json     = str(sess_dir(sid) / "sam2_test_status.json")
    out_progress = str(sess_dir(sid) / "sam2_test_progress.json")
    out_track    = str(sess_dir(sid) / "sam2_track.json")
    log_path     = str(sess_dir(sid) / "sam2_test.log")

    my_ref = meta.get("my_ref_center")
    op_ref = meta.get("op_ref_center")
    if not my_ref or not op_ref:
        meta["sam2_test_status"] = "error"
        meta["sam2_test_error"]  = "No picker reference points."
        write_meta(sid, meta)
        return

    # TODO(production): restore "small" if _SAM2_SMALL.exists() else "tiny"
    model     = "tiny"
    my_box    = meta.get("lab_my_box")
    op_box    = meta.get("lab_op_box")
    fps_kf    = meta.get("fps", 30.0)
    # TODO(production): restore / 6.0
    stride_kf = meta.get("lab_seed_stride") or max(1, round(fps_kf / 3.0))
    seed_fi   = meta.get("lab_seed_fi")   # user-confirmed SAM2 frame index

    # Cap YOLO scan at 5 minutes from seed frame (mirrors sam2_visualizer cap)
    _MAX_DURATION_S = 300
    _seed_raw_fi = (seed_fi or 0) * stride_kf
    _max_raw_fi  = _seed_raw_fi + int(_MAX_DURATION_S * fps_kf)

    # ── YOLO keyframe scan (used for periodic re-anchoring during SAM2) ───────
    kf_path = str(sess_dir(sid) / "lab_yolo_kf.json")
    try:
        yolo_model = get_model()
        cap_kf = cv2.VideoCapture(vpath)
        fi_kf = si_kf = 0
        yolo_kf: dict = {}
        while True:
            ok, frm = cap_kf.read()
            if not ok:
                break
            if fi_kf >= _max_raw_fi:
                break
            if fi_kf % stride_kf == 0:
                res_kf  = yolo_model(frm, verbose=False)[0]
                dets_kf = []
                if res_kf.boxes is not None:
                    for b_kf in res_kf.boxes:
                        if int(b_kf.cls[0]) == 0:
                            x1k,y1k,x2k,y2k = [int(v) for v in b_kf.xyxy[0].tolist()]
                            conf_k = round(float(b_kf.conf[0]), 3)
                            if conf_k >= 0.30:
                                dets_kf.append([x1k, y1k, x2k, y2k, conf_k])
                yolo_kf[str(si_kf)] = dets_kf
                si_kf += 1
            fi_kf += 1
        cap_kf.release()
        Path(kf_path).write_text(json.dumps(yolo_kf))
    except Exception as _kf_err:
        kf_path = ""   # non-fatal — sam2_visualizer falls back to click-only seeding
        import traceback as _tb
        print(f"[lab] YOLO kf scan failed (non-fatal): {_kf_err}\n{_tb.format_exc()}")

    # ── Build + launch sam2_visualizer ────────────────────────────────────────
    cmd = [
        str(_SAM2_VENV), str(_SAM2_VIS_SCRIPT),
        "--video",    vpath,
        "--my_pt",    f"{my_ref[0]},{my_ref[1]}",
        "--op_pt",    f"{op_ref[0]},{op_ref[1]}",
        "--out",      out_video,
        "--status",   out_json,
        "--progress", out_progress,
        "--model",    model,
        "--max_dim",  "1024",
    ]
    if my_box and op_box:
        cmd += ["--my_box", f"{my_box[0]},{my_box[1]},{my_box[2]},{my_box[3]}"]
        cmd += ["--op_box", f"{op_box[0]},{op_box[1]},{op_box[2]},{op_box[3]}"]
    if kf_path and Path(kf_path).exists():
        cmd += ["--yolo_kf", kf_path]
        cmd += ["--stride",  str(stride_kf)]
    if seed_fi is not None:
        cmd += ["--seed_fi", str(seed_fi)]  # start from user-confirmed frame
    cmd += ["--track_json", out_track]       # per-frame bbox sidecar for swap detection

    stderr_lines = []
    def _drain(proc, log_file):
        with open(log_file, "w") as lf:
            for line in proc.stderr:
                lf.write(line)
                lf.flush()
                stderr_lines.append(line)

    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        drain_thread = threading.Thread(
            target=_drain, args=(proc, log_path), daemon=True)
        drain_thread.start()

        try:
            proc.wait(timeout=SAM2_TEST_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            drain_thread.join(timeout=5)
            meta["sam2_test_status"] = "error"
            meta["sam2_test_error"]  = f"Timed out after {SAM2_TEST_TIMEOUT_S // 60} min."
            write_meta(sid, meta)
            return

        drain_thread.join(timeout=10)

        if Path(out_json).exists():
            result = json.loads(Path(out_json).read_text())
            if result.get("ok"):
                meta["sam2_test_model"]       = result.get("model")
                meta["sam2_test_fps"]         = result.get("fps_effective")
                meta["sam2_test_my_tracked"]  = result.get("my_tracked")
                meta["sam2_test_op_tracked"]  = result.get("op_tracked")
                meta["sam2_test_frames"]      = result.get("frame_count")
                meta["sam2_test_scale"]       = result.get("scale")

                # ── Pose enrichment + metrics ────────────────────────────
                meta["sam2_test_status"] = "enriching"
                write_meta(sid, meta)
                # Write progress hint for the polling UI
                try:
                    Path(out_progress).write_text(json.dumps({
                        "pct": 100, "frame": result.get("frame_count", 0),
                        "total": result.get("frame_count", 0),
                        "fps": 0, "device": result.get("device", ""),
                        "phase": "enriching",
                    }))
                except Exception:
                    pass
                try:
                    _run_pose_enrichment(sid)
                    _compute_sam2_metrics(sid)
                except Exception as _enrich_err:
                    print(f"[lab] Enrichment/metrics failed (non-fatal): "
                          f"{_enrich_err}")
                meta = read_meta(sid)
                meta["sam2_test_status"] = "done"
            else:
                meta["sam2_test_status"] = "error"
                meta["sam2_test_error"]  = result.get("error", "Unknown error")
        else:
            last_stderr = "".join(stderr_lines[-30:])
            meta["sam2_test_status"] = "error"
            meta["sam2_test_error"]  = (
                f"Process exited (rc={proc.returncode}) without writing output.\n"
                f"Last log:\n{last_stderr}")

    except Exception as e:
        meta["sam2_test_status"] = "error"
        meta["sam2_test_error"]  = str(e)

    write_meta(sid, meta)


@app.route("/api/status/<sid>")
def api_status(sid):
    meta = read_meta(sid)
    if meta is None: return jsonify(error="not found"), 404
    return jsonify(
        status              = meta["status"],
        progress            = meta.get("progress", 0),
        metrics             = meta.get("metrics"),
        annotated_ready     = meta.get("annotated_ready", False),
        error               = meta.get("error"),
        punch_events_count  = meta.get("punch_events_count"),
        audio_diag          = meta.get("audio_diag"),
        audio_error         = meta.get("audio_error"),
        overlay_error       = meta.get("overlay_error"),
        sam2_status         = meta.get("sam2_status"),
        sam2_model          = meta.get("sam2_model"),
        sam2_fps_eff        = meta.get("sam2_fps_eff"),
        sam2_my_tracked     = meta.get("sam2_my_tracked"),
        sam2_op_tracked     = meta.get("sam2_op_tracked"),
        sam2_frames         = meta.get("sam2_frames"),
        sam2_test_status    = meta.get("sam2_test_status"),
        sam2_test_error     = meta.get("sam2_test_error"),
        sam2_test_model     = meta.get("sam2_test_model"),
        sam2_test_fps       = meta.get("sam2_test_fps"),
        sam2_test_my_tracked= meta.get("sam2_test_my_tracked"),
        sam2_test_op_tracked= meta.get("sam2_test_op_tracked"),
        sam2_test_frames    = meta.get("sam2_test_frames"),
        sam2_test_progress  = _read_sam2_progress(sid),
        analysis_status     = meta.get("analysis_status"),
    )


@app.route("/thumbnail/<sid>")
def thumbnail(sid):
    for name in ("preview.jpg", "first_frame.jpg"):
        p = sess_dir(sid)/name
        if p.exists(): return send_file(str(p), mimetype="image/jpeg")
    return "Not found", 404


@app.route("/video/<sid>/<kind>")
def video(sid, kind):
    meta = read_meta(sid)
    if meta is None: return "Not found", 404
    paths = dict(
        highlights = sess_dir(sid)/"highlights.mp4",
        annotated  = sess_dir(sid)/"annotated.mp4",
        sam2_test  = sess_dir(sid)/"sam2_test.mp4",
    )
    p = paths.get(kind, sess_dir(sid)/f"original{meta['video_ext']}")
    if not p.exists(): return "Not found", 404
    return _range_response(p, "video/mp4")


# ─── SAM2 diagnostic test ─────────────────────────────────────────────────────

_SAM2_VIS_SCRIPT = BASE / "sam2_visualizer.py"

# Per-clip SAM2 test timeout. Raised from 30 min → 2 h so slow laptops can
# finish long clips without aborting. The test runs on a sampled subset
# (stride-based) so even 3-round sparring clips should clear this budget;
# anything that actually blows through 2 h is genuinely stuck, not slow.
SAM2_TEST_TIMEOUT_S = 7200

def _run_sam2_test(sid):
    """Background thread: run sam2_visualizer.py and store results in meta."""
    meta = read_meta(sid)
    if meta is None:
        return

    # Lab sessions have a pre-compressed video stored as lab_compressed.mp4.
    # Main-pipeline sessions may have compressed.mp4 from a previous analysis.
    # Fall back to the original upload if neither exists.
    for candidate in ("lab_compressed.mp4", "compressed.mp4"):
        _c = sess_dir(sid) / candidate
        if _c.exists():
            vpath = str(_c)
            break
    else:
        vpath = str(sess_dir(sid) / f"original{meta['video_ext']}")
    out_video = str(sess_dir(sid) / "sam2_test.mp4")
    out_json  = str(sess_dir(sid) / "sam2_test_status.json")
    my_ref    = meta.get("my_ref_center")
    op_ref    = meta.get("op_ref_center")

    if not my_ref or not op_ref:
        meta["sam2_test_status"] = "error"
        meta["sam2_test_error"]  = "Picker not completed — no boxer reference points."
        write_meta(sid, meta)
        return

    # TODO(production): restore "small" if _SAM2_SMALL.exists() else "tiny"
    model        = "tiny"
    out_progress = str(sess_dir(sid) / "sam2_test_progress.json")
    log_path     = str(sess_dir(sid) / "sam2_test.log")
    my_box       = meta.get("lab_my_box")   # [x1,y1,x2,y2] or None
    op_box       = meta.get("lab_op_box")

    # ── Pre-compute YOLO detections on every SAM2-sampled frame ──────────────
    # This gives sam2_visualizer a JSON of YOLO boxes at every stride-th frame
    # so it can periodically re-anchor SAM2 when YOLO is confident — preventing
    # the tracker from drifting onto ring ropes or the ring canvas over time.
    kf_path = str(sess_dir(sid) / "lab_yolo_kf.json")
    try:
        total_f_kf = meta.get("total_frames", 0)
        # M2 fix: compute stride here and pass it explicitly to sam2_visualizer
        # via --stride so both sides always agree.  Previously Flask computed
        # stride from meta["total_frames"] and sam2_visualizer computed it from
        # cap.get(CAP_PROP_FRAME_COUNT) — some containers return slightly different
        # values from these two calls, causing the keyframe index map to shift.
        stride_kf  = max(1, math.ceil(max(total_f_kf, 1) / 100))  # mirrors MAX_SAM2_FRAMES
        yolo_model = get_model()
        cap_kf     = cv2.VideoCapture(vpath)
        fi_kf = si_kf = 0
        yolo_kf: dict = {}
        while True:
            ok, frm = cap_kf.read()
            if not ok:
                break
            if fi_kf % stride_kf == 0:
                res_kf  = yolo_model(frm, verbose=False)[0]
                dets_kf = []
                if res_kf.boxes is not None:
                    for b_kf in res_kf.boxes:
                        if int(b_kf.cls[0]) == 0:
                            x1k,y1k,x2k,y2k = [int(v) for v in b_kf.xyxy[0].tolist()]
                            conf_k = round(float(b_kf.conf[0]), 3)
                            if conf_k >= 0.30:
                                dets_kf.append([x1k, y1k, x2k, y2k, conf_k])
                yolo_kf[str(si_kf)] = dets_kf
                si_kf += 1
            fi_kf += 1
        cap_kf.release()
        Path(kf_path).write_text(json.dumps(yolo_kf))
    except Exception as _kf_err:
        # Non-fatal — sam2_visualizer falls back to point-only seeding
        kf_path = ""
        import traceback as _tb
        print(f"[lab] YOLO keyframe precompute failed: {_kf_err}\n{_tb.format_exc()}")

    try:
        cmd = [
            str(_SAM2_VENV), str(_SAM2_VIS_SCRIPT),
            "--video",    vpath,
            "--my_pt",    f"{my_ref[0]},{my_ref[1]}",
            "--op_pt",    f"{op_ref[0]},{op_ref[1]}",
            "--out",      out_video,
            "--status",   out_json,
            "--progress", out_progress,
            "--model",    model,
            "--max_dim",  "1024",
        ]
        if my_box and op_box:
            cmd += ["--my_box", f"{my_box[0]},{my_box[1]},{my_box[2]},{my_box[3]}"]
            cmd += ["--op_box", f"{op_box[0]},{op_box[1]},{op_box[2]},{op_box[3]}"]
        if kf_path:
            cmd += ["--yolo_kf", kf_path]
            cmd += ["--stride",  str(stride_kf)]   # M2 fix: explicit stride

        # Use Popen + a drain thread instead of subprocess.run(capture_output=True).
        # With a long-running process that produces lots of stderr, capture_output
        # blocks when the OS pipe buffer fills (~64 KB on Linux/macOS) — the child
        # waits for the parent to drain the pipe, the parent waits for the child to
        # exit: deadlock.  Draining in a background thread avoids this.
        stderr_lines = []
        def _drain(proc, log_file):
            with open(log_file, "w") as lf:
                for line in proc.stderr:
                    lf.write(line)
                    lf.flush()
                    stderr_lines.append(line)

        proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
        )
        drain_thread = threading.Thread(target=_drain, args=(proc, log_path), daemon=True)
        drain_thread.start()

        try:
            proc.wait(timeout=SAM2_TEST_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            drain_thread.join(timeout=5)
            meta["sam2_test_status"] = "error"
            meta["sam2_test_error"]  = f"Timed out after {SAM2_TEST_TIMEOUT_S // 60} min."
            write_meta(sid, meta)
            return

        drain_thread.join(timeout=10)

        if Path(out_json).exists():
            result = json.loads(Path(out_json).read_text())
            if result.get("ok"):
                meta["sam2_test_status"]     = "done"
                meta["sam2_test_model"]       = result.get("model")
                meta["sam2_test_fps"]         = result.get("fps_effective")
                meta["sam2_test_my_tracked"]  = result.get("my_tracked")
                meta["sam2_test_op_tracked"]  = result.get("op_tracked")
                meta["sam2_test_frames"]      = result.get("frame_count")
            else:
                meta["sam2_test_status"] = "error"
                meta["sam2_test_error"]  = result.get("error", "Unknown error")
        else:
            last_stderr = "".join(stderr_lines[-30:])
            meta["sam2_test_status"] = "error"
            meta["sam2_test_error"]  = (
                f"Process exited (rc={proc.returncode}) without writing output.\n"
                f"Last log:\n{last_stderr}"
            )

    except Exception as e:
        meta["sam2_test_status"] = "error"
        meta["sam2_test_error"]  = str(e)

    write_meta(sid, meta)


@app.route("/api/sam2_test/<sid>", methods=["POST"])
def api_sam2_test(sid):
    """Kick off a SAM2 diagnostic visualisation run."""
    meta = read_meta(sid)
    if meta is None:
        return jsonify(error="session not found"), 404
    if not sam2_available():
        return jsonify(error="SAM2 not installed — run setup_sam2.sh first"), 400
    if meta.get("sam2_test_status") == "running":
        return jsonify(status="running"), 200   # already in progress

    meta["sam2_test_status"] = "running"
    meta["sam2_test_error"]  = None
    write_meta(sid, meta)

    threading.Thread(target=_run_sam2_test, args=(sid,), daemon=True).start()
    return jsonify(status="running"), 200


@app.route("/delete/<sid>", methods=["POST"])
def delete_session(sid):
    import shutil
    d = sess_dir(sid)
    meta = read_meta(sid)
    is_lab = meta.get("lab_mode", False) if meta else False
    if d.exists():
        shutil.rmtree(str(d))
    return redirect(url_for("lab_index") if is_lab else url_for("index"))


@app.route("/heatmap/<sid>")
def heatmap(sid):
    p = sess_dir(sid)/"heatmap.png"
    if not p.exists(): return "Not found", 404
    return send_file(str(p), mimetype="image/png")


# ─── Range-request video helper ──────────────────────────────────────────────

def _range_response(path: Path, mimetype: str):
    file_size    = path.stat().st_size
    range_header = request.headers.get("Range")
    if not range_header:
        return send_file(str(path), mimetype=mimetype)
    m = re.match(r"bytes=(\d+)-(\d*)", range_header)
    if not m:
        return send_file(str(path), mimetype=mimetype)
    b1 = int(m.group(1))
    b2 = int(m.group(2)) if m.group(2) else file_size - 1
    b2 = min(b2, file_size - 1)
    length = b2 - b1 + 1
    def gen():
        with open(path, "rb") as fh:
            fh.seek(b1)
            rem = length
            while rem > 0:
                chunk = fh.read(min(65536, rem))
                if not chunk: break
                rem -= len(chunk)
                yield chunk
    rv = Response(gen(), 206, mimetype=mimetype, direct_passthrough=True)
    rv.headers["Content-Range"]  = f"bytes {b1}-{b2}/{file_size}"
    rv.headers["Accept-Ranges"]  = "bytes"
    rv.headers["Content-Length"] = str(length)
    return rv


# ─── Analysis thread ─────────────────────────────────────────────────────────

def _compress_video(src: str, dst: str, max_width: int = 1280) -> bool:
    """
    Resize the video so its width ≤ max_width (height scaled proportionally).
    Uses OpenCV — no ffmpeg dependency.  Returns True on success, False on error.
    This runs once before analysis and cuts memory + CPU cost for all downstream
    steps (YOLO, SAM2, annotated writer).
    """
    cap = cv2.VideoCapture(src)
    fps = _safe(cap.get(cv2.CAP_PROP_FPS), 30.0)
    ok, frame = cap.read()
    if not ok:
        cap.release()
        return False
    fh, fw = frame.shape[:2]

    if fw <= max_width:
        cap.release()
        return False   # already small enough — no need to compress

    scale  = max_width / fw
    new_fw = max_width
    new_fh = int(fh * scale)
    new_fh = new_fh if new_fh % 2 == 0 else new_fh - 1

    writer = cv2.VideoWriter(
        dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (new_fw, new_fh))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ok, f = cap.read()
        if not ok:
            break
        writer.write(cv2.resize(f, (new_fw, new_fh), interpolation=cv2.INTER_AREA))

    cap.release()
    writer.release()
    return True


def analyse(sid):
    meta = read_meta(sid)
    try:
        orig_path = str(sess_dir(sid) / f"original{meta['video_ext']}")
        vpath = orig_path

        cap   = cv2.VideoCapture(vpath)
        fps   = _safe(cap.get(cv2.CAP_PROP_FPS), 30.0)
        total = int(_safe(cap.get(cv2.CAP_PROP_FRAME_COUNT), 0))

        # Derive frame dimensions from the actual decoded frame, not container metadata
        ok, _first = cap.read()
        if not ok:
            raise RuntimeError("Cannot read video — file may be corrupted or unsupported.")
        fh, fw = _first.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # rewind to frame 0

        model = get_model()

        # ── SAM2 tracking (optional, replaces Kalman when available) ─────────
        # Attempt to run SAM2 first.  If it succeeds, sam2_boxes[str(fi)] holds
        # {"my": [x1,y1,x2,y2]|null, "op": [x1,y1,x2,y2]|null} for every frame.
        # The Kalman loop below checks for this dict and skips its own matching.
        sam2_boxes = None
        my_ref  = meta.get("my_ref_center")
        op_ref  = meta.get("op_ref_center")
        if sam2_available() and my_ref and op_ref:
            meta["progress"]    = 2
            meta["sam2_status"] = "running"
            write_meta(sid, meta)
            sam2_result = run_sam2_tracker(vpath, my_ref, op_ref)
            if sam2_result:
                sam2_boxes = sam2_result.get("boxes", {})
                # Count frames where each boxer was successfully tracked
                _s2_my_tracked = sum(1 for v in sam2_boxes.values() if v.get("my") is not None)
                _s2_op_tracked = sum(1 for v in sam2_boxes.values() if v.get("op") is not None)
                meta["sam2_model"]      = sam2_result.get("model")
                meta["sam2_fps_eff"]    = sam2_result.get("fps_effective")
                meta["sam2_status"]     = "ok"
                meta["sam2_my_tracked"] = _s2_my_tracked
                meta["sam2_op_tracked"] = _s2_op_tracked
                meta["sam2_frames"]     = sam2_result.get("frame_count", 0)
            else:
                meta["sam2_status"] = "failed"
            write_meta(sid, meta)

        # ── Seed Kalman trackers from picker selections ───────────────────────
        # (used as fallback when SAM2 is unavailable, and for KPS / punch logic
        #  which still runs on YOLO detections regardless of tracking source)
        # my_ref / op_ref already assigned above for SAM2; reuse them here.
        my_hist = meta.get("my_hist")
        op_hist = meta.get("op_hist")
        my_idx  = int(meta.get("boxer_index") or 0)
        op_idx  = int(meta.get("op_index")    or -1)

        picker_boxes = meta.get("detected_boxes", [])

        def _fallback_box(cx, cy):
            """Estimate a box from a centre point using typical boxer proportions."""
            bw = fw * 0.15; bh = fh * 0.40
            return [int(cx-bw/2), int(cy-bh/2), int(cx+bw/2), int(cy+bh/2)]

        if 0 <= my_idx < len(picker_boxes):
            my_init = picker_boxes[my_idx]
        elif my_ref:
            my_init = _fallback_box(my_ref[0], my_ref[1])
        else:
            my_init = _fallback_box(fw * 0.25, fh * 0.5)

        if 0 <= op_idx < len(picker_boxes):
            op_init = picker_boxes[op_idx]
        elif op_ref:
            op_init = _fallback_box(op_ref[0], op_ref[1])
        else:
            op_init = _fallback_box(fw * 0.75, fh * 0.5)

        my_trk = KalmanBoxTracker(my_init, hist=my_hist)
        op_trk = KalmanBoxTracker(op_init, hist=op_hist)

        # Clean copies of picker histograms — never blended via EMA, so they
        # remain an uncorrupted appearance anchor.  Used as a re-ID fallback
        # when the EMA histogram has drifted too far from the boxer's real look.
        my_clean_hist = list(my_hist) if my_hist else None
        op_clean_hist = list(op_hist) if op_hist else None

        MAX_DIST     = max(fw, fh) * 0.50
        HIST_REFRESH = 20      # YOLO frames between histogram blend updates
        SIZE_EMA     = 0.92    # EMA smoothing for expected boxer height

        def match_score(pred_box, cand_box, trk_hist, c_hist, exp_h):
            """
            28% position (IoU / normalised-distance) + 72% appearance, scaled by
            size plausibility.  Colour dominates but position still provides enough
            pull to recover from histogram drift or lighting changes.
            """
            iou = _box_iou(pred_box, cand_box)
            pos = (iou if iou > 0 else
                   max(0., 1.0 - _dist(_cx_cy(pred_box), _cx_cy(cand_box)) / MAX_DIST) * 0.25)
            app = _hist_sim(trk_hist, c_hist) if trk_hist else 0.5
            sz  = _size_score(cand_box, exp_h)
            return (0.28 * pos + 0.72 * app) * sz

        # ── Annotated video writer ────────────────────────────────────────────
        ann_path   = str(sess_dir(sid)/"annotated.mp4")
        ann_writer = cv2.VideoWriter(
            ann_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh))

        frames           = []
        prev_fd          = None
        fi               = 0
        last_yolo_my_kps = None    # keypoints reused on YOLO-skipped frames
        last_yolo_op_kps = None
        expected_my_h    = None    # EMA of my boxer's bbox height (size filter)
        expected_op_h    = None
        my_reid_cooldown   = 0   # frames remaining in post-re-ID settling window
        op_reid_cooldown   = 0   # (metrics suppressed while > 0)
        my_clinch_cooldown = 0   # post-clinch grey frames (tracks re-emerge noisy)
        op_clinch_cooldown = 0
        clinch             = False  # current-frame clinch state (drives cooldown)
        swap_votes         = 0   # consecutive YOLO frames voting for a swap
        POST_CLINCH_FRAMES = ANALYSIS_STRIDE * 20   # ~1 s of grey after clinch ends
        _last_pos_ms       = 0.0  # last known frame timestamp (ms) — for VFR correction
        _prev_gray         = None  # for scene-cut detection
        SCENE_CUT_THRESH   = 35.0  # mean-absolute-diff threshold (0-255 scale)

        while True:
            ok, frame = cap.read()
            if not ok: break
            _last_pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

            is_yolo_frame = (fi % ANALYSIS_STRIDE == 0)

            # Always advance Kalman at full video fps
            my_pred = my_trk.predict()
            op_pred = op_trk.predict()

            if is_yolo_frame:
                # ── Scene-cut detection ───────────────────────────────────────
                # Compilation/highlight videos contain hard cuts to completely
                # different fights or gyms.  When the mean absolute pixel diff
                # between consecutive YOLO frames exceeds the threshold we treat
                # it as a scene boundary and wipe the tracker state so the
                # trackers can re-acquire the fighters in the new scene instead
                # of chasing histograms from the previous, unrelated clip.
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if _prev_gray is not None and fi > 0:
                    _mad = float(np.mean(np.abs(
                        curr_gray.astype(np.float32) -
                        _prev_gray.astype(np.float32)
                    )))
                    if _mad > SCENE_CUT_THRESH:
                        # Reset both trackers to "lost" state
                        my_trk.coast     = REID_AFTER
                        op_trk.coast     = REID_AFTER
                        # Restore clean picker histograms — EMA may have drifted
                        # to the wrong person's appearance in the previous scene
                        my_hist          = list(my_clean_hist) if my_clean_hist else None
                        op_hist          = list(op_clean_hist) if op_clean_hist else None
                        my_trk.hist      = my_hist
                        op_trk.hist      = op_hist
                        # Clear size priors — new scene may have different scale
                        expected_my_h    = None
                        expected_op_h    = None
                        # Reset ancillary state
                        clinch           = False
                        my_clinch_cooldown = 0
                        op_clinch_cooldown = 0
                        my_reid_cooldown   = 0
                        op_reid_cooldown   = 0
                        swap_votes         = 0
                        last_yolo_my_kps   = None
                        last_yolo_op_kps   = None
                _prev_gray = curr_gray

                # ── YOLO detection ────────────────────────────────────────────
                res = model(frame, verbose=False)[0]
                candidates = []
                if res.boxes is not None:
                    for bi, box in enumerate(res.boxes):
                        if int(box.cls[0]) != 0: continue
                        bx = [int(v) for v in box.xyxy[0].tolist()]
                        # Reject small background detections (seated crowd, distant
                        # people behind ropes).  3.0% of frame area keeps standing
                        # ring-level fighters (including those near frame edges)
                        # while cutting clearly small background people.
                        if (bx[2]-bx[0])*(bx[3]-bx[1]) < fw*fh*0.020: continue
                        kps = (res.keypoints.xy[bi].cpu().numpy()
                               if res.keypoints is not None and bi < len(res.keypoints)
                               else None)
                        candidates.append((bx, kps))

                # Histograms computed once per YOLO frame, reused everywhere
                cand_hists = [_extract_hist_body(frame, bx, kps) for bx, kps in candidates]

                # ── Spatial pre-filter (crowded scenes) ───────────────────────
                # When ≥3 people are detected, trim the candidate list to those
                # within 3.5 × expected boxer height of either Kalman prediction.
                # Background spectators and coaches far from the fighting pair are
                # dropped here, well before colour-histogram matching.  We keep at
                # least 2 candidates even if none pass, as a safety net.
                # Skip right after a scene cut — both Kalman predictions point to
                # the previous scene and would filter the correct new-scene fighters.
                _both_coasting = (my_trk.coast >= REID_AFTER and
                                  op_trk.coast >= REID_AFTER)
                if len(candidates) > 2 and not _both_coasting:
                    _ref_h = (expected_my_h or expected_op_h or fh * 0.40)
                    _max_d = _ref_h * 3.5
                    _p_my  = _cx_cy(my_pred)
                    _p_op  = _cx_cy(op_pred)
                    _prox_order = sorted(
                        range(len(candidates)),
                        key=lambda i: min(
                            _dist(_cx_cy(candidates[i][0]), _p_my),
                            _dist(_cx_cy(candidates[i][0]), _p_op)
                        )
                    )
                    # Keep any candidate within distance budget; guarantee ≥ 2
                    _kept = [i for i in _prox_order
                             if min(_dist(_cx_cy(candidates[i][0]), _p_my),
                                    _dist(_cx_cy(candidates[i][0]), _p_op)) <= _max_d]
                    if len(_kept) < 2:
                        _kept = _prox_order[:2]   # fallback: 2 closest regardless
                    candidates = [candidates[i] for i in _kept]
                    cand_hists = [cand_hists[i]  for i in _kept]

                # ── Assignment ───────────────────────────────────────────────
                used = set()

                def best_unmatched(pred_box, trk_hist, exp_h, clinch_cd=0):
                    best_i, best_s = None, MIN_SCORE
                    # During post-clinch cooldown we tighten the colour gate so a
                    # cornerman who stepped close cannot steal a tracker with a
                    # mediocre histogram score.  Normal threshold resumes once the
                    # cooldown expires (trackers have had time to re-separate).
                    # 0.35 baseline: filters obvious colour mismatches while still
                    # accepting fighters under varying lighting or after minor
                    # clothing/occlusion changes.  The hard distance cap (below)
                    # is the primary defence against spectator contamination —
                    # the colour gate is a secondary screen, not the gatekeeper.
                    hist_floor = 0.52 if clinch_cd > 0 else 0.35
                    # Hard spatial cap: candidate centre must be within 2.5 × boxer
                    # height of the Kalman prediction.  Without this, a drifted
                    # prediction lands near ringside spectators whose colour score
                    # accidentally beats the real (far-away) fighter.
                    max_jump = (exp_h or fh * 0.40) * 2.5
                    p_cx, p_cy = _cx_cy(pred_box)
                    for i, (bx, _) in enumerate(candidates):
                        if i in used: continue
                        # Hard distance gate.
                        c_cx, c_cy = _cx_cy(bx)
                        if math.hypot(p_cx - c_cx, p_cy - c_cy) > max_jump:
                            continue
                        # Colour gate: reject clear mismatches.
                        if trk_hist and cand_hists[i]:
                            if _hist_sim(trk_hist, cand_hists[i]) < hist_floor:
                                continue
                        # Height cap: reject candidates >60% taller than expected
                        # (referee stepping very close to camera appears giant).
                        if exp_h and (bx[3] - bx[1]) > exp_h * 1.60:
                            continue
                        s = match_score(pred_box, bx, trk_hist, cand_hists[i], exp_h)
                        if s > best_s:
                            best_s, best_i = s, i
                    return best_i

                my_i = best_unmatched(my_pred, my_hist, expected_my_h, my_clinch_cooldown)
                if my_i is not None: used.add(my_i)
                op_i = best_unmatched(op_pred, op_hist, expected_op_h, op_clinch_cooldown)
                if op_i is not None: used.add(op_i)

                # ── Swap-check with hysteresis ───────────────────────────────
                # Require SWAP_CONFIRM consecutive YOLO frames all voting for
                # the swap before committing.  Single-frame flip-flops during
                # close exchanges no longer corrupt both histograms at once.
                SWAP_CONFIRM = 3
                if (my_i is not None and op_i is not None and
                        my_hist is not None and op_hist is not None):
                    s_str = (_hist_sim(my_hist, cand_hists[my_i]) +
                             _hist_sim(op_hist, cand_hists[op_i]))
                    s_swp = (_hist_sim(my_hist, cand_hists[op_i]) +
                             _hist_sim(op_hist, cand_hists[my_i]))
                    if s_swp > s_str + 0.08:
                        swap_votes += 1
                        if swap_votes >= SWAP_CONFIRM:
                            my_i, op_i = op_i, my_i
                            swap_votes = 0
                    else:
                        swap_votes = max(0, swap_votes - 1)

                # ── Update / miss ─────────────────────────────────────────────
                # Resolve current-frame boxes FIRST (before histogram blend)
                # so we can do a pre-clinch check with THIS frame's geometry.
                if my_i is not None:
                    my_trk.update(candidates[my_i][0])
                    my_box, my_kps = candidates[my_i]
                    last_yolo_my_kps = my_kps
                    bh = my_box[3] - my_box[1]
                    expected_my_h = (bh if expected_my_h is None
                                     else SIZE_EMA*expected_my_h + (1-SIZE_EMA)*bh)
                else:
                    my_trk.miss()
                    my_box, my_kps = my_pred, last_yolo_my_kps
                    if my_trk.coast >= MAX_COAST:
                        my_trk.coast = REID_AFTER

                if op_i is not None:
                    op_trk.update(candidates[op_i][0])
                    op_box, op_kps = candidates[op_i]
                    last_yolo_op_kps = op_kps
                    bh = op_box[3] - op_box[1]
                    expected_op_h = (bh if expected_op_h is None
                                     else SIZE_EMA*expected_op_h + (1-SIZE_EMA)*bh)
                else:
                    op_trk.miss()
                    op_box, op_kps = op_pred, last_yolo_op_kps
                    if op_trk.coast >= MAX_COAST:
                        op_trk.coast = REID_AFTER

                # ── Pre-clinch check (current frame's boxes) ─────────────────
                # Computed HERE — after my_box/op_box are resolved from YOLO
                # but BEFORE the histogram EMA blend — so the very first
                # clinch frame is caught.  The per-frame clinch detection
                # later in the loop uses the same logic for cooldowns.
                _eh = expected_my_h or expected_op_h or fh * 0.40
                _pre_clinch = (_box_iou(my_box, op_box) > 0.20 or
                               _dist(_cx_cy(my_box), _cx_cy(op_box)) < _eh * 0.55)

                # ── Histogram EMA blend (protected by pre-clinch) ────────────
                if (my_i is not None and my_hist and cand_hists[my_i]
                        and fi % HIST_REFRESH == 0
                        and my_reid_cooldown == 0
                        and not _pre_clinch):
                    my_hist = [0.92*a + 0.08*b
                               for a, b in zip(my_hist, cand_hists[my_i])]
                    my_trk.hist = my_hist

                if (op_i is not None and op_hist and cand_hists[op_i]
                        and fi % HIST_REFRESH == 0
                        and op_reid_cooldown == 0
                        and not _pre_clinch):
                    op_hist = [0.92*a + 0.08*b
                               for a, b in zip(op_hist, cand_hists[op_i])]
                    op_trk.hist = op_hist

                # ── Appearance re-ID for long-coasting trackers ───────────────
                # A candidate must pass ALL of the following gates:
                #   (1) Colour similarity ≥ REID_MIN (Bhattacharyya)
                #   (2) Centre within 1.2× expected boxer height of Kalman prediction
                #       (velocity damping keeps prediction near last known position)
                #   (3) Box height within 45% of EMA expected height
                #   (4) Box area ≥ 1.5% of frame area  →  rejects seated spectators
                #   (5) If the OTHER boxer is live, candidate must be within 2.0×
                #       expected height of that boxer's centre  →  rejects referee/
                #       judges standing metres away during a clinch
                unmatched = [(i, bx, kps)
                             for i, (bx, kps) in enumerate(candidates) if i not in used]

                _REID_DIST_FACTOR  = 3.0    # max dist from Kalman pred: wide search —
                                            # boxer may have genuinely moved; stricter
                                            # REID_MIN + mutual-prox check compensate
                _LIVE_DIST_FACTOR  = 3.5    # max dist from live boxer:  3.5× boxer h
                _MIN_AREA_FRAC     = 0.020  # candidate area ≥ 2.0% of frame area

                min_cand_area = fw * fh * _MIN_AREA_FRAC

                # Live-boxer centres (None when that tracker is also coasting)
                live_my_cx = live_my_cy = live_op_cx = live_op_cy = None
                if my_i is not None:
                    live_my_cx = (my_box[0] + my_box[2]) / 2
                    live_my_cy = (my_box[1] + my_box[3]) / 2
                if op_i is not None:
                    live_op_cx = (op_box[0] + op_box[2]) / 2
                    live_op_cy = (op_box[1] + op_box[3]) / 2

                def _reid_scan(pred_box, trk_hist, exp_h, live_cx, live_cy):
                    """Return index of best unmatched candidate, or None.

                    Two spatial modes:
                    (A) Opponent visible  → Gate 2 (Kalman pred) is SKIPPED.
                        The Kalman prediction may have drifted during a clinch,
                        but the lost boxer MUST emerge adjacent to the visible
                        opponent — Gate 5 (proximity to opponent) is the only
                        reliable spatial anchor in this case.
                    (B) Both boxers lost  → Gate 2 (Kalman pred) is the only
                        spatial anchor; Gate 5 is not available.
                    """
                    best_i, best_sim = None, REID_MIN
                    p_cx = (pred_box[0] + pred_box[2]) / 2
                    p_cy = (pred_box[1] + pred_box[3]) / 2
                    max_pred_d = (exp_h or fh * 0.40) * _REID_DIST_FACTOR
                    max_live_d = (exp_h or fh * 0.40) * _LIVE_DIST_FACTOR
                    for i, bx, _ in unmatched:
                        c_cx = (bx[0] + bx[2]) / 2; c_cy = (bx[1] + bx[3]) / 2
                        # Gate 2: distance from Kalman prediction.
                        # Skipped when the opponent is visible — in that case the
                        # prediction may have drifted away from the clinch area
                        # while the real boxer is right next to the opponent.
                        if live_cx is None:
                            if ((p_cx-c_cx)**2 + (p_cy-c_cy)**2)**0.5 > max_pred_d:
                                continue
                        # Gate 3: height similarity
                        if exp_h and abs((bx[3]-bx[1]) - exp_h) / exp_h > 0.45:
                            continue
                        # Gate 4: minimum box area (rejects seated spectators)
                        if (bx[2]-bx[0]) * (bx[3]-bx[1]) < min_cand_area:
                            continue
                        # Gate 5: near the live opponent (if visible).
                        # When opponent is visible this is the primary spatial
                        # anchor — a boxer exiting a clinch is always adjacent.
                        if live_cx is not None:
                            if ((live_cx-c_cx)**2 + (live_cy-c_cy)**2)**0.5 > max_live_d:
                                continue
                        # Gate 1: colour similarity
                        sim = _hist_sim(trk_hist, cand_hists[i])
                        if sim > best_sim: best_sim, best_i = sim, i
                    return best_i

                my_reid_cand = None
                op_reid_cand = None

                if my_i is None and my_trk.coast >= REID_AFTER and my_hist and unmatched:
                    my_reid_cand = _reid_scan(my_pred, my_hist, expected_my_h,
                                              live_op_cx, live_op_cy)
                    # Fallback: if the EMA histogram has drifted too far, retry
                    # with the clean picker histogram (never contaminated by EMA).
                    if my_reid_cand is None and my_clean_hist:
                        my_reid_cand = _reid_scan(my_pred, my_clean_hist,
                                                  expected_my_h,
                                                  live_op_cx, live_op_cy)

                if op_i is None and op_trk.coast >= REID_AFTER and op_hist and unmatched:
                    # Pass live_my as the opponent anchor; if my_reid_cand was just
                    # found use that as an updated live anchor for op's search.
                    lx = live_my_cx
                    ly = live_my_cy
                    if my_reid_cand is not None and lx is None:
                        lx = (candidates[my_reid_cand][0][0] +
                              candidates[my_reid_cand][0][2]) / 2
                        ly = (candidates[my_reid_cand][0][1] +
                              candidates[my_reid_cand][0][3]) / 2
                    op_reid_cand = _reid_scan(op_pred, op_hist, expected_op_h, lx, ly)
                    if op_reid_cand is None and op_clean_hist:
                        op_reid_cand = _reid_scan(op_pred, op_clean_hist,
                                                  expected_op_h, lx, ly)

                # ── Mutual-proximity sanity check ─────────────────────────────
                # When BOTH trackers were lost AND both found re-ID candidates,
                # verify the two candidates are reasonably close to each other.
                # Real boxers are always fighting within ~4 body-heights of each
                # other; two cornermen on opposite sides of the ring will fail.
                if my_reid_cand is not None and op_reid_cand is not None:
                    m_bx = candidates[my_reid_cand][0]
                    o_bx = candidates[op_reid_cand][0]
                    m_cx = (m_bx[0]+m_bx[2])/2; m_cy = (m_bx[1]+m_bx[3])/2
                    o_cx = (o_bx[0]+o_bx[2])/2; o_cy = (o_bx[1]+o_bx[3])/2
                    max_pair_sep = (expected_my_h or fh*0.40) * 4.5
                    if ((m_cx-o_cx)**2 + (m_cy-o_cy)**2)**0.5 > max_pair_sep:
                        # Too far apart — both candidates are suspect; keep coasting.
                        my_reid_cand = op_reid_cand = None

                if my_reid_cand is not None:
                    my_i = my_reid_cand; used.add(my_i)
                    my_trk.update(candidates[my_i][0])
                    my_box, my_kps = candidates[my_i]
                    last_yolo_my_kps = my_kps
                    my_reid_cooldown = ANALYSIS_STRIDE * 5
                    unmatched = [(i,bx,kps) for i,bx,kps in unmatched if i != my_i]

                if op_reid_cand is not None:
                    op_i = op_reid_cand; used.add(op_i)
                    op_trk.update(candidates[op_i][0])
                    op_box, op_kps = candidates[op_i]
                    last_yolo_op_kps = op_kps
                    op_reid_cooldown = ANALYSIS_STRIDE * 5

            else:
                # ── Skipped frame: Kalman prediction + last known keypoints ───
                # Deliberately not calling miss() — this is a planned skip,
                # not a tracking failure.
                my_box, my_kps = my_pred, last_yolo_my_kps
                op_box, op_kps = op_pred, last_yolo_op_kps

            # ── SAM2 box override ─────────────────────────────────────────────
            # When SAM2 tracking ran successfully, its per-frame bounding boxes
            # replace whatever the Kalman/YOLO pipeline produced.  Keypoints are
            # still sourced from YOLO (SAM2 gives masks, not pose skeletons).
            # IMPORTANT: also reset the Kalman coast counter to 0 for whichever
            # tracker SAM2 has a valid box for — otherwise the coast counter
            # would make the box appear grey/thin even though SAM2 is tracking
            # the boxer perfectly.
            if sam2_boxes is not None:
                s2 = sam2_boxes.get(str(fi), {})
                # Use `is not None` — SAM2 returns explicit None when the
                # tracker lost the person; a falsy list [] is also valid (empty
                # mask) but distinct from "no data for this frame".
                s2_my = s2.get("my")
                s2_op = s2.get("op")
                if s2_my is not None:
                    my_box = s2_my
                    my_trk.coast = 0   # SAM2 is tracking — don't coast-grey the box
                if s2_op is not None:
                    op_box = s2_op
                    op_trk.coast = 0   # SAM2 is tracking — don't coast-grey the box

            # ── Confidence: 1.0 = live detection, decays with coast ───────────
            my_conf = max(0.0, 1.0 - my_trk.coast * 0.20) * (0.85 if not is_yolo_frame else 1.0)
            op_conf = max(0.0, 1.0 - op_trk.coast * 0.20) * (0.85 if not is_yolo_frame else 1.0)

            # ── Tick down settling counters ───────────────────────────────────
            if my_reid_cooldown   > 0: my_reid_cooldown   -= 1
            if op_reid_cooldown   > 0: op_reid_cooldown   -= 1

            # ── Clinch detection ──────────────────────────────────────────────
            # Two triggers — either is sufficient:
            #
            # (A) IoU overlap: both boxes substantially overlap.  Coast check
            #     removed — when one tracker claims the merged box and the
            #     other coasts, the old coast==0 requirement silently missed
            #     the clinch and allowed metrics to accumulate from bad data.
            #
            # (B) Proximity: centres are within 0.55× the expected boxer height
            #     (chest-to-chest) even if IoU is small (front-on camera angle).
            #     Uses predicted / Kalman box for the coasting tracker so the
            #     check still fires when one boxer "wins" the merged detection.
            exp_h_ref = expected_my_h or expected_op_h or fh * 0.40
            clinch_iou  = _box_iou(my_box, op_box) > 0.20
            # 0.55× height ≈ chest-to-chest only; 1.2 was firing at normal
            # sparring distance and kept clinch_cooldown perpetually > 0.
            clinch_prox = _dist(_cx_cy(my_box), _cx_cy(op_box)) < exp_h_ref * 0.55
            clinch = clinch_iou or clinch_prox

            # Keep grey for POST_CLINCH_FRAMES after separation — trackers
            # re-acquire noisily in the first few frames after a clinch breaks.
            if clinch:
                my_clinch_cooldown = POST_CLINCH_FRAMES
                op_clinch_cooldown = POST_CLINCH_FRAMES
            else:
                if my_clinch_cooldown > 0: my_clinch_cooldown -= 1
                if op_clinch_cooldown > 0: op_clinch_cooldown -= 1

            # ── Reliability: live + settled + not in / near clinch ────────────
            # Any False here turns that boxer's metrics grey in the charts.
            my_reliable = (my_trk.coast == 0 and my_reid_cooldown == 0
                           and my_clinch_cooldown == 0)
            op_reliable = (op_trk.coast == 0 and op_reid_cooldown == 0
                           and op_clinch_cooldown == 0)

            fd = dict(
                my=dict(box=my_box, kps=my_kps, coast=my_trk.coast, conf=my_conf,
                        reliable=my_reliable),
                op=dict(box=op_box, kps=op_kps, coast=op_trk.coast, conf=op_conf,
                        reliable=op_reliable),
            )

            ann_writer.write(_draw_annotations(frame, fd, prev_fd, fh))
            frames.append(fd)
            prev_fd = fd
            fi += 1
            if fi % 30 == 0:
                meta["progress"] = int(fi / max(total, 1) * 75)
                write_meta(sid, meta)

        # ── VFR correction ────────────────────────────────────────────────────
        # cap.get(CAP_PROP_FPS) reads the container header which is wrong for
        # Variable Frame Rate recordings (common on phones).  We use the actual
        # elapsed time of the last decoded frame to compute the real average fps.
        if _last_pos_ms > 500 and fi > 10:
            corrected_fps = fi / (_last_pos_ms / 1000.0)
            if 10.0 < corrected_fps < 240.0:   # sanity bounds
                fps = corrected_fps

        cap.release()
        ann_writer.release()

        # ── Audio punch detection ─────────────────────────────────────────────
        meta["progress"] = 76; write_meta(sid, meta)
        audio_diag = {}
        try:
            punch_events = _process_audio_punches(vpath, frames, fps,
                                                  diag=audio_diag)
        except Exception as e:
            punch_events = []
            meta["audio_error"] = str(e)

        meta["punch_events_count"] = len(punch_events)
        meta["audio_diag"] = audio_diag
        write_meta(sid, meta)

        meta["progress"] = 78; write_meta(sid, meta)
        metrics = _compute_metrics(frames, fw, fh, fps, punch_events=punch_events)

        meta["progress"] = 84; write_meta(sid, meta)
        # Stamp punch events onto the annotated video (second pass — fast, text only)
        overlay_error = None
        if punch_events:
            try:
                _add_punch_overlays(
                    str(sess_dir(sid) / "annotated.mp4"),
                    punch_events, fps, int(fw), int(fh))
            except Exception as e:
                overlay_error = str(e)
                meta["overlay_error"] = overlay_error
                write_meta(sid, meta)

        meta["annotated_ready"] = True

        meta["progress"] = 88; write_meta(sid, meta)
        cv2.imwrite(str(sess_dir(sid)/"heatmap.png"), _build_heatmap(frames, fw, fh))

        meta["progress"] = 94; write_meta(sid, meta)
        _make_highlights(vpath, frames, fps, fw, fh,
                         str(sess_dir(sid)/"highlights.mp4"))

        meta["progress"] = 100
        meta["status"]   = "done"
        meta["metrics"]  = metrics
        write_meta(sid, meta)

    except Exception as exc:
        import traceback
        meta["status"] = "error"
        meta["error"]  = str(exc)
        write_meta(sid, meta)
        print("[analyse error]", traceback.format_exc())


# ─── Per-frame annotation drawing ────────────────────────────────────────────

def _draw_annotations(frame, fd, prev_fd, fh_px):
    out = frame.copy()
    if fd is None: return out

    fh_out, fw_out = out.shape[:2]
    my = fd["my"]
    op = fd["op"]

    my_coast = my.get("coast", 0) > 0
    op_coast = op.get("coast", 0) > 0

    # Default colours — overridden below when we have live detections
    my_col = C_COAST if my_coast else C_NEUTRAL
    op_col = C_COAST if op_coast else C_NEUTRAL

    # Movement-state colouring: only when both trackers have live detections.
    # Portrait-aware: use Y-axis separation when fh > fw * 1.2, else X-axis.
    if (not my_coast and not op_coast and prev_fd is not None and
            prev_fd["my"].get("coast", 0) == 0 and
            prev_fd["op"].get("coast", 0) == 0):
        mc,  oc  = _cx_cy(my["box"]),           _cx_cy(op["box"])
        pmc, poc = _cx_cy(prev_fd["my"]["box"]), _cx_cy(prev_fd["op"]["box"])
        _portrait = fh_out > fw_out * 1.2
        if _portrait:
            toward = 1.0 if oc[1] > mc[1] else -1.0
            my_dx  = mc[1] - pmc[1]
            op_dx  = oc[1] - poc[1]
        else:
            toward = 1.0 if oc[0] > mc[0] else -1.0
            my_dx  = mc[0] - pmc[0]
            op_dx  = oc[0] - poc[0]
        THR     = 5.0
        my_col = (C_ATTACK  if my_dx *  toward  >  THR else
                  C_RETREAT if my_dx *  toward  < -THR else C_NEUTRAL)
        op_col = (C_ATTACK  if op_dx * -toward  >  THR else
                  C_RETREAT if op_dx * -toward  < -THR else C_NEUTRAL)

    slack = fh_px * 0.06
    for person, box_color, coasting in (
            (my, my_col, my_coast),
            (op, op_col, op_coast)):
        x1, y1, x2, y2 = person["box"]
        kps             = person["kps"]

        # Coasting: grey box, no skeleton.  Keep 2px so boxes remain visible at
        # any display scale — a 1px line disappears on high-resolution output.
        thickness = 2
        cv2.rectangle(out, (x1,y1), (x2,y2), box_color, thickness)
        if coasting or kps is None:
            continue

        # Skeleton
        for i, j in COCO_SKELETON:
            a, b = _kp(kps, i), _kp(kps, j)
            if a and b:
                cv2.line(out, (int(a[0]),int(a[1])), (int(b[0]),int(b[1])),
                         C_SKELETON, 1, cv2.LINE_AA)

        # Keypoints (excluding wrists — drawn separately)
        for ki in range(17):
            if ki in (9, 10): continue
            p = _kp(kps, ki)
            if p:
                cv2.circle(out, (int(p[0]),int(p[1])), 3, C_KP, -1, cv2.LINE_AA)

        # Wrist guard indicators
        ls, rs = _kp(kps, 5), _kp(kps, 6)
        for ki, shoulder in ((9, ls), (10, rs)):
            w = _kp(kps, ki)
            if not w: continue
            col = C_GUARD_UP if (shoulder and w[1] <= shoulder[1]+slack) else C_GUARD_DN
            cv2.circle(out, (int(w[0]),int(w[1])), 7, col,     -1, cv2.LINE_AA)
            cv2.circle(out, (int(w[0]),int(w[1])), 7, (0,0,0),  1, cv2.LINE_AA)

    # ── Debug corner marker: tracking quality indicator ───────────────────────
    # Green = both boxers tracked above CONF_THRESH (metrics counted this frame)
    # Red   = one or both below threshold (frame excluded from metrics)
    my_ok = my.get("conf", 0.0) >= CONF_THRESH
    op_ok = op.get("conf", 0.0) >= CONF_THRESH
    marker_col = (30, 200, 30) if (my_ok and op_ok) else (30, 30, 220)
    mx1, my1, mx2, my2 = fw_out - 26, 8, fw_out - 8, 26
    cv2.rectangle(out, (mx1, my1), (mx2, my2), marker_col, -1)
    cv2.rectangle(out, (mx1, my1), (mx2, my2), (0, 0, 0), 1)

    return out


# ─── Metrics ─────────────────────────────────────────────────────────────────

KP_L_SHOULDER, KP_R_SHOULDER, KP_L_WRIST, KP_R_WRIST = 5, 6, 9, 10

def _compute_metrics(frames, fw, fh, fps, punch_events=None):
    N = len(frames)
    if N == 0:
        return dict(blitz_score=0, aggression_score=0, guard_score=0,
                    stamina_drop=0, ring_control_score=0,
                    op_blitz_score=0, op_aggression_score=0, op_guard_score=0,
                    op_stamina_drop=0, op_ring_control_score=0)

    # Normalize punch threshold to median boxer height so detection is
    # scale-invariant.  A boxer far from camera appears ~150px tall; close-up
    # they may be 450px.  Using frame width (old approach) made the threshold
    # ~3× too large, silently discarding most real punches.
    _bh_samples = (
        [fd["my"]["box"][3] - fd["my"]["box"][1]
         for fd in frames if fd["my"].get("reliable", True)] +
        [fd["op"]["box"][3] - fd["op"]["box"][1]
         for fd in frames if fd["op"].get("reliable", True)]
    )
    _bh_samples.sort()
    ref_h = float(_bh_samples[len(_bh_samples) // 2]) if _bh_samples else fh * 0.40
    PUNCH_THRESH = ref_h * PUNCH_THRESH_FRAC   # e.g. 7% of median boxer height

    # ── Orientation detection ────────────────────────────────────────────────
    # Portrait (fh > fw * 1.2): phone / Instagram / social media clips filmed
    # vertically.  All axis-dependent logic must use the primary separation axis
    # (y in portrait, x in landscape) so metrics remain meaningful.
    IS_PORTRAIT = fh > fw * 1.2
    norm_dim    = fh if IS_PORTRAIT else fw   # axis for activity, advance normalisation

    lo, hi = 0.35 * norm_dim, 0.65 * norm_dim  # centre-ring band along primary axis

    # Advancing threshold: minimum per-frame displacement (toward opponent) to
    # count as pressing forward.  Normalised to primary axis and fps so it is
    # the same physical speed regardless of camera zoom or recording frame rate.
    ADVANCE_THRESH = norm_dim * 0.07 / max(fps, 1)  # ≈ 3px at 1280px/30fps

    # ── Per-frame activity: body movement + wrist speed ───────────────────────
    my_act, op_act = [], []
    my_punch_ev,   op_punch_ev   = [], []   # 1/0 — punch thrown (CV wrist speed)
    my_landed_ev,  op_landed_ev  = [], []   # 1/0 — punch landed (audio confirmed)
    my_guard_ev,   op_guard_ev   = [], []   # 1/0 — guard up
    my_advance_ev, op_advance_ev = [], []   # 1/0 — advancing toward opponent
    my_ring_ev,    op_ring_ev    = [], []   # 1/0 — holding centre while opp outside it
    my_reliable_ev, op_reliable_ev = [], [] # 1/0 — tracking confident this frame
    my_cxs, op_cxs = [], []
    prev_my_box = prev_op_box = None
    prev_my_kps = prev_op_kps = None

    # Frame sets for audio-confirmed landed punches (fast O(1) lookup)
    _my_landed_frames = set()
    _op_landed_frames = set()
    if punch_events:
        for ev in punch_events:
            if ev['boxer'] == 'my':
                _my_landed_frames.add(ev['frame_idx'])
            else:
                _op_landed_frames.add(ev['frame_idx'])

    def _guard_up(kps):
        """Guard is up unless BOTH wrists are clearly below shoulder level.

        User requirement: "hands somewhere around the head = guard up,
        only if BOTH are down = guard off."  This gives generous credit —
        any wrist near or above shoulder line keeps guard = 1.

        Slack: 8% of frame height below shoulders still counts as "up"
        (covers natural stance with elbows flared, gloves at chin level).
        """
        if kps is None: return 1   # no keypoints → assume guard up (extrapolate)
        shoulders = [_kp(kps, i) for i in (5, 6) if _kp(kps, i) is not None]
        wrists    = [_kp(kps, i) for i in (9, 10) if _kp(kps, i) is not None]
        if not shoulders or not wrists: return 1   # no data → assume guard up
        avg_sh_y = sum(s[1] for s in shoulders) / len(shoulders)
        slack = fh * 0.08   # generous: 8% frame height below shoulders
        # Guard is OFF only when ALL visible wrists are below threshold
        return 0 if all(w[1] > avg_sh_y + slack for w in wrists) else 1

    for fi_local, fd in enumerate(frames):
        my = fd["my"]; op = fd["op"]
        my_box = my["box"]; op_box = op["box"]
        my_kps = my.get("kps"); op_kps = op.get("kps")
        my_cx  = _cx_cy(my_box); op_cx  = _cx_cy(op_box)
        my_cxs.append(my_cx); op_cxs.append(op_cx)

        # ── Activity (always compute — extrapolate through uncertain periods) ─
        if prev_my_box is not None:
            s = (_dist(_cx_cy(prev_my_box), my_cx) + _wrist_speed(my_kps, prev_my_kps) * 1.5) / norm_dim
            my_act.append(s)
        else:
            my_act.append(0.0)

        if prev_op_box is not None:
            s = (_dist(_cx_cy(prev_op_box), op_cx) + _wrist_speed(op_kps, prev_op_kps) * 1.5) / norm_dim
            op_act.append(s)
        else:
            op_act.append(0.0)

        # ── Per-frame punch + guard events (always compute) ───────────────────
        if prev_my_kps is not None:
            my_punch_ev.append(1 if _wrist_speed(my_kps, prev_my_kps) > PUNCH_THRESH else 0)
        else:
            my_punch_ev.append(0)
        if prev_op_kps is not None:
            op_punch_ev.append(1 if _wrist_speed(op_kps, prev_op_kps) > PUNCH_THRESH else 0)
        else:
            op_punch_ev.append(0)
        my_guard_ev.append(_guard_up(my_kps))
        op_guard_ev.append(_guard_up(op_kps))

        # ── Audio-confirmed landed punch events ───────────────────────────────
        my_landed_ev.append(1 if fi_local in _my_landed_frames else 0)
        op_landed_ev.append(1 if fi_local in _op_landed_frames else 0)

        # ── Advance + ring control per-frame events (always compute) ──────────
        if prev_my_box is not None and prev_op_box is not None:
            pmc = _cx_cy(prev_my_box)
            poc = _cx_cy(prev_op_box)
            if IS_PORTRAIT:
                # Primary separation axis is vertical in portrait
                toward_op  = 1.0 if op_cx[1] > my_cx[1] else -1.0
                my_advance_ev.append(1 if (my_cx[1] - pmc[1]) * toward_op  > ADVANCE_THRESH else 0)
                op_advance_ev.append(1 if (op_cx[1] - poc[1]) * -toward_op > ADVANCE_THRESH else 0)
            else:
                toward_op  = 1.0 if op_cx[0] > my_cx[0] else -1.0
                my_advance_ev.append(1 if (my_cx[0] - pmc[0]) * toward_op  > ADVANCE_THRESH else 0)
                op_advance_ev.append(1 if (op_cx[0] - poc[0]) * -toward_op > ADVANCE_THRESH else 0)
        else:
            my_advance_ev.append(0)
            op_advance_ev.append(0)

        # Ring control: three overlapping signals — axis-aware
        op_ropes = _is_on_ropes(op_box, fw, fh)
        my_ropes = _is_on_ropes(my_box, fw, fh)
        if IS_PORTRAIT:
            my_in_ctr = lo <= my_cx[1] <= hi
            op_in_ctr = lo <= op_cx[1] <= hi
            my_nearer = abs(my_cx[1]/fh - 0.5) < abs(op_cx[1]/fh - 0.5)
            op_nearer = abs(op_cx[1]/fh - 0.5) < abs(my_cx[1]/fh - 0.5)
        else:
            my_in_ctr = lo <= my_cx[0] <= hi
            op_in_ctr = lo <= op_cx[0] <= hi
            my_nearer = abs(my_cx[0]/fw - 0.5) < abs(op_cx[0]/fw - 0.5)
            op_nearer = abs(op_cx[0]/fw - 0.5) < abs(my_cx[0]/fw - 0.5)
        my_ring_ev.append(1 if (my_in_ctr and not op_in_ctr)
                                or op_ropes
                                or (my_nearer and my_ropes is False) else 0)
        op_ring_ev.append(1 if (op_in_ctr and not my_in_ctr)
                                or my_ropes
                                or (op_nearer and op_ropes is False) else 0)

        # Reliability: always 1 — no grey bars, metrics are extrapolated
        my_reliable_ev.append(1)
        op_reliable_ev.append(1)

        # Always advance prev-frame refs (we extrapolate through gaps)
        prev_my_box = my_box
        prev_op_box = op_box
        prev_my_kps = my_kps
        prev_op_kps = op_kps

    # ── Blitz: best 5-second burst window ────────────────────────────────────
    # Activity is in fw-normalised units.  elite = ~0.18 fw per frame means
    # near-continuous punching + rapid footwork — very hard to sustain.
    win   = max(1, int(fps * 5))
    elite = fps * 5 * 0.18   # normalised; replaces old absolute-pixel reference

    def _blitz(act):
        best = max((sum(act[i:i+win]) for i in range(max(1, N - win + 1))), default=0.0)
        return min(100, max(1, round(best / elite * 100)))

    # ── Aggression: punching + pressing a cornered / retreating opponent ──────
    my_agg_yes = my_agg_tot = 0
    op_agg_yes = op_agg_tot = 0

    for i in range(1, N):
        fd = frames[i]; pfd = frames[i - 1]
        my = fd["my"]; op = fd["op"]
        p_my = pfd["my"]; p_op = pfd["op"]

        mc  = my_cxs[i];     oc  = op_cxs[i]
        pmc = my_cxs[i - 1]; poc = op_cxs[i - 1]

        if IS_PORTRAIT:
            toward_op = 1.0 if oc[1] > mc[1] else -1.0
            my_dx = mc[1] - pmc[1]
            op_dx = oc[1] - poc[1]
        else:
            toward_op = 1.0 if oc[0] > mc[0] else -1.0
            my_dx = mc[0] - pmc[0]
            op_dx = oc[0] - poc[0]

        my_punch = _wrist_speed(my.get("kps"), pfd["my"].get("kps")) > PUNCH_THRESH
        op_punch = _wrist_speed(op.get("kps"), pfd["op"].get("kps")) > PUNCH_THRESH

        my_advancing  = my_dx * toward_op   >  ADVANCE_THRESH
        op_advancing  = op_dx * -toward_op  >  ADVANCE_THRESH
        op_cornered   = _is_on_ropes(op["box"], fw, fh)
        my_cornered   = _is_on_ropes(my["box"], fw, fh)
        op_retreating = op_dx * -toward_op  < -ADVANCE_THRESH
        my_retreating = my_dx * toward_op   < -ADVANCE_THRESH

        my_agg_tot += 1
        if my_punch or (my_advancing and (op_cornered or (op_retreating and not op_punch))):
            my_agg_yes += 1

        op_agg_tot += 1
        if op_punch or (op_advancing and (my_cornered or (my_retreating and not my_punch))):
            op_agg_yes += 1

    my_agg = round(my_agg_yes / max(my_agg_tot, 1) * 100)
    op_agg = round(op_agg_yes / max(op_agg_tot, 1) * 100)

    # ── Guard: aggregate score — delegates to _guard_up() so the same
    #    definition (8% slack, both-down = off) is used for both
    #    the per-second chart and the summary number.
    def _guard(role):
        g_yes = g_tot = 0
        for fd in frames:
            who = fd[role]
            kps = who.get("kps")
            g_tot += 1
            g_yes += _guard_up(kps)
        return round(g_yes / max(g_tot, 1) * 100)

    # ── Pace timeline ─────────────────────────────────────────────────────────
    # Strategy: for each second of video, sample the 80th-percentile of
    # fw-normalised activity within a wide symmetric window.
    #
    # Why 80th percentile + wide window?
    #   • A boxer waiting / clinching for a few seconds produces zero-activity
    #     frames, but those are simply low values in a large pool of samples.
    #     The 80th percentile still picks up the surrounding active frames, so
    #     the curve does NOT drop during tactical pauses.
    #   • Random noise spikes (camera shake, single fast frame) cannot push the
    #     80th percentile up dramatically because they are just one sample among
    #     hundreds of window frames — the curve stays stable.
    fps_int   = max(1, int(round(fps)))
    n_secs    = max(1, N // fps_int)
    # Adaptive smoothing half-window: 5 s min, 15 s max, ~1/6 of video length
    sw_secs   = min(15, max(5, n_secs // 6))
    sw_frames = sw_secs * fps_int

    def _pace_series(act):
        series = []
        last_good = 0.0   # carry-forward: pace never drops to zero
        for s in range(n_secs):
            c    = s * fps_int
            vals = [v for v in act[max(0, c - sw_frames) : min(N, c + sw_frames + 1)]
                    if v > 0]
            if len(vals) >= 3:
                v = float(np.percentile(vals, 80))
                last_good = v
                series.append(v)
            else:
                # Carry forward last known pace (never zero)
                series.append(last_good)
        return series

    my_pace_raw = _pace_series(my_act)
    op_pace_raw = _pace_series(op_act)

    # Normalise both curves to 0-100 against the joint peak so the two lines
    # are directly comparable on the same chart axis.
    _peak = max(max(my_pace_raw, default=0.0), max(op_pace_raw, default=0.0), 1e-9)
    my_pace = [round(v / _peak * 100) for v in my_pace_raw]
    op_pace = [round(v / _peak * 100) for v in op_pace_raw]

    # ── Per-second punch + guard timelines ───────────────────────────────────
    def _per_sec_sum(ev):
        return [sum(ev[s * fps_int : min(N, (s + 1) * fps_int)]) for s in range(n_secs)]

    def _per_sec_pct(ev):
        return [round(sum(ev[s * fps_int : min(N, (s + 1) * fps_int)]) /
                      max(min(N, (s + 1) * fps_int) - s * fps_int, 1) * 100)
                for s in range(n_secs)]

    # No grey bars — metrics are always available (extrapolated if needed)
    my_reliable_sec = [1] * n_secs
    op_reliable_sec = [1] * n_secs

    my_p_raw = _per_sec_sum(my_punch_ev)
    op_p_raw = _per_sec_sum(op_punch_ev)

    # If we have audio-confirmed landed punches, weight them 3× vs unconfirmed throws.
    # landed × 3  +  max(0, thrown − landed)  gives extra credit for clean shots
    # while still counting volume when audio data is absent.
    has_audio = bool(punch_events)
    my_l_raw  = _per_sec_sum(my_landed_ev)
    op_l_raw  = _per_sec_sum(op_landed_ev)
    if has_audio:
        my_eff_raw = [l * 3 + max(0, p - l) for p, l in zip(my_p_raw, my_l_raw)]
        op_eff_raw = [l * 3 + max(0, p - l) for p, l in zip(op_p_raw, op_l_raw)]
    else:
        my_eff_raw, op_eff_raw = my_p_raw, op_p_raw
    _p_peak  = max(max(my_eff_raw, default=0), max(op_eff_raw, default=0), 1)
    my_punches_sec = [round(v / _p_peak * 100) for v in my_eff_raw]
    op_punches_sec = [round(v / _p_peak * 100) for v in op_eff_raw]
    my_guard_sec   = _per_sec_pct(my_guard_ev)
    op_guard_sec   = _per_sec_pct(op_guard_ev)

    # ── Advance + ring control per-second series ──────────────────────────────
    my_advance_sec = _per_sec_pct(my_advance_ev)
    op_advance_sec = _per_sec_pct(op_advance_ev)
    my_ring_sec    = _per_sec_pct(my_ring_ev)
    op_ring_sec    = _per_sec_pct(op_ring_ev)

    # ── Aggression composite: 60% punch rate + 40% forward pressure ───────────
    my_aggression_sec = [round(0.60 * p + 0.40 * a)
                         for p, a in zip(my_punches_sec, my_advance_sec)]
    op_aggression_sec = [round(0.60 * p + 0.40 * a)
                         for p, a in zip(op_punches_sec, op_advance_sec)]

    # ── Per-second punch totals (used for break detection) ────────────────────
    _punch_sec = [sum(my_punch_ev[s * fps_int : min(N, (s + 1) * fps_int)]) +
                  sum(op_punch_ev[s * fps_int : min(N, (s + 1) * fps_int)])
                  for s in range(n_secs)]

    # ── Control composite: 40% ring + 35% aggression + 25% pace ──────────────
    # Raw weighted score per boxer per second, then normalise to 100% stack.
    # When neither boxer throws a punch the round is a break/clinch — return
    # 50/50 so stale positional data doesn't freeze a false advantage.
    my_ctrl_raw = [0.40 * r + 0.35 * a + 0.25 * p
                   for r, a, p in zip(my_ring_sec, my_aggression_sec, my_pace)]
    op_ctrl_raw = [0.40 * r + 0.35 * a + 0.25 * p
                   for r, a, p in zip(op_ring_sec, op_aggression_sec, op_pace)]

    my_control_sec = []
    op_control_sec = []
    for s, (mv, ov) in enumerate(zip(my_ctrl_raw, op_ctrl_raw)):
        # Use only total-activity gate, not punch count.
        # Gating on _punch_sec caused 50/50 collapse whenever CV punch
        # detection produced no events (e.g. when tracking was lost and
        # keypoints were absent). total < 0.5 already catches true inactivity
        # because all three sub-signals (ring, aggression, pace) are near-zero.
        total = mv + ov
        if total < 0.5:
            my_control_sec.append(50)
            op_control_sec.append(50)
        else:
            pct = round(mv / total * 100)
            my_control_sec.append(pct)
            op_control_sec.append(100 - pct)

    # ── Stamina score (derived from smoothed pace curve) ──────────────────────
    # Compares first-third to last-third of the pace series.
    # Because the curve is already smoothed, brief pauses can't distort it.
    # Returns None when < 6 s of data (too short to be meaningful).
    def _stamina(pace):
        if len(pace) < 6:
            return None
        seg = max(1, len(pace) // 3)
        fa  = sum(pace[:seg])  / seg    # opening third
        la  = sum(pace[-seg:]) / seg    # closing third
        if fa <= 0:
            return 100
        return min(100, round(la / fa * 100))

    # ── Ring control: overall score (lo/hi already defined above) ────────────
    # Must use the same axis as the per-frame ring events — y-axis in portrait.
    my_rc_yes = my_rc_tot = 0
    op_rc_yes = op_rc_tot = 0
    for fd, mc, oc in zip(frames, my_cxs, op_cxs):
        if mc is None or oc is None: continue
        my_rc_tot += 1; op_rc_tot += 1
        if IS_PORTRAIT:
            _mc_pos = mc[1]; _oc_pos = oc[1]
        else:
            _mc_pos = mc[0]; _oc_pos = oc[0]
        if lo <= _mc_pos <= hi and not (lo <= _oc_pos <= hi): my_rc_yes += 1
        if lo <= _oc_pos <= hi and not (lo <= _mc_pos <= hi): op_rc_yes += 1
    my_ring = round(my_rc_yes / max(my_rc_tot, 1) * 100)
    op_ring = round(op_rc_yes / max(op_rc_tot, 1) * 100)

    # ── Period-weighted 100-point round score ────────────────────────────────
    # The bout is split into ~15-second scoring periods.  Each period's
    # composite share is time-weighted convexly toward the end, mirroring how
    # real judges mentally award more importance to late-round performance.
    #
    # Composite per period: 45% ring control + 35% aggression + 20% pace.
    # All three are normalised to a 0–1 share between the two fighters so the
    # final result is always my_round_score + op_round_score == 100.
    #
    # Time weight for period k (1-indexed):  w_k = k^α / Σ j^α
    # α = 1.3 → last period of a 60-second video gets ≈ 35-40 % of the weight.

    PERIOD_SECS  = 15    # target seconds per scoring period
    PERIOD_ALPHA = 1.3   # convex exponent; increase for stronger end-weighting

    K = max(2, n_secs // PERIOD_SECS)          # at least 2 periods
    period_size = max(1, n_secs // K)

    def _pavg(series, s0, s1):
        chunk = series[s0:s1]
        return sum(chunk) / max(len(chunk), 1)

    raw_periods = []
    for k in range(K):
        s0 = k * period_size
        s1 = min(n_secs, s0 + period_size)

        # Control is already a normalised 0-100 stack → convert to 0-1 share
        my_ctrl_share = _pavg(my_control_sec, s0, s1) / 100.0

        # Aggression and pace: normalise to 0-1 share within this period
        ma = _pavg(my_aggression_sec, s0, s1)
        oa = _pavg(op_aggression_sec, s0, s1)
        my_agg_share = ma / max(ma + oa, 1.0)

        mp = _pavg(my_pace, s0, s1)
        op_p = _pavg(op_pace, s0, s1)
        my_pace_share = mp / max(mp + op_p, 1.0)

        my_share = 0.45 * my_ctrl_share + 0.35 * my_agg_share + 0.20 * my_pace_share
        raw_periods.append({'my': my_share, 's0': s0, 's1': s1})

    raw_w   = [(k + 1) ** PERIOD_ALPHA for k in range(K)]
    total_w = sum(raw_w)
    weights = [w / total_w for w in raw_w]

    my_weighted = sum(w * p['my'] for w, p in zip(weights, raw_periods))

    my_round_score = round(my_weighted * 100)
    op_round_score = 100 - my_round_score

    period_scores = [
        {
            'my':     round(p['my'] * 100),
            'op':     round((1.0 - p['my']) * 100),
            'weight': round(w * 100),
            'start':  p['s0'],
            'end':    p['s1'],
        }
        for p, w in zip(raw_periods, weights)
    ]

    return dict(
        blitz_score           = _blitz(my_act),
        aggression_score      = my_agg,
        guard_score           = _guard("my"),
        stamina_score         = _stamina(my_pace),
        ring_control_score    = my_ring,
        op_blitz_score        = _blitz(op_act),
        op_aggression_score   = op_agg,
        op_guard_score        = _guard("op"),
        op_stamina_score      = _stamina(op_pace),
        op_ring_control_score = op_ring,
        # Pace chart data (one value per second of video)
        my_pace               = my_pace,
        op_pace               = op_pace,
        # Activity timeline component datasets
        my_punches_sec        = my_punches_sec,
        op_punches_sec        = op_punches_sec,
        my_guard_sec          = my_guard_sec,
        op_guard_sec          = op_guard_sec,
        # Swimlane composite series
        my_control_sec        = my_control_sec,
        op_control_sec        = op_control_sec,
        my_aggression_sec     = my_aggression_sec,
        op_aggression_sec     = op_aggression_sec,
        # Reliability masks (1 = confident tracking, 0 = grey out)
        my_reliable_sec       = my_reliable_sec,
        op_reliable_sec       = op_reliable_sec,
        # Period-weighted 100-point round score
        my_round_score        = my_round_score,
        op_round_score        = op_round_score,
        period_scores         = period_scores,
    )


# ─── Audio punch detection ───────────────────────────────────────────────────

def _process_audio_punches(vpath, frames, fps, diag=None):
    """Detect punch impacts in the audio track and cross-reference with CV data.

    Strategy (two-stage to handle noisy gym environments):
      1. librosa onset detection: fast pre-filter that finds sudden energy spikes.
         Only the top-30% strongest onsets are passed to stage 2.
      2. PANNs CNN14 classification: verifies each candidate is actually a
         punch/impact sound.  Requires confidence > 0.40 (high bar for noise).
      3. CV cross-reference: the boxer whose wrist speed peaks in the ±150 ms
         window around the audio event gets credit.  If no boxer's wrist speed
         clears 50% of PUNCH_THRESH, the event is discarded (pure background).

    Falls back gracefully when:
      - Video has no audio track  → returns []
      - librosa not installed     → returns []
      - PANNs model unavailable   → uses normalised onset strength as proxy
        (less accurate; audio events only count if CV wrist speed is strong)

    Returns a list of dicts sorted by frame_idx:
      { frame_idx, time_sec, boxer ('my'|'op'), hand ('L'|'R'),
        wrist_speed (px), audio_conf (0-1), intensity ('LIGHT'|'MED'|'HARD'),
        intensity_val (0-1) }
    """
    import subprocess, tempfile, shutil as _shutil, csv as _csv

    if diag is None:
        diag = {}

    N = len(frames)
    if N == 0:
        diag['stage'] = 'no_frames'; return []

    # Estimate punch threshold from median boxer height in frames
    bh_samples = [fd['my']['box'][3] - fd['my']['box'][1]
                  for fd in frames if fd['my']['box'] is not None]
    bh_samples += [fd['op']['box'][3] - fd['op']['box'][1]
                   for fd in frames if fd['op']['box'] is not None]
    ref_h = float(sorted(bh_samples)[len(bh_samples) // 2]) if bh_samples else 200.0
    cv_punch_thresh = ref_h * PUNCH_THRESH_FRAC
    # When PANNs is available the CV gate is tighter (50% of punch thresh) since
    # audio is already a strong signal.  Without PANNs the fallback onset-strength
    # is weaker, so we lower the CV gate to 20% to avoid missing real punches —
    # especially for back-facing shots where wrist keypoints are less reliable.
    # We set a tentative value here; it may be overridden below once we know
    # whether PANNs loaded successfully.
    cv_attr_thresh_with_panns    = cv_punch_thresh * 0.50
    cv_attr_thresh_without_panns = cv_punch_thresh * 0.20
    cv_attr_thresh = cv_attr_thresh_without_panns  # default; tightened if PANNs loads
    diag['ref_h'] = round(ref_h, 1)
    diag['cv_attr_thresh'] = round(cv_attr_thresh, 1)

    events = []

    with tempfile.TemporaryDirectory() as tmp:
        audio_path = Path(tmp) / 'audio.wav'

        # ── 1. Extract audio (mono 32 kHz — PANNs native rate) ───────────────
        r = subprocess.run(
            ['ffmpeg', '-y', '-i', str(vpath),
             '-ac', '1', '-ar', '32000', '-vn', str(audio_path)],
            capture_output=True)
        wav_size = audio_path.stat().st_size if audio_path.exists() else 0
        diag['ffmpeg_rc']  = r.returncode
        diag['wav_bytes']  = wav_size
        if r.returncode != 0 or not audio_path.exists() or wav_size < 4096:
            diag['stage'] = 'no_audio_track'
            return []   # no audio track or extraction failed

        # ── 2. librosa onset detection ────────────────────────────────────────
        try:
            import librosa
        except ImportError:
            diag['stage'] = 'librosa_missing'; return []

        y22, sr22 = librosa.load(str(audio_path), sr=22050, mono=True)
        diag['audio_dur_s'] = round(len(y22) / sr22, 1)
        hop = 512

        # Separate percussive component — isolates punch transients from
        # crowd noise, music, and voices that dominate gym recordings.
        try:
            D        = librosa.stft(y22, hop_length=hop)
            _, D_p   = librosa.decompose.hpss(D, margin=3)
            y_perc   = librosa.istft(D_p, hop_length=hop, length=len(y22))
        except Exception:
            y_perc = y22   # fallback: use raw signal

        onset_env   = librosa.onset.onset_strength(y=y_perc, sr=sr22, hop_length=hop)
        onset_fidx  = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr22, hop_length=hop,
            pre_max=1, post_max=1, pre_avg=3, post_avg=3,
            delta=0.07, wait=4)           # ≥47 ms between candidates; sensitive
        onset_times = librosa.frames_to_time(onset_fidx, sr=sr22, hop_length=hop)
        onset_strs  = onset_env[onset_fidx]
        diag['onsets_raw'] = len(onset_times)

        if len(onset_times) == 0:
            diag['stage'] = 'no_onsets'; return []

        # ── 3. PANNs classification ───────────────────────────────────────────
        # Attempt to load PANNs first so we know which onset filter to apply.
        AUDIO_CONF_THRESH = 0.40   # high bar for noisy environments
        panns_at      = None
        punch_indices = []

        try:
            import soundfile as sf
            import panns_inference as _panns_mod
            from panns_inference import AudioTagging
            import pkg_resources

            # CSV may live inside the package dir, in ~/panns_data/, or wherever
            # pkg_resources resolves to — try all candidate paths.
            _csv_candidates = [
                str(Path(_panns_mod.__file__).parent / 'class_labels_indices.csv'),
                pkg_resources.resource_filename(
                    'panns_inference', 'class_labels_indices.csv'),
                str(Path.home() / 'panns_data' / 'class_labels_indices.csv'),
            ]
            labels_path = next(
                (p for p in _csv_candidates if Path(p).exists()), None)

            PUNCH_KW = {'punch', 'beat', 'slap', 'smack', 'thump', 'thud',
                        'whack', 'thwack', 'hit', 'impact', 'strike'}
            if labels_path:
                with open(labels_path) as f:
                    for row in _csv.DictReader(f):
                        if any(kw in row['display_name'].lower() for kw in PUNCH_KW):
                            punch_indices.append(int(row['index']))
            else:
                # AudioSet hard-coded fallback indices for impact/punch sounds
                # (Slap/smack=422, Whack/thwack=423, Smash/crash=424, Bang=461,
                #  Thud=462, Knock=463 — stable across AudioSet revisions)
                punch_indices = [422, 423, 424, 461, 462, 463]

            if punch_indices:
                panns_at = AudioTagging(checkpoint_path=None, device='cpu')
                audio32, _ = sf.read(str(audio_path), dtype='float32')
                if audio32.ndim > 1:
                    audio32 = audio32[:, 0]
            diag['panns'] = f'loaded (csv={"found" if labels_path else "hardcoded"})'
            cv_attr_thresh = cv_attr_thresh_with_panns   # PANNs loaded — use tighter CV gate
        except Exception as pe:
            diag['panns'] = f'unavailable: {pe}'
            # PANNs unavailable — use looser onset filter + looser CV gate

        # ── 3b. Onset strength filter — applied after PANNs load attempt ─────
        # With PANNs: keep top 30% (strict — CNN14 is the primary gate).
        # Without PANNs: keep top 50% (looser — onset strength is our only audio signal).
        pct_cutoff  = 70 if panns_at is not None else 50
        str_thresh  = float(np.percentile(onset_strs, pct_cutoff))
        onset_times = onset_times[onset_strs >= str_thresh]
        onset_strs  = onset_strs[ onset_strs >= str_thresh]
        diag['onsets_filtered'] = len(onset_times)

        # ── 4. Score each onset candidate ─────────────────────────────────────
        n_passed_audio = 0
        n_passed_cv    = 0
        max_str_global = float(onset_strs.max()) + 1e-9
        for t, ostr in zip(onset_times, onset_strs):
            if panns_at is not None:
                # Extract 2-second window centred on onset for CNN14
                sr32   = 32000
                centre = int(t * sr32)
                half   = sr32          # 1 s each side → 2 s window
                chunk  = audio32[max(0, centre - half): centre + half]
                if len(chunk) < sr32 * 2:
                    chunk = np.pad(chunk, (0, sr32 * 2 - len(chunk)))
                try:
                    probs, _ = panns_at.inference(chunk[np.newaxis, :])
                    audio_conf = float(np.sum(probs[0, punch_indices]))
                except Exception:
                    audio_conf = 0.0
            else:
                # Fallback: normalised onset strength (no PANNs).
                # After percussive separation, the strongest onsets are likely
                # real impacts, so pass everything through to the CV gate.
                audio_conf = float(ostr / max_str_global) * 0.50
                AUDIO_CONF_THRESH = 0.10   # very low bar — CV cross-ref is the real filter

            if audio_conf < AUDIO_CONF_THRESH:
                continue
            n_passed_audio += 1

            # ── 5. CV cross-reference: attribute to boxer + which hand ─────────
            frame_idx    = max(0, min(N - 1, int(t * fps)))
            window_f     = max(1, int(0.15 * fps))   # ±150 ms
            lo, hi       = max(0, frame_idx - window_f), min(N - 1, frame_idx + window_f)

            best_boxer = None
            best_ws    = 0.0
            best_hand  = 'R'
            best_fi    = frame_idx

            for fi in range(lo, hi + 1):
                if fi == 0:
                    continue
                fd      = frames[fi]
                prev_fd = frames[fi - 1]
                for boxer_key in ('my', 'op'):
                    l_spd, r_spd = _wrist_speed_lr(
                        fd[boxer_key].get('kps'),
                        prev_fd[boxer_key].get('kps'))
                    peak = max(l_spd, r_spd)
                    if peak > best_ws:
                        best_ws    = peak
                        best_boxer = boxer_key
                        best_hand  = 'L' if l_spd >= r_spd else 'R'
                        best_fi    = fi

            # Discard if CV shows no meaningful wrist movement (background noise)
            if best_boxer is None or best_ws < cv_attr_thresh:
                continue
            n_passed_cv += 1

            events.append(dict(
                frame_idx  = best_fi,
                time_sec   = best_fi / max(fps, 1.0),
                boxer      = best_boxer,
                hand       = best_hand,
                wrist_speed= best_ws,
                audio_conf = audio_conf,
            ))

        diag['n_passed_audio'] = n_passed_audio
        diag['n_passed_cv']    = n_passed_cv
        diag['stage']          = 'complete'

    # ── 6. Deduplicate: within a 200 ms window keep the highest-confidence hit ─
    events.sort(key=lambda e: e['frame_idx'])
    merged, min_gap = [], max(1, int(0.20 * fps))
    for ev in events:
        if merged and ev['frame_idx'] - merged[-1]['frame_idx'] < min_gap:
            if ev['audio_conf'] > merged[-1]['audio_conf']:
                merged[-1] = ev
        else:
            merged.append(ev)

    # ── 7. Normalise wrist speeds → intensity label ────────────────────────────
    if merged:
        max_ws = max(e['wrist_speed'] for e in merged) + 1e-9
        for ev in merged:
            iv = 0.6 * (ev['wrist_speed'] / max_ws) + 0.4 * ev['audio_conf']
            ev['intensity_val'] = iv
            ev['intensity']     = ('HARD' if iv > 0.65 else
                                   'MED'  if iv > 0.35 else 'LIGHT')

    return merged


def _add_punch_overlays(ann_path, punch_events, fps, fw, fh):
    """Second-pass: read annotated.mp4, stamp punch events in top corners, overwrite.

    Top-left  = "ME"       (boxer selected by the user)
    Top-right = "OPPONENT"

    Each entry shows: timestamp · hand · intensity dots (●●●)
    Up to 5 most-recent entries per boxer are shown; newest at the top.
    Text fades with age so the current event stands out.
    """
    import shutil as _shutil

    if not punch_events or not Path(ann_path).exists():
        return

    tmp_path = ann_path.replace('.mp4', '_po_tmp.mp4')

    cap    = cv2.VideoCapture(ann_path)
    if not cap.isOpened():
        raise RuntimeError(f"_add_punch_overlays: cannot open {ann_path}")

    # Read actual dimensions from the video — more reliable than passed-in values
    # (avoids numpy int64 / float mismatch issues with cv2.VideoWriter)
    real_fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    real_fps = cap.get(cv2.CAP_PROP_FPS) or fps

    if real_fw == 0 or real_fh == 0:
        cap.release()
        raise RuntimeError(f"_add_punch_overlays: zero dimensions ({real_fw}x{real_fh})")

    writer = cv2.VideoWriter(
        tmp_path, cv2.VideoWriter_fourcc(*'mp4v'),
        real_fps, (real_fw, real_fh))

    # Build per-boxer sorted event lists
    my_evs = sorted([e for e in punch_events if e['boxer'] == 'my'],
                    key=lambda e: e['frame_idx'])
    op_evs = sorted([e for e in punch_events if e['boxer'] == 'op'],
                    key=lambda e: e['frame_idx'])

    my_ptr = op_ptr = 0
    my_recent: list = []   # newest first, max 5
    op_recent: list = []

    ROWS   = 5
    PAD    = max(10, int(real_fh * 0.013))
    ROW_H  = max(20, int(real_fh * 0.026))
    FSCALE = ROW_H / 38.0
    FONT   = cv2.FONT_HERSHEY_SIMPLEX
    FBOLD  = cv2.FONT_HERSHEY_DUPLEX

    DOTS   = {'HARD': '\xe2\x97\x8f\xe2\x97\x8f\xe2\x97\x8f',
              'MED':  '\xe2\x97\x8f\xe2\x97\x8f',
              'LIGHT':'\xe2\x97\x8f'}   # UTF-8 bullets (●●●)
    # OpenCV putText doesn't render Unicode; use ASCII equivalents
    DOTS   = {'HARD': '* * *', 'MED': '* *', 'LIGHT': '*'}

    INT_COL = {'HARD':  (50,  90, 255),
               'MED':   (50, 180, 255),
               'LIGHT': (140, 210, 255)}

    def _ts(sec):
        return f"{int(sec)//60}:{int(sec)%60:02d}"

    def _draw_panel(out, recent, right_side):
        if not recent:
            return
        n       = min(ROWS, len(recent))
        title   = "OPPONENT" if right_side else "ME"
        title_w, _ = cv2.getTextSize(title, FBOLD, FSCALE * 0.85, 1)[0], None
        title_w = title_w[0]

        # Measure widest data line
        samples = [f"{_ts(e['time_sec'])}  {e['hand']}  {DOTS[e['intensity']]}"
                   for e in recent[:n]]
        max_tw  = max((cv2.getTextSize(s, FONT, FSCALE, 1)[0][0] for s in samples),
                      default=0)
        box_w   = max(max_tw, title_w) + PAD * 2
        box_h   = (n + 1) * ROW_H + PAD * 2

        x0 = real_fw - box_w - PAD if right_side else PAD
        y0 = PAD

        # Semi-transparent dark background
        overlay = out.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.60, out, 0.40, 0, out)

        # Title
        t_col = (100, 120, 255) if right_side else (100, 100, 220)
        cv2.putText(out, title,
                    (x0 + PAD, y0 + PAD + ROW_H - 5),
                    FBOLD, FSCALE * 0.85, t_col, 1, cv2.LINE_AA)

        # Punch entries — newest (index 0) at top, fades with age
        for i, (ev, line) in enumerate(zip(recent[:n], samples)):
            alpha   = max(0.35, 1.0 - i * 0.18)
            base    = INT_COL.get(ev['intensity'], (180, 180, 180))
            col     = tuple(int(v * alpha) for v in base)
            y_txt   = y0 + PAD + (i + 2) * ROW_H - 5
            cv2.putText(out, line, (x0 + PAD, y_txt),
                        FONT, FSCALE, col, 1, cv2.LINE_AA)

    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Advance pointers — add any events that occurred up to this frame
        while my_ptr < len(my_evs) and my_evs[my_ptr]['frame_idx'] <= fi:
            my_recent.insert(0, my_evs[my_ptr])
            my_recent = my_recent[:ROWS]
            my_ptr += 1
        while op_ptr < len(op_evs) and op_evs[op_ptr]['frame_idx'] <= fi:
            op_recent.insert(0, op_evs[op_ptr])
            op_recent = op_recent[:ROWS]
            op_ptr += 1

        _draw_panel(frame, my_recent, right_side=False)
        _draw_panel(frame, op_recent, right_side=True)

        writer.write(frame)
        fi += 1

    cap.release()
    writer.release()

    # Swap in the overlaid version
    if Path(tmp_path).exists() and Path(tmp_path).stat().st_size > 10_000:
        _shutil.move(tmp_path, ann_path)
    elif Path(tmp_path).exists():
        Path(tmp_path).unlink(missing_ok=True)


# ─── Heatmap ─────────────────────────────────────────────────────────────────

def _build_heatmap(frames, fw, fh, size=400):
    all_h = []
    for fd in frames:
        b = fd["my"]["box"]; all_h.append(b[3]-b[1])
        b = fd["op"]["box"]; all_h.append(b[3]-b[1])
    ref_h = float(np.median(all_h)) if all_h else fh * 0.5

    my_h = np.zeros((size, size), dtype=np.float32)
    op_h = np.zeros((size, size), dtype=np.float32)

    for fd in frames:
        for h_arr, who in ((my_h, fd["my"]), (op_h, fd["op"])):
            box    = who["box"]
            bx     = (box[0]+box[2]) / 2.0
            bbox_h = float(box[3]-box[1])
            # Clip to valid grid bounds — Kalman can predict outside the frame
            gx     = int(np.clip(bx / fw * size, 0, size - 1))
            rel    = bbox_h / max(ref_h, 1.0)
            depth  = float(np.clip((rel - 0.65) / 0.70, 0.0, 1.0))
            gy     = int(np.clip((1.0 - depth) * (size - 1), 0, size - 1))
            h_arr[gy, gx] += 1

    my_h = cv2.GaussianBlur(my_h, (31, 31), 0)
    op_h = cv2.GaussianBlur(op_h, (31, 31), 0)
    if my_h.max() > 0: my_h /= my_h.max()
    if op_h.max() > 0: op_h /= op_h.max()

    canvas = np.full((size, size, 3), 18, dtype=np.uint8)
    m = 10
    cv2.rectangle(canvas, (m,m), (size-m, size-m), (65,65,65), 2)
    for frac in (0.33, 0.66):
        y = int(m + (size-2*m) * frac)
        cv2.line(canvas, (m,y), (size-m,y), (40,40,40), 1)
    cv2.putText(canvas, "CAMERA",   (size//2-28, size-3), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (70,70,70), 1)
    cv2.putText(canvas, "FAR SIDE", (size//2-30, 18),     cv2.FONT_HERSHEY_SIMPLEX, 0.36, (70,70,70), 1)

    for h_arr, bgr in ((my_h, (0,130,255)), (op_h, (220,60,0))):
        mask  = h_arr[:,:,np.newaxis]
        layer = np.zeros((size, size, 3), dtype=np.uint8)
        for c, v in enumerate(bgr):
            layer[:,:,c] = np.clip(h_arr * v, 0, 255).astype(np.uint8)
        canvas = np.clip(
            canvas.astype(np.float32) * (1-mask*0.85) +
            layer.astype(np.float32)  *    mask*0.85,
            0, 255).astype(np.uint8)

    for label, col, y in (("YOU",(0,130,255),size-36), ("OPP",(220,60,0),size-16)):
        cv2.rectangle(canvas, (12,y-8), (20,y), col, -1)
        cv2.putText(canvas, label, (24,y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1)
    return canvas


# ─── Highlights clip ─────────────────────────────────────────────────────────

def _make_highlights(vpath, frames, fps, fw, fh, out_path):
    """
    3-phase highlight reel (≤20 s total):
      Phase 1 — Quick moments (~4 s): top individual seconds from first 40%
      Phase 2 — Peak exchange (~12 s): best contiguous 12-second window anywhere
      Phase 3 — Quick moments (~4 s): top individual seconds from last 40%

    Short clips (≤20 s) are passed through unchanged.
    """
    N       = len(frames)
    fps_int = max(1, int(round(fps)))
    n_secs  = max(1, N // fps_int)

    TOTAL_CAP  = 20
    PEAK_SECS  = 12
    QUICK_EACH = (TOTAL_CAP - PEAK_SECS) // 2   # 4 s each side

    def _sec_combined(s):
        start, end = s * fps_int, min(N, (s + 1) * fps_int)
        total  = 0.0
        pb_my  = pb_op = None
        for fd in frames[start:end]:
            if not fd["my"].get("coast", 0):
                if pb_my: total += _dist(_cx_cy(pb_my), _cx_cy(fd["my"]["box"]))
                pb_my = fd["my"]["box"]
            if not fd["op"].get("coast", 0):
                if pb_op: total += _dist(_cx_cy(pb_op), _cx_cy(fd["op"]["box"]))
                pb_op = fd["op"]["box"]
        return total

    if n_secs <= TOTAL_CAP:
        picks = list(range(n_secs))
    else:
        sec_comb = [_sec_combined(s) for s in range(n_secs)]

        # Phase 2: best contiguous PEAK_SECS window
        win     = min(PEAK_SECS, n_secs)
        best_ws = max(range(n_secs - win + 1),
                      key=lambda ws: sum(sec_comb[ws : ws + win]), default=0)
        peak    = set(range(best_ws, best_ws + win))

        # Phase 1: top QUICK_EACH seconds from first 40%, not in peak
        front = [s for s in range(max(1, int(n_secs * 0.40))) if s not in peak]
        intro = sorted(sorted(front, key=lambda s: sec_comb[s], reverse=True)[:QUICK_EACH])

        # Phase 3: top QUICK_EACH seconds from last 40%, not in peak
        back  = [s for s in range(int(n_secs * 0.60), n_secs) if s not in peak]
        outro = sorted(sorted(back,  key=lambda s: sec_comb[s], reverse=True)[:QUICK_EACH])

        # Assemble: quick intro → peak → quick outro
        picks = intro + sorted(peak) + outro

    cap = cv2.VideoCapture(vpath)
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh))
    for s in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, s * fps_int)
        for _ in range(fps_int):
            ok, frame = cap.read()
            if not ok: break
            out.write(frame)
    cap.release()
    out.release()


# ─── Lab — standalone SAM2 pipeline ─────────────────────────────────────────
#
# Separate tab for verifying SAM2 works:
#   1. Upload video  →  compress to ≤1280px wide
#   2. Click ME + OPPONENT on the frame  →  save picker points
#   3. SAM2 runs (sam2_visualizer.py) in background
#   4. Show result video with coloured masks
#
# No Kalman, no pose estimation, no metrics — pure SAM2 smoke-test.
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/lab")
def lab_index():
    # Collect all lab sessions for the session list
    lab_sessions = [m for m in list_sessions() if m.get("lab_mode")]
    return render_template("lab.html", meta=None, lab_sessions=lab_sessions)


@app.route("/lab/upload", methods=["POST"])
def lab_upload():
    """
    Upload a boxing video for the Lab SAM2 pipeline.
    Compresses to ≤1280px wide, reads basic metadata, then redirects to the
    picker where the user scrubs the video to their desired start frame.
    """
    f = request.files.get("video")
    if not f:
        return "No file received", 400

    sid = uuid.uuid4().hex[:8]
    sess_dir(sid).mkdir(parents=True, exist_ok=True)

    ext = Path(f.filename).suffix.lower() or ".mp4"

    # Save to a staging path, then hand off to the content-hash cache.
    # If this video was uploaded before, all cached stages are reused.
    staging = sess_dir(sid) / f"_staging{ext}"
    f.save(str(staging))
    video_hash = pipeline_cache.ingest_upload(staging)

    # Compressed video is stage 0 of the pipeline — cache it if needed.
    comp_cache = pipeline_cache.stage_path(video_hash, "compressed", "mp4")
    if not comp_cache.exists():
        orig_in_cache = pipeline_cache.original_path(video_hash)
        _compress_video(str(orig_in_cache), str(comp_cache), max_width=1280)

    # Wire up session-dir symlinks so existing read paths keep working.
    pipeline_cache.link_into_session(video_hash, sess_dir(sid))

    vpath = str(comp_cache)
    cap     = cv2.VideoCapture(vpath)
    fps_r   = _safe(cap.get(cv2.CAP_PROP_FPS),            30.0)
    total_f = int(_safe(cap.get(cv2.CAP_PROP_FRAME_COUNT), 0))
    fw      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Stash the video-level facts in the cache meta so other sessions benefit.
    if pipeline_cache.read_video_meta(video_hash) is None:
        pipeline_cache.write_video_meta(video_hash, {
            "frame_size":   [fw, fh],
            "fps":          fps_r,
            "total_frames": total_f,
        })

    meta = dict(
        id               = sid,
        video_hash       = video_hash,
        filename         = f.filename,
        video_ext        = ext,
        upload_time      = datetime.utcnow().isoformat(),
        status           = "picking",
        lab_mode         = True,
        frame_size       = [fw, fh],
        fps              = fps_r,
        total_frames     = total_f,
        my_ref_center    = None,
        op_ref_center    = None,
        sam2_test_status = None,
        sam2_test_error  = None,
    )
    write_meta(sid, meta)
    return redirect(url_for("lab_session", sid=sid))


@app.route("/lab/upload_url", methods=["POST"])
def lab_upload_url():
    """
    Download a video from a URL (Vimeo, Instagram, etc.) for the Lab SAM2 pipeline.
    Same yt-dlp download logic as /upload_url, but routes to the lab picker.
    Returns JSON {sid} on success or {error} on failure.
    """
    data = request.get_json(silent=True) or {}
    url  = (data.get("url") or "").strip()
    if not url:
        return jsonify(error="No URL provided"), 400

    try:
        import yt_dlp
    except ImportError:
        return jsonify(error="yt-dlp is not installed — run: pip install yt-dlp"), 500

    _yt_hosts = ("youtube.com", "youtu.be", "www.youtube.com", "m.youtube.com")
    if any(h in url for h in _yt_hosts):
        return jsonify(error=(
            "YouTube downloads are currently blocked by YouTube's SABR streaming "
            "protection. Please download the video manually and upload the file instead."
        )), 400

    sid = uuid.uuid4().hex[:8]
    sess_dir(sid).mkdir(parents=True, exist_ok=True)

    out_tmpl  = str(sess_dir(sid) / "original.%(ext)s")
    base_opts = {
        "format":              "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
        "outtmpl":             out_tmpl,
        "merge_output_format": "mp4",
        "quiet":               True,
        "no_warnings":         True,
        "noplaylist":          True,
    }
    strategies = [
        {"extractor_args": {"youtube": {"player_client": ["ios"]}}},
        {"extractor_args": {"youtube": {"player_client": ["ios"]}},
         "cookiesfrombrowser": ("safari",)},
        {"cookiesfrombrowser": ("safari",)},
        {},
    ]

    info = None
    last_error = None
    for extra in strategies:
        opts = {**base_opts, **extra}
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
            break
        except Exception as e:
            last_error = e
            for f in sess_dir(sid).glob("original.*"):
                try: f.unlink()
                except Exception: pass

    if info is None:
        return jsonify(error=f"Download failed: {last_error}"), 400

    title    = info.get("title", "video")
    filename = f"{title[:60]}.mp4"

    vpath = sess_dir(sid) / "original.mp4"
    if not vpath.exists():
        candidates_f = list(sess_dir(sid).glob("original.*"))
        if not candidates_f:
            return jsonify(error="Download produced no output file"), 500
        vpath = candidates_f[0]

    ext = vpath.suffix.lower()

    # Hand off to the content-hash cache — reuses all stages if previously seen.
    video_hash = pipeline_cache.ingest_upload(vpath)

    comp_cache = pipeline_cache.stage_path(video_hash, "compressed", "mp4")
    if not comp_cache.exists():
        orig_in_cache = pipeline_cache.original_path(video_hash)
        _compress_video(str(orig_in_cache), str(comp_cache), max_width=1280)

    pipeline_cache.link_into_session(video_hash, sess_dir(sid))
    read_path = str(comp_cache)

    cap     = cv2.VideoCapture(read_path)
    fps_r   = _safe(cap.get(cv2.CAP_PROP_FPS),            30.0)
    total_f = int(_safe(cap.get(cv2.CAP_PROP_FRAME_COUNT), 0))
    fw      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if pipeline_cache.read_video_meta(video_hash) is None:
        pipeline_cache.write_video_meta(video_hash, {
            "frame_size":   [fw, fh],
            "fps":          fps_r,
            "total_frames": total_f,
        })

    meta = dict(
        id               = sid,
        video_hash       = video_hash,
        filename         = filename,
        video_ext        = ext,
        upload_time      = datetime.utcnow().isoformat(),
        status           = "picking",
        lab_mode         = True,
        frame_size       = [fw, fh],
        fps              = fps_r,
        total_frames     = total_f,
        my_ref_center    = None,
        op_ref_center    = None,
        sam2_test_status = None,
        sam2_test_error  = None,
    )
    write_meta(sid, meta)
    return jsonify(sid=sid)


@app.route("/lab/<sid>/refresh_arena", methods=["POST"])
def lab_refresh_arena(sid):
    """
    Recompute the arena polygon (and re-render its screenshots + the
    heatmap + diagnostics bundle) from already-cached data. Does NOT
    re-run SAM2, pose enrichment, or motion compensation — uses the
    warps already stored in arena.json.
    """
    meta = read_meta(sid)
    if meta is None:
        return jsonify({"error": "no session"}), 404

    session_dir = sess_dir(sid)
    arena = arena_detector.refresh_arena_polygon(session_dir)
    if arena is None:
        return jsonify({"error": "missing cache or empty arena — run the full pipeline first"}), 400

    # Locate the compressed video for screenshot rendering.
    vpath = session_dir / "lab_compressed.mp4"
    if not vpath.exists():
        return jsonify({"ok": True, "warning": "arena.json updated but video not found — screenshots not refreshed"})

    arena["fps_pose"] = arena.get("fps_pose") or (
        meta.get("arena_metrics", {}).get("fps_pose") or 15.0
    )

    try:
        shots_dir = session_dir / "screenshots" / "arena"
        n1 = screenshots.render_arena(str(vpath), arena, shots_dir, n=6)
        heat_dir = session_dir / "screenshots" / "arena_heatmap"
        n2 = screenshots.render_arena_heatmap(str(vpath), arena, heat_dir)
        _rebuild_session_diagnostics(sid)
        return jsonify({"ok": True, "arena_shots": n1, "heatmap_shots": n2,
                        "n_foot_points": arena.get("n_foot_points", 0)})
    except Exception as e:
        return jsonify({"error": f"refresh succeeded but rendering failed: {e}"}), 500


@app.route("/lab/<sid>/screenshots/<stage>/list")
def lab_screenshots_list(sid, stage):
    """Return JSON list of screenshot filenames for a given pipeline stage."""
    meta = read_meta(sid)
    if meta is None:
        return jsonify({"error": "No session"}), 404

    # Stage → cache location. Video-content stages live in cache/videos/<hash>/,
    # session-scoped stages live in sessions_data/<sid>/.
    # All current stages are session-scoped. (Video-content-hash stages are
    # no longer in the stage list; add handling back here if we reintroduce
    # them later.)
    stage_dir = sess_dir(sid) / "screenshots" / stage

    files = screenshots.list_screenshots(stage_dir)
    return jsonify({"stage": stage, "files": files})


@app.route("/lab/<sid>/screenshots/<stage>/<fname>")
def lab_screenshots_file(sid, stage, fname):
    """Serve a single screenshot PNG."""
    # Prevent path escape.
    if "/" in fname or "\\" in fname or ".." in fname:
        return "Bad filename", 400
    meta = read_meta(sid)
    if meta is None:
        return "No session", 404

    p = sess_dir(sid) / "screenshots" / stage / fname

    if not p.exists():
        return "Not found", 404
    return send_file(str(p), mimetype="image/png", conditional=True)


@app.route("/lab/<sid>/diagnostics.json")
def lab_diagnostics_json(sid):
    """Serve the per-frame diagnostic overlay bundle for the Lab viewer."""
    # Always rebuild from current cache state so newly-finished background
    # stages (e.g. scene-reference) become visible without a re-upload.
    _rebuild_session_diagnostics(sid)
    p = sess_dir(sid) / "diagnostics.json"
    if not p.exists():
        return jsonify({"error": "No diagnostics yet"}), 404
    return send_file(str(p), mimetype="application/json", conditional=True)


def _rebuild_session_diagnostics(sid: str) -> None:
    """
    Rebuild diagnostics.json for a session by merging all available
    per-stage diagnostic layers for its video.
    """
    session_dir = sess_dir(sid)

    # Start from whatever baseline diagnostics we have (fighter boxes).
    diagnostics.refresh_from_enriched(session_dir)

    bundle_path = session_dir / "diagnostics.json"
    if not bundle_path.exists():
        return
    try:
        bundle = json.loads(bundle_path.read_text())
    except Exception as e:
        print(f"[diag] read bundle failed: {e}")
        return

    # Arena overlay — merge in if computed for this session.
    arena_path = session_dir / "arena.json"
    if arena_path.exists():
        try:
            arena = json.loads(arena_path.read_text())
            diagnostics.merge_arena(bundle, arena)
        except Exception as e:
            print(f"[diag] merge arena failed: {e}")

    # Target zones + landed-hit flashes — requires sam2_enriched + metrics.
    enriched_path = session_dir / "sam2_enriched.json"
    metrics_path  = session_dir / "arena_metrics.json"
    if enriched_path.exists() and metrics_path.exists():
        try:
            enriched = json.loads(enriched_path.read_text())
            arena_metrics = json.loads(metrics_path.read_text())
            if arena_metrics.get("ok"):
                diagnostics.merge_landed_hits(bundle, enriched, arena_metrics)
        except Exception as e:
            print(f"[diag] merge landed hits failed: {e}")

    try:
        bundle_path.write_text(json.dumps(bundle, separators=(",", ":")))
    except Exception as e:
        print(f"[diag] write bundle failed: {e}")


@app.route("/lab/video/<sid>")
def lab_video(sid):
    """Stream the compressed lab video (supports Range requests for browser scrubbing)."""
    for name in ("lab_compressed.mp4", "compressed.mp4"):
        p = sess_dir(sid) / name
        if p.exists():
            return send_file(str(p), mimetype="video/mp4", conditional=True)
    ext = (read_meta(sid) or {}).get("video_ext", ".mp4")
    p = sess_dir(sid) / f"original{ext}"
    if p.exists():
        return send_file(str(p), mimetype="video/mp4", conditional=True)
    return "Video not found", 404


@app.route("/lab/<sid>")
def lab_session(sid):
    meta = read_meta(sid)
    if meta is None:
        return "Session not found", 404
    return render_template("lab.html", meta=meta)


@app.route("/lab/<sid>/pick", methods=["POST"])
def lab_pick(sid):
    """
    Receive the user-locked frame time + ME/OP click coordinates.
    The user scrubbed the video to a frame where both boxers are clearly visible,
    locked it, then clicked on ME and OP.  We convert the timestamp to a SAM2
    frame index, snap the clicks to the nearest YOLO-detected person on that
    exact frame, and kick off the SAM2 subprocess.
    """
    meta = read_meta(sid)
    if meta is None:
        return "Session not found", 404

    try:
        frame_time = float(request.form["frame_time"])   # seconds into the video
        my_x = float(request.form["my_x"])
        my_y = float(request.form["my_y"])
        op_x = float(request.form["op_x"])
        op_y = float(request.form["op_y"])
    except (KeyError, ValueError) as e:
        return f"Invalid picker data: {e}", 400

    meta["my_ref_center"] = [int(my_x), int(my_y)]
    meta["op_ref_center"] = [int(op_x), int(op_y)]

    # ── Convert frame_time → SAM2 frame index ────────────────────────────────
    fps       = meta.get("fps", 30.0)
    total_f   = meta.get("total_frames", 0)
    # TODO(production): restore / 6.0
    stride_kf = max(1, round(fps / 3.0))       # TEMP: 3 fps for M2 Air testing
    raw_fi    = int(frame_time * fps)          # frame index in compressed video
    sam2_fi   = raw_fi // stride_kf            # index as SAM2 sees it (after stride)

    meta["lab_seed_fi"]     = sam2_fi
    meta["lab_seed_stride"] = stride_kf
    meta["lab_seed_raw_fi"] = raw_fi
    meta["lab_seed_time_s"] = round(frame_time, 1)

    # ── Snap clicks to nearest YOLO box on the locked frame ──────────────────
    # Gives SAM2 a tight bounding box rather than a single point for seeding.
    for candidate in ("lab_compressed.mp4", "compressed.mp4"):
        _c = sess_dir(sid) / candidate
        if _c.exists():
            vpath = str(_c)
            break
    else:
        vpath = str(sess_dir(sid) / f"original{meta['video_ext']}")

    meta["lab_my_box"] = None
    meta["lab_op_box"] = None
    try:
        cap = cv2.VideoCapture(vpath)
        cap.set(cv2.CAP_PROP_POS_FRAMES, raw_fi)
        ok, locked_frame = cap.read()
        cap.release()
        if ok:
            res = get_model()(locked_frame, verbose=False)[0]
            raw_boxes = []
            if res.boxes is not None:
                for box in res.boxes:
                    if int(box.cls[0]) == 0:
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                        raw_boxes.append([x1, y1, x2, y2])
            if len(raw_boxes) >= 2:
                def _nearest(cx, cy, boxes):
                    return min(boxes,
                               key=lambda b: (cx-(b[0]+b[2])/2)**2 + (cy-(b[1]+b[3])/2)**2)
                my_box = _nearest(my_x, my_y, raw_boxes)
                op_box = _nearest(op_x, op_y, raw_boxes)
                if my_box != op_box:
                    meta["lab_my_box"] = my_box
                    meta["lab_op_box"] = op_box
    except Exception as _e:
        print(f"[lab] YOLO snap on locked frame failed (non-fatal): {_e}")

    meta["sam2_test_status"] = "running"
    meta["sam2_test_error"]  = None
    write_meta(sid, meta)

    threading.Thread(target=_run_sam2_subprocess, args=(sid,), daemon=True).start()
    return redirect(url_for("lab_session", sid=sid))


@app.route("/lab/<sid>/reset", methods=["POST"])
def lab_reset(sid):
    """Clear picker points + SAM2 result so the user can re-pick and re-run."""
    meta = read_meta(sid)
    if meta is None:
        return "Session not found", 404
    meta["my_ref_center"]    = None
    meta["op_ref_center"]    = None
    meta["sam2_test_status"] = None
    meta["sam2_test_error"]  = None
    for key in ("lab_seed_fi", "lab_seed_stride", "lab_seed_raw_fi", "lab_seed_time_s",
                "lab_my_box", "lab_op_box"):
        meta.pop(key, None)
    for fname in ("sam2_test.mp4",):
        p = sess_dir(sid) / fname
        if p.exists():
            p.unlink(missing_ok=True)
    write_meta(sid, meta)
    return redirect(url_for("lab_session", sid=sid))


@app.route("/lab/<sid>/correct", methods=["POST"])
def lab_correct(sid):
    """
    Receive a correction seed (frame_time + ME/OP clicks) from the result page.
    Trims the existing output at the correction point, re-tracks from there,
    then stitches the two halves back into sam2_test.mp4.
    """
    meta = read_meta(sid)
    if meta is None:
        return "Session not found", 404

    try:
        frame_time = float(request.form["frame_time"])
        my_x = float(request.form["my_x"])
        my_y = float(request.form["my_y"])
        op_x = float(request.form["op_x"])
        op_y = float(request.form["op_y"])
    except (KeyError, ValueError) as e:
        return f"Invalid correction data: {e}", 400

    fps       = meta.get("fps", 30.0)
    stride_kf = meta.get("lab_seed_stride") or max(1, round(fps / 3.0))
    fps_out   = fps / stride_kf
    raw_fi    = int(frame_time * fps)
    sam2_fi   = raw_fi // stride_kf
    time_out  = sam2_fi / fps_out      # time in the output video at correction point

    meta["correction_frame_time"] = round(frame_time, 3)
    meta["correction_sam2_fi"]    = sam2_fi
    meta["correction_raw_fi"]     = raw_fi
    meta["correction_time_out"]   = round(time_out, 3)
    meta["correction_my_ref"]     = [int(my_x), int(my_y)]
    meta["correction_op_ref"]     = [int(op_x), int(op_y)]
    meta["correction_my_box"]     = None
    meta["correction_op_box"]     = None

    # ── YOLO snap on the correction frame ────────────────────────────────────
    for candidate in ("lab_compressed.mp4", "compressed.mp4"):
        _c = sess_dir(sid) / candidate
        if _c.exists():
            vpath_snap = str(_c)
            break
    else:
        vpath_snap = str(sess_dir(sid) / f"original{meta['video_ext']}")

    try:
        cap = cv2.VideoCapture(vpath_snap)
        cap.set(cv2.CAP_PROP_POS_FRAMES, raw_fi)
        ok, locked_frame = cap.read()
        cap.release()
        if ok:
            res = get_model()(locked_frame, verbose=False)[0]
            raw_boxes = []
            if res.boxes is not None:
                for box in res.boxes:
                    if int(box.cls[0]) == 0:
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                        raw_boxes.append([x1, y1, x2, y2])
            if len(raw_boxes) >= 2:
                def _nearest(cx, cy, boxes):
                    return min(boxes,
                               key=lambda b: (cx-(b[0]+b[2])/2)**2 + (cy-(b[1]+b[3])/2)**2)
                my_box = _nearest(my_x, my_y, raw_boxes)
                op_box = _nearest(op_x, op_y, raw_boxes)
                if my_box != op_box:
                    meta["correction_my_box"] = my_box
                    meta["correction_op_box"] = op_box
    except Exception as _e:
        print(f"[lab_correct] YOLO snap failed (non-fatal): {_e}")

    meta["sam2_test_status"] = "correcting"
    meta["sam2_test_error"]  = None
    write_meta(sid, meta)

    threading.Thread(target=_run_sam2_correction, args=(sid,), daemon=True).start()
    return redirect(url_for("lab_session", sid=sid))


def _detect_label_swap(frames):
    """
    Scan per-frame bbox data from sam2_track.json for the first persistent label swap.

    A swap is flagged when swapping ME and OPPONENT assignments is significantly
    cheaper (by centroid distance) than keeping them as-is, consecutively for
    SWAP_WINDOW frames.  Clinch frames (centroids too close) are skipped.

    Returns the index into `frames` where the swap first starts, or None.
    """
    SWAP_WINDOW = 5    # consecutive frames that must vote "swap" before we commit
    SWAP_RATIO  = 0.65  # cost_swap < cost_keep * ratio → swap is clearly better

    def centroid(b):
        if not b:
            return None
        return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

    def dist(a, b):
        if not a or not b:
            return float("inf")
        return math.hypot(a[0] - b[0], a[1] - b[1])

    swap_votes = [False] * len(frames)
    for i in range(1, len(frames)):
        prev, curr = frames[i - 1], frames[i]
        pm = centroid(prev.get("my_bbox"))
        po = centroid(prev.get("op_bbox"))
        cm = centroid(curr.get("my_bbox"))
        co = centroid(curr.get("op_bbox"))
        if not all([pm, po, cm, co]):
            continue
        # Skip clinch frames: centroids within (combined half-width) of each other
        my_w = max((curr["my_bbox"][2] - curr["my_bbox"][0]), 1)
        op_w = max((curr["op_bbox"][2] - curr["op_bbox"][0]), 1)
        if dist(cm, co) < (my_w + op_w) / 4.0:
            continue
        cost_keep = dist(pm, cm) + dist(po, co)
        cost_swap = dist(pm, co) + dist(po, cm)
        if cost_keep > 0 and cost_swap < cost_keep * SWAP_RATIO:
            swap_votes[i] = True

    consecutive = 0
    for i, vote in enumerate(swap_votes):
        if vote:
            consecutive += 1
            if consecutive >= SWAP_WINDOW:
                return i - SWAP_WINDOW + 1
        else:
            consecutive = 0
    return None


@app.route("/lab/<sid>/detect_swap", methods=["POST"])
def lab_detect_swap(sid):
    """
    Read the per-frame tracking sidecar, detect first persistent label swap,
    and auto-trigger a correction run with swapped references.
    """
    meta = read_meta(sid)
    if meta is None:
        return jsonify({"error": "Session not found"}), 404
    if meta.get("sam2_test_status") not in ("done",):
        return jsonify({"error": "Tracking not complete yet"}), 400

    track_path = sess_dir(sid) / "sam2_track.json"
    if not track_path.exists():
        return jsonify({"error": "No tracking sidecar found — re-run tracking first"}), 400

    try:
        frames = json.loads(track_path.read_text())
    except Exception as e:
        return jsonify({"error": f"Could not read track JSON: {e}"}), 500

    swap_idx = _detect_label_swap(frames)
    if swap_idx is None:
        return jsonify({"swap_found": False,
                        "message": "No persistent label swap detected."})

    swap_frame = frames[swap_idx]
    swap_time  = swap_frame.get("time_s", 0.0)
    sam2_fi    = swap_frame.get("sam2_fi", 0)
    raw_fi     = swap_frame.get("raw_fi", 0)

    # At the swap point the ME track is following the actual opponent and vice
    # versa, so we flip the reference centres.
    my_bbox_at_swap = swap_frame.get("my_bbox")   # actually the OP's body
    op_bbox_at_swap = swap_frame.get("op_bbox")   # actually ME's body

    def _cx(b):
        return int((b[0] + b[2]) / 2) if b else None
    def _cy(b):
        return int((b[1] + b[3]) / 2) if b else None

    # sam2_track.json bboxes are in SAM2's downscaled spatial coordinate space
    # (the visualizer downscales frames to max_dim=1024 before processing).
    # --my_pt / --op_pt are expected in the *original compressed-video* coordinate
    # space; the visualizer then scales them internally (orig * scale → SAM2 space).
    # So we must divide by scale to convert SAM2-space centroids → orig coords.
    scale = meta.get("sam2_test_scale", 1.0) or 1.0
    inv   = (1.0 / scale) if scale > 0 else 1.0

    def _cx_orig(b): return int(_cx(b) * inv) if _cx(b) is not None else None
    def _cy_orig(b): return int(_cy(b) * inv) if _cy(b) is not None else None
    def _bbox_orig(b):
        if not b: return None
        return [int(b[0]*inv), int(b[1]*inv), int(b[2]*inv), int(b[3]*inv)]

    # Correction point: the corrected "ME" ref is where the OP track was (and v.v.)
    corr_my_ref = [_cx_orig(op_bbox_at_swap), _cy_orig(op_bbox_at_swap)]  # actual ME
    corr_op_ref = [_cx_orig(my_bbox_at_swap), _cy_orig(my_bbox_at_swap)]  # actual OP

    if None in corr_my_ref or None in corr_op_ref:
        return jsonify({"error": "Swap frame has missing bboxes — cannot auto-correct"}), 400

    # Derive output-video timestamp: sam2_fi in output-fps time
    fps_kf    = meta.get("fps", 30.0)
    stride_kf = meta.get("lab_seed_stride") or max(1, round(fps_kf / 3.0))
    fps_out   = fps_kf / stride_kf
    t_out     = sam2_fi / fps_out

    meta["correction_sam2_fi"]  = sam2_fi
    meta["correction_raw_fi"]   = raw_fi
    meta["correction_time_out"] = round(t_out, 4)
    meta["correction_my_ref"]   = corr_my_ref
    meta["correction_op_ref"]   = corr_op_ref
    meta["correction_my_box"]   = _bbox_orig(op_bbox_at_swap)   # swapped, orig coords
    meta["correction_op_box"]   = _bbox_orig(my_bbox_at_swap)   # swapped, orig coords
    meta["sam2_test_status"]    = "correcting"
    write_meta(sid, meta)

    t = threading.Thread(target=_run_sam2_correction, args=(sid,), daemon=True)
    t.start()

    return jsonify({
        "swap_found":  True,
        "swap_at_s":   round(swap_time, 2),
        "swap_sam2_fi": sam2_fi,
        "message":     f"Label swap detected at {swap_time:.1f}s — correction started.",
    })


@app.route("/lab/<sid>/recompute_metrics", methods=["POST"])
def lab_recompute_metrics(sid):
    """
    Re-run YOLO-pose enrichment at 15fps + metrics v2 without re-running SAM2.
    Useful after code changes to test new metrics on the same tracking data.
    """
    meta = read_meta(sid)
    if meta is None:
        return jsonify({"error": "Session not found"}), 404
    if meta.get("sam2_test_status") not in ("done",):
        return jsonify({"error": "Tracking not complete yet"}), 400

    track_path = sess_dir(sid) / "sam2_track.json"
    if not track_path.exists():
        return jsonify({"error": "No tracking sidecar — re-run tracking first"}), 400

    meta["sam2_test_status"] = "enriching"
    write_meta(sid, meta)

    def _bg():
        try:
            _run_pose_enrichment(sid)
            _compute_sam2_metrics(sid)
        except Exception as e:
            print(f"[recompute] Failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            m = read_meta(sid)
            if m:
                m["sam2_test_status"] = "done"
                write_meta(sid, m)

    t = threading.Thread(target=_bg, daemon=True)
    t.start()
    return jsonify({"status": "started",
                    "message": "Re-running enrichment at 15fps + metrics v2..."})


# ── Analysis video renderer ──────────────────────────────────────────────────
# Full-speed annotated video with bboxes, skeleton, labels, punch flashes.

# COCO skeleton connections
_SKEL_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # head
    (5, 6),                                     # shoulders
    (5, 7), (7, 9),                             # left arm
    (6, 8), (8, 10),                            # right arm
    (5, 11), (6, 12),                           # torso
    (11, 12),                                   # hips
    (11, 13), (13, 15),                         # left leg
    (12, 14), (14, 16),                         # right leg
]


def _render_analysis_video(sid):
    """
    Render a full-speed annotated video:
    - Bbox around each boxer (red = aggressing, blue = retreating under pressure, green = neutral)
    - ME / OP labels next to bbox
    - Head / stomach hit zones (green = safe, red = hit by opponent's wrist during punch)
    Aggression = punching OR (advancing while opponent retreating/cornered & not counter-punching)
    Retreat = stepping back while opponent is punching.
    Reads enriched data, re-computes per-frame signals, writes sam2_analysis.mp4.
    """
    import bisect

    meta = read_meta(sid)
    if meta is None:
        return False

    enriched_path = sess_dir(sid) / "sam2_enriched.json"
    if not enriched_path.exists():
        print(f"[analysis] No enriched data for {sid}")
        return False

    data = json.loads(enriched_path.read_text())
    frames_e = data["frames"]
    fps_pose = data["fps_pose"]
    fps_orig = data["fps_orig"]

    if len(frames_e) < 4:
        return False

    # Open source video
    for candidate in ("lab_compressed.mp4", "compressed.mp4"):
        _c = sess_dir(sid) / candidate
        if _c.exists():
            vpath = str(_c)
            break
    else:
        vpath = str(sess_dir(sid) / f"original{meta['video_ext']}")

    cap = cv2.VideoCapture(vpath)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_vid = cap.get(cv2.CAP_PROP_FPS) or fps_orig
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = str(sess_dir(sid) / "sam2_analysis.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps_vid, (fw, fh))

    KP_CONF = 0.35

    # ── Pre-compute per-enriched-frame signals ───────────────────────────
    def _sh_centroid(kps):
        if not kps:
            return None
        pts = [(kps[i][0], kps[i][1])
               for i in (5, 6) if kps[i][2] >= KP_CONF]
        if not pts:
            return None
        return (sum(p[0] for p in pts) / len(pts),
                sum(p[1] for p in pts) / len(pts))

    def _bbox_h(bbox):
        return (bbox[3] - bbox[1]) if bbox else 0.3

    def _wrist_positions(kps, sh):
        if not kps or not sh:
            return [None, None]
        out = []
        for wi in (9, 10):
            if kps[wi][2] >= KP_CONF:
                out.append((kps[wi][0] - sh[0], kps[wi][1] - sh[1]))
            else:
                out.append(None)
        return out

    def _wrist_extension(kps, sh, torso_h):
        if not kps or not sh or torso_h < 0.01:
            return 0.0
        mx = 0.0
        for wi in (9, 10):
            if kps[wi][2] >= KP_CONF:
                dx = kps[wi][0] - sh[0]
                dy = kps[wi][1] - sh[1]
                mx = max(mx, math.hypot(dx, dy) / torso_h)
        return mx

    def _wrist_accel(wc, wp):
        mx = 0.0
        for a, b in zip(wc, wp):
            if a is not None and b is not None:
                mx = max(mx, math.hypot(a[0] - b[0], a[1] - b[1]))
        return mx

    def _bcenter(bbox):
        if not bbox:
            return None
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    # Punch detection is heuristic-only.

    _heuristic_punches = {}
    all_h = []
    _all_ext = []
    _all_acc = []
    for _i, _f in enumerate(frames_e):
        _prev = frames_e[_i - 1] if _i > 0 else None
        for _pfx in ("my", "op"):
            _kps = _f.get(f"{_pfx}_kps")
            _sh  = _sh_centroid(_kps)
            _th  = _bbox_h(_f.get(f"{_pfx}_bbox"))
            _pk  = _prev.get(f"{_pfx}_kps") if _prev else None
            _psh = _sh_centroid(_pk) if _prev else None
            _w   = _wrist_positions(_kps, _sh)
            _wp  = _wrist_positions(_pk, _psh) if _prev else [None, None]
            _e   = _wrist_extension(_kps, _sh, _th)
            _a   = _wrist_accel(_w, _wp)
            b = _f.get(f"{_pfx}_bbox")
            if b:
                all_h.append(b[3] - b[1])
            if _e > 0:
                _all_ext.append(_e)
            if _a > 0:
                _all_acc.append(_a)
    median_h = sorted(all_h)[len(all_h) // 2] if all_h else 0.3
    _all_ext.sort(); _all_acc.sort()
    ext_p90 = _all_ext[int(len(_all_ext) * 0.90)] if _all_ext else 0.30
    acc_p90 = _all_acc[int(len(_all_acc) * 0.90)] if _all_acc else 0.10
    accel_floor = median_h * 0.03
    def _is_punch_h(ext, accel):
        if accel < accel_floor: return False
        return (0.30*(ext/max(ext_p90,0.01))+0.70*(accel/max(acc_p90,0.001))) > 1.10
    # Compute per-enriched-frame signals (including per-frame movement)
    ADVANCE_THRESH_VID = 0.003
    frame_signals = []
    prev_my_cx = None
    prev_op_cx = None
    for i, f in enumerate(frames_e):
        prev = frames_e[i - 1] if i > 0 else None
        sig = {"clinch": f.get("clinch", False), "time_s": f["time_s"]}
        for pfx in ("my", "op"):
            kps = f.get(f"{pfx}_kps")
            sh = _sh_centroid(kps)
            th = _bbox_h(f.get(f"{pfx}_bbox"))
            pk = prev.get(f"{pfx}_kps") if prev else None
            psh = _sh_centroid(pk) if prev else None
            w = _wrist_positions(kps, sh)
            wp = _wrist_positions(pk, psh) if prev else [None, None]
            ext = _wrist_extension(kps, sh, th)
            acc = _wrist_accel(w, wp)
            sig[f"{pfx}_punch"] = _is_punch_h(ext, acc)
            sig[f"{pfx}_kps"] = kps
            sig[f"{pfx}_bbox"] = f.get(f"{pfx}_bbox")

        # Per-frame advancing/retreating
        my_c = _bcenter(f.get("my_bbox"))
        op_c = _bcenter(f.get("op_bbox"))
        my_cx = my_c[0] if my_c else None
        op_cx = op_c[0] if op_c else None

        for pfx, opfx, cur_cx, prev_cx_val, opp_cx in [
            ("my", "op", my_cx, prev_my_cx, op_cx),
            ("op", "my", op_cx, prev_op_cx, my_cx),
        ]:
            advancing = False
            retreating = False
            if prev_cx_val is not None and cur_cx is not None and opp_cx is not None:
                toward_op = 1.0 if opp_cx > cur_cx else -1.0
                dx = cur_cx - prev_cx_val
                advancing = dx * toward_op > ADVANCE_THRESH_VID
                retreating = dx * toward_op < -ADVANCE_THRESH_VID
            sig[f"{pfx}_advancing"] = advancing
            sig[f"{pfx}_retreating"] = retreating
        # Cornered check
        sig["my_cornered"] = my_cx is not None and (my_cx < 0.18 or my_cx > 0.82)
        sig["op_cornered"] = op_cx is not None and (op_cx < 0.18 or op_cx > 0.82)

        # Aggressing / retreating-under-pressure (punch-driven, like old YOLO)
        for pfx, opfx in [("my", "op"), ("op", "my")]:
            my_pch = sig[f"{pfx}_punch"]
            my_adv = sig[f"{pfx}_advancing"]
            op_corn = sig[f"{opfx}_cornered"]
            op_ret = sig[f"{opfx}_retreating"]
            op_pch = sig[f"{opfx}_punch"]
            sig[f"{pfx}_aggressing"] = my_pch or (my_adv and (op_corn or (op_ret and not op_pch)))
            # Retreat = I'm moving backward AND opponent is punching
            sig[f"{pfx}_retreat"] = sig[f"{pfx}_retreating"] and op_pch

        prev_my_cx = my_cx
        prev_op_cx = op_cx
        frame_signals.append(sig)

    # (advance/retreat is now computed per-frame inside frame_signals above)

    # ── Index enriched frames by raw_fi for lookup ───────────────────────
    ef_raw_fis = [f["raw_fi"] for f in frames_e]

    def _nearest_enriched_idx(raw_fi):
        idx = bisect.bisect_left(ef_raw_fis, raw_fi)
        if idx == 0:
            return 0
        if idx >= len(ef_raw_fis):
            return len(ef_raw_fis) - 1
        if (ef_raw_fis[idx] - raw_fi) <= (raw_fi - ef_raw_fis[idx - 1]):
            return idx
        return idx - 1

    # ── Drawing helpers ──────────────────────────────────────────────────
    COL_ATTACK  = (0, 0, 255)     # red in BGR
    COL_RETREAT = (255, 140, 0)   # blue-ish in BGR
    COL_NEUTRAL = (0, 180, 0)     # green in BGR
    COL_ME_LBL  = (0, 140, 255)   # orange in BGR
    COL_OP_LBL  = (255, 140, 30)  # blue in BGR
    # (skeleton and wrist flash removed per user request)

    def _denorm_bbox(nb):
        """[0,1] → pixel coords."""
        if not nb:
            return None
        return (int(nb[0] * fw), int(nb[1] * fh),
                int(nb[2] * fw), int(nb[3] * fh))

    def _denorm_kp(kp):
        """[x_norm, y_norm, conf] → (px, py, conf)."""
        return (int(kp[0] * fw), int(kp[1] * fh), kp[2])

    def _draw_label(img, text, x, y, color, bg_color=(0, 0, 0)):
        """Draw text label with background."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        thick = 2
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
        cv2.rectangle(img, (x, y - th - 6), (x + tw + 8, y + 4), bg_color, -1)
        cv2.putText(img, text, (x + 4, y - 2), font, scale, color, thick,
                    cv2.LINE_AA)

    # ── Hit-detection screenshot folder ─────────────────────────────────
    hits_dir = sess_dir(sid) / "hit_screenshots"
    if hits_dir.exists():
        import shutil
        shutil.rmtree(hits_dir)
    hits_dir.mkdir(parents=True, exist_ok=True)
    hit_screenshot_count = 0

    # ── Main rendering loop ──────────────────────────────────────────────
    fi = 0
    _max_fi = int(fps_vid * 300)  # 5 min cap
    written = 0

    print(f"[analysis] Rendering analysis video: {fw}x{fh} @ {fps_vid:.1f}fps")

    while True:
        ok, frame = cap.read()
        if not ok or fi > _max_fi:
            break

        # Find nearest enriched frame
        ei = _nearest_enriched_idx(fi)
        ef = frames_e[ei]
        sig = frame_signals[ei]

        for pfx, opfx, lbl, lbl_col in [
            ("my", "op", "ME", COL_ME_LBL),
            ("op", "my", "OP", COL_OP_LBL),
        ]:
            bbox = _denorm_bbox(ef.get(f"{pfx}_bbox"))
            kps = sig.get(f"{pfx}_kps")
            is_aggressing = sig.get(f"{pfx}_aggressing", False)
            is_retreat = sig.get(f"{pfx}_retreat", False)

            # Choose bbox colour based on state
            if is_aggressing:
                box_col = COL_ATTACK
            elif is_retreat:
                box_col = COL_RETREAT
            else:
                box_col = COL_NEUTRAL

            # Draw bbox
            if bbox:
                x1, y1, x2, y2 = bbox
                thickness = 3 if is_aggressing else 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_col, thickness,
                              cv2.LINE_AA)

                # Draw label above bbox
                _draw_label(frame, lbl, x1, y1 - 4, lbl_col)

                # State indicator below bbox
                if is_aggressing:
                    _draw_label(frame, "ATTACK", x1, y2 + 18, COL_ATTACK)
                elif is_retreat:
                    _draw_label(frame, "RETREAT", x1, y2 + 18, COL_RETREAT)

            # ── Head / Stomach hit zones ─────────────────────────────
            # Head zone: centered square on the head.
            #   Bottom = shoulder level, top = top of bbox.
            #   Width = max(height, bbox_width * 0.45) so it's never
            #   too narrow when seen from the side.
            # Stomach zone: armpits (20% below shoulder y) to hips.
            # "Effective" hit: opponent's wrist inside zone AND NOT
            #   blocked by defender's own wrist/elbow.
            op_punching = sig.get(f"{opfx}_punch", False)
            op_kps = sig.get(f"{opfx}_kps")

            if kps and bbox:
                pts = [_denorm_kp(k) for k in kps]
                bx1, by1, bx2, by2 = bbox  # pixel bbox
                bbox_w = bx2 - bx1

                # ── Head zone (centered square, min-width enforced) ──
                head_kp_pts = [(pts[i][0], pts[i][1]) for i in range(5)
                               if pts[i][2] >= KP_CONF]
                sh_ys = [pts[i][1] for i in (5, 6) if pts[i][2] >= KP_CONF]

                if head_kp_pts and sh_ys:
                    hcx = sum(p[0] for p in head_kp_pts) // len(head_kp_pts)
                    head_bottom = min(sh_ys)           # shoulder level
                    head_top    = by1                   # top of person bbox
                    head_h = max(head_bottom - head_top, 20)
                    # Minimum width: ~45% of bbox width (body-wide even from side)
                    min_w = int(bbox_w * 0.45)
                    half_side = max(head_h // 2, min_w // 2)
                    head_rect = (hcx - half_side, head_top,
                                 hcx + half_side, head_bottom)
                else:
                    head_rect = None

                # ── Stomach zone (armpits to hips) ───────────────────
                # Armpit ≈ 20% below shoulder toward hips
                sh_xs = [pts[i][0] for i in (5, 6) if pts[i][2] >= KP_CONF]
                hip_ys = [pts[i][1] for i in (11, 12) if pts[i][2] >= KP_CONF]

                if sh_xs and sh_ys and hip_ys:
                    shoulder_y = min(sh_ys)
                    hip_y = max(hip_ys)
                    # Armpit is ~20% of the shoulder-to-hip distance below shoulders
                    armpit_offset = int((hip_y - shoulder_y) * 0.20)
                    stom_top = shoulder_y + armpit_offset

                    stom_left  = min(sh_xs)
                    stom_right = max(sh_xs)
                    # Enforce minimum width (body-wide even from side)
                    stom_w = stom_right - stom_left
                    min_stom_w = int(bbox_w * 0.40)
                    if stom_w < min_stom_w:
                        stom_cx = (stom_left + stom_right) // 2
                        stom_left  = stom_cx - min_stom_w // 2
                        stom_right = stom_cx + min_stom_w // 2

                    pad_x = max(4, int((stom_right - stom_left) * 0.10))
                    stomach_rect = (stom_left - pad_x, stom_top,
                                    stom_right + pad_x, hip_y)
                else:
                    stomach_rect = None

                # ── Effective-hit check ──────────────────────────────
                head_hit = False
                stomach_hit = False
                head_blocked = False
                stomach_blocked = False

                if op_punching and op_kps:
                    op_pts = [_denorm_kp(k) for k in op_kps]

                    # Defender's arm-guard pixel positions
                    guard_pts = []
                    for gi in (7, 8, 9, 10):  # elbows + wrists
                        if pts[gi][2] >= KP_CONF:
                            guard_pts.append((pts[gi][0], pts[gi][1]))

                    for wi in (9, 10):
                        if op_pts[wi][2] < KP_CONF:
                            continue
                        wx, wy = op_pts[wi][0], op_pts[wi][1]

                        # Head zone check
                        if head_rect and (head_rect[0] <= wx <= head_rect[2]
                                          and head_rect[1] <= wy <= head_rect[3]):
                            block_r = max(20, head_rect[2] - head_rect[0]) * 0.30
                            blk = any(math.hypot(gx - wx, gy - wy) < block_r
                                      for gx, gy in guard_pts)
                            if blk:
                                head_blocked = True
                            else:
                                head_hit = True

                        # Stomach zone check
                        if stomach_rect and (stomach_rect[0] <= wx <= stomach_rect[2]
                                             and stomach_rect[1] <= wy <= stomach_rect[3]):
                            block_r_s = max(20, stomach_rect[2] - stomach_rect[0]) * 0.25
                            blk = any(math.hypot(gx - wx, gy - wy) < block_r_s
                                      for gx, gy in guard_pts)
                            if blk:
                                stomach_blocked = True
                            else:
                                stomach_hit = True

                # Draw zones
                COL_SAFE = (0, 180, 0)      # green
                COL_HIT  = (0, 0, 255)      # red
                if head_rect:
                    hcol = COL_HIT if head_hit else COL_SAFE
                    cv2.rectangle(frame, (head_rect[0], head_rect[1]),
                                  (head_rect[2], head_rect[3]), hcol, 2, cv2.LINE_AA)
                if stomach_rect:
                    scol = COL_HIT if stomach_hit else COL_SAFE
                    cv2.rectangle(frame, (stomach_rect[0], stomach_rect[1]),
                                  (stomach_rect[2], stomach_rect[3]), scol, 2, cv2.LINE_AA)

                # ── Save hit-detection screenshots ───────────────────
                if head_hit or head_blocked or stomach_hit or stomach_blocked:
                    hit_screenshot_count += 1
                    tags = []
                    if head_hit:       tags.append("head_LANDED")
                    if head_blocked:   tags.append("head_BLOCKED")
                    if stomach_hit:    tags.append("stomach_LANDED")
                    if stomach_blocked: tags.append("stomach_BLOCKED")
                    tag_str = "_".join(tags)
                    fname = f"f{fi:05d}_{pfx}_{tag_str}.jpg"
                    cv2.imwrite(str(hits_dir / fname), frame,
                                [cv2.IMWRITE_JPEG_QUALITY, 85])

        # Clinch indicator
        if sig.get("clinch"):
            _draw_label(frame, "CLINCH", fw // 2 - 40, 30, (0, 200, 255))

        writer.write(frame)
        written += 1
        fi += 1

        if written % 500 == 0:
            pct = round(fi / max(total_frames, 1) * 100)
            print(f"[analysis] {written} frames rendered ({pct}%)")

    cap.release()
    writer.release()
    print(f"[analysis] Saved {hit_screenshot_count} hit-detection screenshots → hit_screenshots/")

    # Re-encode with ffmpeg for browser compatibility (mp4v → h264)
    final_path = str(sess_dir(sid) / "sam2_analysis_h264.mp4")
    try:
        import subprocess
        subprocess.run([
            "ffmpeg", "-y", "-i", out_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            final_path,
        ], capture_output=True, timeout=300)
        Path(out_path).unlink(missing_ok=True)
        # Rename final to the standard name
        Path(final_path).rename(sess_dir(sid) / "sam2_analysis.mp4")
        print(f"[analysis] Done: {written} frames → sam2_analysis.mp4")
    except Exception as e:
        print(f"[analysis] ffmpeg re-encode failed ({e}), keeping mp4v version")
        # Keep the mp4v version as fallback

    return True


@app.route("/lab/<sid>/render_analysis", methods=["POST"])
def lab_render_analysis(sid):
    """Trigger background rendering of the annotated analysis video."""
    meta = read_meta(sid)
    if meta is None:
        return jsonify({"error": "Session not found"}), 404
    if meta.get("sam2_test_status") not in ("done",):
        return jsonify({"error": "Tracking not complete yet"}), 400

    enriched_path = sess_dir(sid) / "sam2_enriched.json"
    if not enriched_path.exists():
        return jsonify({"error": "No enriched data — run metrics first"}), 400

    meta["analysis_status"] = "rendering"
    write_meta(sid, meta)

    def _bg():
        try:
            _render_analysis_video(sid)
        except Exception as e:
            print(f"[analysis] Rendering failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            m = read_meta(sid)
            if m:
                m["analysis_status"] = "done" if (
                    sess_dir(sid) / "sam2_analysis.mp4").exists() else "error"
                write_meta(sid, m)

    t = threading.Thread(target=_bg, daemon=True)
    t.start()
    return jsonify({"status": "started",
                    "message": "Rendering analysis video…"})


@app.route("/lab/<sid>/analysis_video")
def lab_analysis_video(sid):
    """Serve the rendered analysis video."""
    p = sess_dir(sid) / "sam2_analysis.mp4"
    if not p.exists():
        return "Not found", 404
    return _range_response(p, "video/mp4")


def _run_sam2_correction(sid):
    """
    Background thread: re-tracks from the user-specified correction point and
    stitches the two halves of the output video back together.

    Segment 1 — kept from the existing sam2_test.mp4 (0 → correction point)
    Segment 2 — freshly tracked from the correction point onward
    Final     — seg1 + seg2 concatenated → replaces sam2_test.mp4
    """
    meta = read_meta(sid)
    if meta is None:
        return

    for candidate in ("lab_compressed.mp4", "compressed.mp4"):
        _c = sess_dir(sid) / candidate
        if _c.exists():
            vpath = str(_c)
            break
    else:
        vpath = str(sess_dir(sid) / f"original{meta['video_ext']}")

    correction_fi     = meta.get("correction_sam2_fi", 0)
    correction_raw_fi = meta.get("correction_raw_fi", 0)
    correction_t_out  = meta.get("correction_time_out", 0.0)
    my_ref = meta.get("correction_my_ref")
    op_ref = meta.get("correction_op_ref")
    my_box = meta.get("correction_my_box")
    op_box = meta.get("correction_op_box")

    if not my_ref or not op_ref:
        meta["sam2_test_status"] = "error"
        meta["sam2_test_error"]  = "No correction reference points."
        write_meta(sid, meta)
        return

    fps_kf    = meta.get("fps", 30.0)
    stride_kf = meta.get("lab_seed_stride") or max(1, round(fps_kf / 3.0))

    seg1_path    = str(sess_dir(sid) / "sam2_seg1.mp4")
    seg2_path    = str(sess_dir(sid) / "sam2_seg2.mp4")
    new_out_path = str(sess_dir(sid) / "sam2_correction.mp4")
    existing_out = str(sess_dir(sid) / "sam2_test.mp4")
    out_json     = str(sess_dir(sid) / "sam2_test_status.json")
    out_progress = str(sess_dir(sid) / "sam2_test_progress.json")
    out_track    = str(sess_dir(sid) / "sam2_track.json")   # overwrite with corrected tracking data
    log_path     = str(sess_dir(sid) / "sam2_correction.log")

    def _ffmpeg(*args):
        """Run an ffmpeg command, return True on success."""
        try:
            subprocess.run(["ffmpeg", "-y"] + list(args),
                           check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"[lab_correct] ffmpeg failed: {e.stderr.decode()[-500:]}")
            return False

    # ── Step 1: trim existing output to correction point (seg1) ──────────────
    seg1_ok = _ffmpeg(
        "-i", existing_out,
        "-t", f"{correction_t_out:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-an",
        seg1_path,
    )

    # ── Step 2: YOLO kf scan from correction point onward ────────────────────
    _MAX_DURATION_S = 300
    _max_raw_fi = correction_raw_fi + int(_MAX_DURATION_S * fps_kf)

    kf_path = str(sess_dir(sid) / "lab_yolo_kf_correction.json")
    try:
        yolo_model = get_model()
        cap_kf = cv2.VideoCapture(vpath)
        # Seek to the nearest stride-aligned frame at or before the correction
        seek_to = correction_raw_fi - (correction_raw_fi % stride_kf)
        cap_kf.set(cv2.CAP_PROP_POS_FRAMES, seek_to)
        fi_kf  = seek_to
        si_kf  = seek_to // stride_kf   # SAM2 frame index (matches visualizer's count)
        yolo_kf: dict = {}
        while True:
            ok, frm = cap_kf.read()
            if not ok or fi_kf >= _max_raw_fi:
                break
            if fi_kf % stride_kf == 0:
                res_kf  = yolo_model(frm, verbose=False)[0]
                dets_kf = []
                if res_kf.boxes is not None:
                    for b_kf in res_kf.boxes:
                        if int(b_kf.cls[0]) == 0:
                            x1k, y1k, x2k, y2k = [int(v) for v in b_kf.xyxy[0].tolist()]
                            conf_k = round(float(b_kf.conf[0]), 3)
                            if conf_k >= 0.30:
                                dets_kf.append([x1k, y1k, x2k, y2k, conf_k])
                yolo_kf[str(si_kf)] = dets_kf
                si_kf += 1
            fi_kf += 1
        cap_kf.release()
        Path(kf_path).write_text(json.dumps(yolo_kf))
    except Exception as _kf_err:
        kf_path = ""
        print(f"[lab_correct] YOLO kf scan failed (non-fatal): {_kf_err}")

    # ── Step 3: run sam2_visualizer with correction seed ─────────────────────
    model = "tiny"  # TODO(production): restore small
    cmd = [
        str(_SAM2_VENV), str(_SAM2_VIS_SCRIPT),
        "--video",    vpath,
        "--my_pt",    f"{my_ref[0]},{my_ref[1]}",
        "--op_pt",    f"{op_ref[0]},{op_ref[1]}",
        "--out",      new_out_path,
        "--status",   out_json,
        "--progress", out_progress,
        "--model",    model,
        "--max_dim",  "1024",
        "--seed_fi",  str(correction_fi),
        "--stride",   str(stride_kf),
    ]
    if my_box and op_box:
        cmd += ["--my_box", f"{my_box[0]},{my_box[1]},{my_box[2]},{my_box[3]}"]
        cmd += ["--op_box", f"{op_box[0]},{op_box[1]},{op_box[2]},{op_box[3]}"]
    if kf_path and Path(kf_path).exists():
        cmd += ["--yolo_kf", kf_path]
    cmd += ["--track_json", out_track]   # overwrite sidecar with corrected tracking data

    stderr_lines = []
    def _drain(proc, log_file):
        with open(log_file, "w") as lf:
            for line in proc.stderr:
                lf.write(line); lf.flush()
                stderr_lines.append(line)

    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            text=True, bufsize=1)
        drain_t = threading.Thread(target=_drain, args=(proc, log_path), daemon=True)
        drain_t.start()
        proc.wait()
        drain_t.join(timeout=5)

        if proc.returncode != 0:
            raise RuntimeError(f"sam2_visualizer exited {proc.returncode}")

        # ── Step 4: trim new output — keep only tracked portion (seg2) ───────
        seg2_ok = _ffmpeg(
            "-i", new_out_path,
            "-ss", f"{correction_t_out:.3f}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-an",
            seg2_path,
        )

        # ── Step 5: stitch seg1 + seg2 → replace sam2_test.mp4 ───────────────
        stitched_path = str(sess_dir(sid) / "sam2_stitched.mp4")
        stitch_ok = False
        if seg1_ok and seg2_ok:
            concat_list = str(sess_dir(sid) / "concat_list.txt")
            Path(concat_list).write_text(
                f"file '{seg1_path}'\nfile '{seg2_path}'\n"
            )
            stitch_ok = _ffmpeg(
                "-f", "concat", "-safe", "0", "-i", concat_list,
                "-c", "copy",
                stitched_path,
            )

        import shutil
        if stitch_ok and Path(stitched_path).exists():
            shutil.move(stitched_path, existing_out)
        elif seg2_ok and Path(seg2_path).exists():
            # Fallback: stitching failed, just replace with corrected portion
            shutil.move(seg2_path, existing_out)

        # Clean up temp files
        for tmp in (seg1_path, seg2_path, new_out_path, stitched_path):
            try: Path(tmp).unlink(missing_ok=True)
            except Exception: pass

        # ── Step 6: update meta ───────────────────────────────────────────────
        result = {}
        try:
            result = json.loads(Path(out_json).read_text())
        except Exception:
            pass

        meta = read_meta(sid)
        meta["sam2_test_frames"]     = result.get("frame_count")
        meta["sam2_test_my_tracked"] = result.get("my_tracked")
        meta["sam2_test_op_tracked"] = result.get("op_tracked")
        meta["sam2_test_fps"]        = result.get("fps_effective")
        meta["sam2_test_model"]      = result.get("model")

        # Re-run enrichment + metrics on corrected video
        meta["sam2_test_status"] = "enriching"
        write_meta(sid, meta)
        try:
            _run_pose_enrichment(sid)
            _compute_sam2_metrics(sid)
        except Exception as _enrich_err:
            print(f"[lab_correct] Enrichment/metrics failed (non-fatal): "
                  f"{_enrich_err}")
        meta = read_meta(sid)
        meta["sam2_test_status"] = "done"
        write_meta(sid, meta)

    except Exception as e:
        import traceback as _tb
        err_detail = f"{e}\n{''.join(stderr_lines[-30:])}"
        meta = read_meta(sid)
        meta["sam2_test_status"] = "error"
        meta["sam2_test_error"]  = f"Correction failed: {err_detail}"
        write_meta(sid, meta)


@app.route("/lab/frame/<sid>")
def lab_frame(sid):
    # Serve the annotated preview (dashed boxer outlines) if it exists,
    # otherwise fall back to the clean frame.
    for name in ("lab_preview.jpg", "lab_frame.jpg"):
        p = sess_dir(sid) / name
        if p.exists():
            return send_file(str(p), mimetype="image/jpeg")
    return "Not found", 404


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False, threaded=True)
