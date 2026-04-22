"""
sam2_visualizer.py  —  Diagnostic SAM2 test tool.

Runs SAM2 on the given video and produces an annotated MP4 where:
  • ME       → orange semi-transparent mask  +  solid orange bounding box + label
  • OPPONENT → blue   semi-transparent mask  +  solid blue   bounding box + label

Two-level memory management to avoid "Invalid buffer size" OOM:
  1. Spatial downscale  — frames are written to temp dir at max_dim on the
     longest side (default 1024).  This keeps each feature-map smaller.
  2. Temporal subsample — only every STRIDE-th frame is sent to SAM2 so that
     at most MAX_SAM2_FRAMES frames are loaded.  The output video is written at
     (original_fps / stride) so playback speed is unchanged.

Called by the Flask app via subprocess (inside sam2_venv/).

Usage:
    sam2_venv/bin/python sam2_visualizer.py \
        --video   /path/to/original.mp4  \
        --my_pt   x,y                    \
        --op_pt   x,y                    \
        --out     /path/to/sam2_test.mp4 \
        --status  /path/to/status.json   \
        --model   tiny                   \
        --max_dim 1024

status.json on success:
    {"ok": true, "frame_count": N, "fps_effective": F,
     "my_tracked": M, "op_tracked": O, "model": "...",
     "scale": 0.391, "stride": 4}
"""

import sys, json, time, math, tempfile, argparse, traceback
from pathlib import Path

import numpy as np
import cv2
import torch

# ── Memory budget ──────────────────────────────────────────────────────────────
# SAM2 stores per-frame feature tensors in CPU RAM.  Even at 1024-px resolution,
# a typical Hiera-tiny backbone needs ~13–15 MB per frame, so:
#   100 frames →  ~1.5 GB  (very safe, good temporal coverage)
#   300 frames →  ~4 GB   (safe on 8 GB machines)
#   600 frames →  ~8 GB   (tight on 8 GB machines)
#  1350 frames → ~17 GB   (the "Invalid buffer size" crash we're avoiding)
# Target output frame-rate for tracking.  A fixed fps gives consistent temporal
# coverage regardless of clip length — 6 fps captures boxer movement well while
# keeping processing time proportional to real duration, not total frame count.
# TODO(production): restore TARGET_TRACK_FPS = 6 before shipping
TARGET_TRACK_FPS  = 3       # TEMP: 3 fps for M2 Air testing (half the frame count)
MAX_DURATION_S    = 300     # cap at 5 minutes from the seed frame

# ─── Colours (BGR) ────────────────────────────────────────────────────────────
COL_MY = (0,   130, 255)   # orange — ME
COL_OP = (220,  60,   0)   # blue   — OPPONENT


def _label(frame, text, x1, y1, col, scale):
    """Draw a filled-background label above a bounding box."""
    font  = cv2.FONT_HERSHEY_DUPLEX
    thick = max(1, int(scale * 1.5))
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
    px = x1
    py = max(y1 - 8, th + 6)
    cv2.rectangle(frame,
                  (px - 4,       py - th - 4),
                  (min(px + tw + 4, frame.shape[1] - 1), py + baseline),
                  (0, 0, 0), -1)
    cv2.putText(frame, text, (px, py), font, scale, col, thick, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",    required=True)
    ap.add_argument("--my_pt",   required=True,  help="click centre for ME: x,y")
    ap.add_argument("--op_pt",   required=True,  help="click centre for OP: x,y")
    ap.add_argument("--out",     required=True,  help="output MP4 path")
    ap.add_argument("--status",  required=True,  help="output JSON status path")
    ap.add_argument("--progress", default=None,  help="live progress JSON path (optional)")
    ap.add_argument("--model",   default="tiny", choices=["tiny", "small"])
    ap.add_argument("--max_dim", default=1024,   type=int,
                    help="Downscale longest side to this (default 1024)")
    # YOLO-hybrid arguments (optional — improve accuracy when available)
    ap.add_argument("--my_box",  default=None,
                    help="YOLO bounding box for ME on picker frame: x1,y1,x2,y2")
    ap.add_argument("--op_box",  default=None,
                    help="YOLO bounding box for OP on picker frame: x1,y1,x2,y2")
    ap.add_argument("--yolo_kf", default=None,
                    help="JSON path: {sam2_frame_idx: [[x1,y1,x2,y2,conf],...]} "
                         "for periodic YOLO re-seeding during propagation")
    ap.add_argument("--stride",  default=None, type=int,
                    help="Temporal stride pre-computed by Flask (M2 fix: avoids "
                         "frame-count divergence between Flask and OpenCV reads)")
    ap.add_argument("--seed_fi", default=None, type=int,
                    help="Force SAM2 seeding from this specific SAM2 frame index "
                         "(user-confirmed via UI).  Skips the automatic scan.")
    ap.add_argument("--track_json", default=None,
                    help="Output path for per-frame bbox sidecar JSON "
                         "(used by automated label-swap detection).")
    args = ap.parse_args()

    try:
        _run(args)
    except Exception as e:
        err = {"ok": False, "error": str(e), "traceback": traceback.format_exc()}
        Path(args.status).write_text(json.dumps(err))
        print(f"[sam2_visualizer] FAILED: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)


def _run(args):
    # ── Device ────────────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    import os as _os
    mps_fallback = _os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    print(
        f"[sam2_visualizer] device={device}  "
        f"torch={torch.__version__}  "
        f"mps_built={torch.backends.mps.is_built()}  "
        f"mps_available={torch.backends.mps.is_available()}  "
        f"cuda={torch.cuda.is_available()}  "
        f"MPS_FALLBACK={mps_fallback}",
        file=sys.stderr, flush=True,
    )
    if mps_fallback == "1" and device.type == "mps":
        print(
            "[sam2_visualizer] WARNING: PYTORCH_ENABLE_MPS_FALLBACK=1 is set — "
            "unsupported MPS ops silently fall back to CPU, giving CPU-level speed. "
            "This is expected for SAM2 on MPS; real throughput is CPU-bound.",
            file=sys.stderr, flush=True,
        )
    # Maximise CPU thread count — helps whether running on CPU or hitting MPS fallbacks
    n_threads = max(1, _os.cpu_count() or 4)
    torch.set_num_threads(n_threads)
    print(f"[sam2_visualizer] CPU threads set to {n_threads}", file=sys.stderr, flush=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    SCRIPT_DIR = Path(__file__).parent
    CKPT_DIR   = SCRIPT_DIR / "sam2_checkpoints"
    cfg_map    = {
        "tiny":  ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
        "small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
    }
    cfg, ckpt_name = cfg_map[args.model]
    ckpt = CKPT_DIR / ckpt_name
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}  —  run:  bash setup_sam2.sh"
        )

    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(cfg, str(ckpt), device=device)
    # Limit the object-pointer memory bank fed back into the encoder each step.
    # Default is 16 (= attend to last 16 frames), which causes per-frame time to
    # grow until frame 16 on MPS.  Setting it to 8 is a good compromise: per-frame
    # time stabilises by frame ~8 while keeping reasonable temporal context for
    # tracking quality (4 was too aggressive and hurt segmentation accuracy).
    if hasattr(predictor, "max_obj_ptrs_in_encoder"):
        predictor.max_obj_ptrs_in_encoder = 8
    try:
        actual_device = next(predictor.parameters()).device
    except StopIteration:
        actual_device = "unknown"
    print(f"[sam2_visualizer] model loaded: {cfg}  actual_param_device={actual_device}",
          file=sys.stderr, flush=True)

    # ── Parse picker points (in the input video's coordinate space) ───────────
    my_x_orig, my_y_orig = map(float, args.my_pt.split(","))
    op_x_orig, op_y_orig = map(float, args.op_pt.split(","))

    # ── Open video: measure original dimensions + total frames ────────────────
    cap0      = cv2.VideoCapture(args.video)
    fps_orig  = cap0.get(cv2.CAP_PROP_FPS) or 30.0
    orig_fw   = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_fh   = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_raw = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT))
    cap0.release()

    # ── Spatial downscale factor ───────────────────────────────────────────────
    longest = max(orig_fw, orig_fh)
    scale   = min(1.0, args.max_dim / longest)
    new_fw  = max(1, int(orig_fw * scale))
    new_fh  = max(1, int(orig_fh * scale))
    new_fw  = new_fw  if new_fw  % 2 == 0 else new_fw  - 1
    new_fh  = new_fh  if new_fh  % 2 == 0 else new_fh  - 1
    print(f"[sam2_visualizer] spatial: {orig_fw}x{orig_fh} → {new_fw}x{new_fh}  "
          f"scale={scale:.3f}", file=sys.stderr)

    # Scale picker points into the downscaled spatial coordinate space
    my_x = my_x_orig * scale
    my_y = my_y_orig * scale
    op_x = op_x_orig * scale
    op_y = op_y_orig * scale

    # ── YOLO-hybrid helpers ────────────────────────────────────────────────────

    def _parse_box(s):
        """Parse 'x1,y1,x2,y2' (in compressed-video coords) → scaled np array."""
        if not s:
            return None
        x1, y1, x2, y2 = map(float, s.split(","))
        return np.array([x1 * scale, y1 * scale, x2 * scale, y2 * scale],
                        dtype=np.float32)

    def _person_points(cx, cy, frame_w, frame_h):
        """3-point vertical stack: head / chest / abdomen — fallback when no box."""
        pts = [
            [cx, max(4.0, cy - 60)],
            [cx, cy],
            [cx, min(frame_h - 4.0, cy + 45)],
        ]
        return (np.array(pts, dtype=np.float32),
                np.array([1, 1, 1], dtype=np.int32))

    def _iou(a, b):
        """Intersection-over-union of two [x1,y1,x2,y2] boxes."""
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)

    def _match_yolo(dets_raw, last_my_bbox, last_op_bbox, min_iou=0.20):
        """
        Given raw YOLO dets [[x1,y1,x2,y2,conf]…] in compressed-video coords,
        scale them and match to the last known SAM2 bounding boxes by IoU.
        Returns (my_box_np, op_box_np) or (None, None) on failure.
        """
        if len(dets_raw) < 2 or last_my_bbox is None or last_op_bbox is None:
            return None, None
        # Keep only confident detections and scale to SAM2 space
        scaled = []
        for b in dets_raw:
            if b[4] >= 0.45:
                scaled.append([b[0]*scale, b[1]*scale, b[2]*scale, b[3]*scale])
        if len(scaled) < 2:
            return None, None
        # Hungarian-like: find best match for ME then best remaining for OP
        my_scores = [(_iou(b, last_my_bbox), i) for i, b in enumerate(scaled)]
        op_scores = [(_iou(b, last_op_bbox), i) for i, b in enumerate(scaled)]
        my_scores.sort(reverse=True)
        op_scores.sort(reverse=True)
        best_my_iou, best_my_i = my_scores[0]
        # Best OP = highest IoU that isn't also the best ME match
        best_op_iou, best_op_i = next(
            ((iou, i) for iou, i in op_scores if i != best_my_i),
            (0.0, -1)
        )
        if best_my_iou < min_iou or best_op_iou < min_iou or best_op_i < 0:
            return None, None
        return (np.array(scaled[best_my_i], dtype=np.float32),
                np.array(scaled[best_op_i], dtype=np.float32))

    # Load YOLO keyframes JSON (SAM2 frame index → list of detections)
    yolo_kf: dict = {}
    if args.yolo_kf and Path(args.yolo_kf).exists():
        try:
            raw_kf  = json.loads(Path(args.yolo_kf).read_text())
            yolo_kf = {int(k): v for k, v in raw_kf.items()}
            print(f"[sam2_visualizer] loaded YOLO keyframes for {len(yolo_kf)} frames",
                  file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[sam2_visualizer] WARNING: could not load YOLO keyframes: {e}",
                  file=sys.stderr, flush=True)

    # Parse YOLO box prompts for frame-0 seeding
    my_box0 = _parse_box(args.my_box)
    op_box0 = _parse_box(args.op_box)
    if my_box0 is not None:
        print(f"[sam2_visualizer] YOLO box prompts available for initial seeding",
              file=sys.stderr, flush=True)
    else:
        print(f"[sam2_visualizer] No YOLO box prompts — using 3-point stacks",
              file=sys.stderr, flush=True)

    # ── Temporal stride — target TARGET_TRACK_FPS output, cap at MAX_DURATION_S ──
    # We subsample every STRIDE-th frame so that the output plays at
    # (fps_orig / stride), keeping real-time speed correct.
    #
    # M2 fix: if Flask pre-computed the stride and passed it via --stride, use
    # that value directly.  This avoids divergence where Flask uses
    # meta["total_frames"] and sam2_visualizer uses cap.get(CAP_PROP_FRAME_COUNT)
    # — some containers return slightly different values from these two calls,
    # causing the yolo_kf index map to be off by 1 and re-anchors to miss.
    total_est = max(total_raw, 1)
    if args.stride is not None:
        stride = max(1, args.stride)
        print(f"[sam2_visualizer] using Flask-supplied stride={stride}",
              file=sys.stderr, flush=True)
    else:
        stride = max(1, round(fps_orig / TARGET_TRACK_FPS))
    fps_out = fps_orig / stride   # output fps — keeps playback speed correct

    # Cap extraction at MAX_DURATION_S from the seed frame
    seed_raw_fi    = (args.seed_fi or 0) * stride
    max_raw_fi     = seed_raw_fi + int(MAX_DURATION_S * fps_orig)
    capped_total   = min(total_est, max_raw_fi)
    sam2_frames_est = math.ceil(capped_total / stride)

    print(f"[sam2_visualizer] temporal: total={total_est}  stride={stride}  "
          f"fps_orig={fps_orig:.2f} → fps_out={fps_out:.2f}  "
          f"seed_raw={seed_raw_fi}  cap_raw={max_raw_fi}  "
          f"sam2_frames≈{sam2_frames_est}", file=sys.stderr)

    label_scale = max(0.5, new_fh / 800.0)
    box_thick   = max(2, int(new_fh / 300))

    # ── Extract downscaled, strided frames to temp dir ────────────────────────
    tmp_dir    = tempfile.mkdtemp(prefix="sam2vis_")
    frames_dir = Path(tmp_dir)
    frames     = []   # keep downscaled BGR arrays for compositing

    cap   = cv2.VideoCapture(args.video)
    fi    = 0    # raw frame counter
    si    = 0    # sampled frame counter (= index SAM2 sees)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fi >= max_raw_fi:
            # 5-minute cap from seed frame reached — stop loading frames
            break
        if fi % stride == 0:
            if scale < 1.0:
                frame = cv2.resize(frame, (new_fw, new_fh), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(frames_dir / f"{si:05d}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            frames.append(frame)
            si += 1
        fi += 1
    cap.release()
    total = si   # number of frames SAM2 will see
    print(f"[sam2_visualizer] extracted {total} sampled frames "
          f"(raw={fi}, cap_raw={max_raw_fi})  at {new_fw}x{new_fh}", file=sys.stderr)

    if total == 0:
        raise RuntimeError("No frames extracted — video may be unreadable.")

    # ── SAM2 propagation + composite ──────────────────────────────────────────
    t0         = time.time()
    my_tracked = op_tracked = 0

    # Output video at sampled fps × spatial size
    writer = cv2.VideoWriter(
        args.out,
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1.0, fps_out),
        (new_fw, new_fh),
    )

    # bfloat16 is only reliable on CUDA; MPS supports float16 but not bfloat16;
    # CPU is fine without autocast (it just runs in float32).
    def _autocast():
        if device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return torch.inference_mode()   # no-op context for MPS / CPU

    # Re-anchor SAM2 every this many sampled frames using YOLO detections.
    # Smaller = more frequent corrections (better accuracy, tiny extra overhead).
    REINIT_EVERY = 12

    track_frames: list = []   # populated when --track_json is supplied

    try:
        with torch.inference_mode(), _autocast():
            state = predictor.init_state(video_path=str(frames_dir))
            predictor.reset_state(state)

            # ── Find (or accept) the seed frame ───────────────────────────────
            # When the Flask UI ran the YOLO scan and the user confirmed a specific
            # frame (--seed_fi), we skip the scan and use that frame directly.
            # Otherwise we scan yolo_kf forward for the first frame where YOLO
            # clearly sees BOTH boxers (same heuristics as Flask-side _find_seed_frame):
            #   • Prefer frames with EXACTLY 2 inner-zone detections (unambiguous).
            #   • Fall back to 3+ detections, taking the 2 largest by area.
            #   • Edge filter: skip detections whose horizontal centre is within 8%
            #     of the frame edges (corner workers / ringside spectators).
            # Pre-seed frames are written as raw video so the full clip is preserved.

            SEED_CONF      = 0.50
            SEED_EDGE_FRAC = 0.08

            def _edge_ok(b):
                cx = (b[0] + b[2]) / 2
                return new_fw * SEED_EDGE_FRAC < cx < new_fw * (1 - SEED_EDGE_FRAC)

            def _box_area(b):
                return (b[2] - b[0]) * (b[3] - b[1])

            seed_fi   = None
            seed_dets = []

            if args.seed_fi is not None:
                # User-confirmed seed frame — use it directly (skip scan)
                seed_fi   = args.seed_fi
                raw_dets  = yolo_kf.get(seed_fi, [])
                inner     = [b for b in raw_dets
                             if len(b) >= 5 and b[4] >= SEED_CONF and _edge_ok(b)]
                if len(inner) >= 2:
                    seed_dets = sorted(inner, key=_box_area, reverse=True)[:2]
                elif len(raw_dets) >= 2:
                    seed_dets = sorted(raw_dets, key=lambda b: -b[4])[:2]
                print(f"[sam2_visualizer] using confirmed seed fi={seed_fi} "
                      f"({len(seed_dets)} dets available)",
                      file=sys.stderr, flush=True)
            else:
                # Automatic scan — same two-pass heuristic as Flask-side
                # Pass 1: prefer exactly 2 inner detections (unambiguous)
                for fi_cand in sorted(yolo_kf.keys()):
                    inner = [b for b in yolo_kf[fi_cand]
                             if len(b) >= 5 and b[4] >= SEED_CONF and _edge_ok(b)]
                    if len(inner) == 2:
                        seed_fi   = fi_cand
                        seed_dets = inner
                        break

                # Pass 2: accept 3+, take 2 largest by area
                if seed_fi is None:
                    for fi_cand in sorted(yolo_kf.keys()):
                        inner = [b for b in yolo_kf[fi_cand]
                                 if len(b) >= 5 and b[4] >= SEED_CONF and _edge_ok(b)]
                        if len(inner) >= 2:
                            seed_fi   = fi_cand
                            seed_dets = sorted(inner, key=_box_area, reverse=True)[:2]
                            break

            if seed_fi is None:
                # No frame in the whole video had 2 confident boxers.
                # Fall back to frame 0 with click-based 3-point stacks.
                seed_fi = 0
                print(f"[sam2_visualizer] WARNING: no frame with 2 people found. "
                      f"Falling back to click-based seeding.",
                      file=sys.stderr, flush=True)

            print(f"[sam2_visualizer] seed frame: SAM2 fi={seed_fi}  "
                  f"(raw frame {seed_fi * stride} / {fi})",
                  file=sys.stderr, flush=True)

            # ── Match user clicks to the two seed detections ───────────────────
            # Both the clicks (my_x_orig, my_y_orig) and yolo_kf boxes are in the
            # compressed-video coordinate space — same space, no heuristics needed.
            # Click-to-centre Euclidean distance directly identifies which detected
            # box is ME and which is OP.
            def _scale_det(b):
                return np.array([b[0]*scale, b[1]*scale, b[2]*scale, b[3]*scale],
                                dtype=np.float32)

            def _centre(b):
                return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)

            def _dist2(cx, cy, b):
                bx, by = _centre(b)
                return (cx - bx) ** 2 + (cy - by) ** 2

            if len(seed_dets) >= 2:
                # Assign ME = box closest to user's ME click; OP = the other.
                d0 = _dist2(my_x_orig, my_y_orig, seed_dets[0])
                d1 = _dist2(my_x_orig, my_y_orig, seed_dets[1])
                if d0 <= d1:
                    seed_my_box = _scale_det(seed_dets[0])
                    seed_op_box = _scale_det(seed_dets[1])
                else:
                    seed_my_box = _scale_det(seed_dets[1])
                    seed_op_box = _scale_det(seed_dets[0])

                predictor.add_new_points_or_box(
                    inference_state=state, frame_idx=seed_fi, obj_id=1, box=seed_my_box)
                predictor.add_new_points_or_box(
                    inference_state=state, frame_idx=seed_fi, obj_id=2, box=seed_op_box)
                print(f"[sam2_visualizer] ME box: {seed_my_box.tolist()}",
                      file=sys.stderr, flush=True)
                print(f"[sam2_visualizer] OP box: {seed_op_box.tolist()}",
                      file=sys.stderr, flush=True)
                last_my_bbox = [int(v) for v in seed_my_box]
                last_op_bbox = [int(v) for v in seed_op_box]
            else:
                # Absolute fallback: 3-point click stacks at frame 0
                my_pts, my_lbs = _person_points(my_x, my_y, new_fw, new_fh)
                op_pts, op_lbs = _person_points(op_x, op_y, new_fw, new_fh)
                predictor.add_new_points_or_box(
                    inference_state=state, frame_idx=0, obj_id=1,
                    points=my_pts, labels=my_lbs)
                predictor.add_new_points_or_box(
                    inference_state=state, frame_idx=0, obj_id=2,
                    points=op_pts, labels=op_lbs)
                seed_fi      = 0
                last_my_bbox = [int(v) for v in my_box0] if my_box0 is not None else None
                last_op_bbox = [int(v) for v in op_box0] if op_box0 is not None else None
                print(f"[sam2_visualizer] fallback: 3-point click stacks at frame 0",
                      file=sys.stderr, flush=True)

            # Write pre-seed frames as raw video (no tracking overlay).
            # The viewer sees the full clip duration; tracking boxes only appear
            # once SAM2 has a reliable initial seed.
            for pre_fi in range(seed_fi):
                writer.write(frames[pre_fi])
                if args.track_json:
                    track_frames.append({
                        "sam2_fi": pre_fi,
                        "raw_fi":  pre_fi * stride,
                        "time_s":  round(pre_fi / fps_out, 4),
                        "my_bbox": None,
                        "op_bbox": None,
                    })

            # Write initial progress so the UI immediately shows device info
            if args.progress:
                try:
                    Path(args.progress).write_text(json.dumps({
                        "pct": 0, "frame": seed_fi, "total": total,
                        "fps": 0, "device": str(device),
                    }))
                except Exception:
                    pass

            report_every    = max(1, total // 10)
            reinit_count    = 0
            last_written_fi = seed_fi - 1   # pre-seed frames already written above

            # ── Chunked propagation with periodic YOLO re-anchoring ───────────
            chunk_start = seed_fi   # start from first good frame, not frame 0
            while chunk_start < total:
                chunk_end = min(chunk_start + REINIT_EVERY, total)

                for out_fi, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                    state,
                    start_frame_idx=chunk_start,
                    max_frame_num_to_track=chunk_end - chunk_start,
                ):
                    # SAM2 includes the boundary frame in both the outgoing chunk
                    # and the next chunk.  Skip frames we already wrote.
                    if out_fi <= last_written_fi:
                        # Still update bboxes so re-anchor matching stays current.
                        for oi, obj_id in enumerate(out_obj_ids):
                            rm = out_mask_logits[oi]
                            if rm.ndim == 3: rm = rm.squeeze(0)
                            mask = (rm > 0.0).cpu().numpy().astype(bool)
                            bbox = None
                            if mask.any():
                                ys, xs = np.where(mask)
                                bbox = [int(xs.min()), int(ys.min()),
                                        int(xs.max()), int(ys.max())]
                            if obj_id == 1 and bbox: last_my_bbox = bbox
                            if obj_id == 2 and bbox: last_op_bbox = bbox
                        continue
                    if out_fi >= total:
                        break

                    orig    = frames[out_fi].copy()
                    overlay = orig.copy()
                    my_bbox = op_bbox = None

                    for oi, obj_id in enumerate(out_obj_ids):
                        raw_mask = out_mask_logits[oi]
                        if raw_mask.ndim == 3:
                            raw_mask = raw_mask.squeeze(0)
                        mask = (raw_mask > 0.0).cpu().numpy().astype(bool)

                        if mask.any():
                            ys, xs = np.where(mask)
                            bbox   = [int(xs.min()), int(ys.min()),
                                      int(xs.max()), int(ys.max())]
                        else:
                            bbox = None

                        if obj_id == 1:
                            if mask.any(): overlay[mask] = COL_MY
                            my_bbox = bbox
                            if bbox:
                                my_tracked  += 1
                                last_my_bbox = bbox
                        else:
                            if mask.any(): overlay[mask] = COL_OP
                            op_bbox = bbox
                            if bbox:
                                op_tracked  += 1
                                last_op_bbox = bbox

                    # Blend + draw
                    out_frame = cv2.addWeighted(orig, 0.60, overlay, 0.40, 0)
                    if my_bbox:
                        x1, y1, x2, y2 = my_bbox
                        cv2.rectangle(out_frame, (x1, y1), (x2, y2), COL_MY, box_thick)
                        _label(out_frame, "ME", x1, y1, COL_MY, label_scale)
                    if op_bbox:
                        x1, y1, x2, y2 = op_bbox
                        cv2.rectangle(out_frame, (x1, y1), (x2, y2), COL_OP, box_thick)
                        _label(out_frame, "OPPONENT", x1, y1, COL_OP, label_scale)

                    real_fi = out_fi * stride
                    cv2.putText(out_frame,
                                f"frame {real_fi+1}/{fi}  |  "
                                f"ME {'OK' if my_bbox else '--'}  "
                                f"OPP {'OK' if op_bbox else '--'}",
                                (8, new_fh - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                max(0.35, new_fh / 2200.0),
                                (180, 180, 180), 1, cv2.LINE_AA)
                    writer.write(out_frame)
                    if args.track_json:
                        track_frames.append({
                            "sam2_fi": out_fi,
                            "raw_fi":  out_fi * stride,
                            "time_s":  round(out_fi / fps_out, 4),
                            "my_bbox": my_bbox,
                            "op_bbox": op_bbox,
                        })
                    last_written_fi = out_fi

                    if out_fi % report_every == 0 or out_fi == total - 1:
                        pct = int(out_fi / max(total - 1, 1) * 100)
                        elapsed_so_far = time.time() - t0
                        fps_so_far = (out_fi + 1) / elapsed_so_far if elapsed_so_far > 0 else 0
                        print(f"[sam2_visualizer] {pct}%  frame {out_fi+1}/{total}  "
                              f"{fps_so_far:.2f} fps  "
                              f"ME={'OK' if my_bbox else '--'}  "
                              f"OPP={'OK' if op_bbox else '--'}  "
                              f"reinits={reinit_count}",
                              file=sys.stderr, flush=True)
                        if args.progress:
                            try:
                                Path(args.progress).write_text(json.dumps({
                                    "pct":    pct,
                                    "frame":  out_fi + 1,
                                    "total":  total,
                                    "fps":    round(fps_so_far, 2),
                                    "device": str(device),
                                }))
                            except Exception:
                                pass

                chunk_start = chunk_end

                # ── State reset + re-seed at every chunk boundary ─────────────
                # F4 fix: ALWAYS reset_state here, regardless of whether YOLO
                # fires.  Without this, the object-pointer memory bank accumulates
                # across ALL chunks and per-frame time grows unboundedly
                # (observed: 2.5s/frame → 6.3s/frame by chunk 6).
                # reset_state clears tracking state but keeps the frame-feature
                # cache, so no re-encoding is needed.
                #
                # Re-seeding priority at each boundary:
                #   A) YOLO box from keyframe → most accurate
                #   B) Last known SAM2 bbox converted to box prompt → keeps tracking
                #      alive through clinches where YOLO only sees 1 person
                #   C) Nothing if both last bboxes are None (should not happen post F1 fix)
                if chunk_start < total:
                    yolo_dets   = yolo_kf.get(chunk_start, []) if yolo_kf else []
                    new_my_box, new_op_box = _match_yolo(yolo_dets, last_my_bbox, last_op_bbox)

                    # Always reset the memory bank to keep per-frame time stable
                    predictor.reset_state(state)

                    if new_my_box is not None:
                        # Case A: YOLO gave us fresh boxes
                        predictor.add_new_points_or_box(
                            inference_state=state,
                            frame_idx=chunk_start, obj_id=1, box=new_my_box)
                        predictor.add_new_points_or_box(
                            inference_state=state,
                            frame_idx=chunk_start, obj_id=2, box=new_op_box)
                        reinit_count += 1
                        print(f"[sam2_visualizer] YOLO re-anchor #{reinit_count} "
                              f"at SAM2 frame {chunk_start}  "
                              f"ME={new_my_box.tolist()}  OP={new_op_box.tolist()}",
                              file=sys.stderr, flush=True)
                    else:
                        # Case B: YOLO unavailable — re-seed from last known positions.
                        # This keeps the tracker alive through frames where YOLO can't
                        # find two separate boxers (clinch, one off-screen, blur).
                        if last_my_bbox is not None:
                            fb_my = np.array(last_my_bbox, dtype=np.float32)
                            predictor.add_new_points_or_box(
                                inference_state=state,
                                frame_idx=chunk_start, obj_id=1, box=fb_my)
                        if last_op_bbox is not None:
                            fb_op = np.array(last_op_bbox, dtype=np.float32)
                            predictor.add_new_points_or_box(
                                inference_state=state,
                                frame_idx=chunk_start, obj_id=2, box=fb_op)
                        print(f"[sam2_visualizer] fallback re-seed at SAM2 frame "
                              f"{chunk_start} (YOLO unavailable — using last known bbox)",
                              file=sys.stderr, flush=True)

    finally:
        writer.release()
        if args.track_json and track_frames:
            try:
                Path(args.track_json).write_text(json.dumps(track_frames))
                print(f"[sam2_visualizer] track_json → {args.track_json} "
                      f"({len(track_frames)} frames)", file=sys.stderr)
            except Exception as _tj_err:
                print(f"[sam2_visualizer] track_json write failed: {_tj_err}",
                      file=sys.stderr)
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    elapsed       = time.time() - t0
    fps_effective = total / elapsed if elapsed > 0 else 0
    print(f"[sam2_visualizer] done — {total} sampled frames ({fi} raw) "
          f"in {elapsed:.1f}s  ({fps_effective:.1f} fps effective)  "
          f"ME={my_tracked}  OPP={op_tracked}", file=sys.stderr)

    result = {
        "ok":            True,
        "model":         f"sam2.1_hiera_{args.model}",
        "device":        str(device),
        "frame_count":   total,
        "fps_effective": round(fps_effective, 2),
        "scale":         round(scale, 4),
        "stride":        stride,
        "my_tracked":    my_tracked,
        "op_tracked":    op_tracked,
        "yolo_reinits":  reinit_count,
        "box_seeded":    my_box0 is not None,
    }
    Path(args.status).write_text(json.dumps(result))


if __name__ == "__main__":
    main()
