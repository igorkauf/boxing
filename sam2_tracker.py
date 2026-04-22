"""
sam2_tracker.py  —  Called by the Flask app via subprocess.

Usage (from Flask):
    sam2_venv/bin/python sam2_tracker.py \
        --video   /path/to/original.mp4  \
        --my_pt   x,y                    \
        --op_pt   x,y                    \
        --model   tiny                   \   # tiny|small  (default: tiny)
        --max_dim 1024                   \   # downscale longest side to this (default 1024)
        --out     /path/to/output.json

Output JSON:
    {
      "ok": true,
      "model": "sam2.1_hiera_tiny",
      "frame_count": 2400,       # ORIGINAL (unstrided) frame count
      "fps_effective": 12.3,
      "scale": 0.391,
      "stride": 4,               # temporal stride used
      "boxes": {
        "0":  {"my": [x1,y1,x2,y2], "op": [x1,y1,x2,y2]},  # original pixel coords
        "1":  {"my": [x1,y1,x2,y2], "op": null},
        ...
        "2399": { ... }          # every original frame present (boxes repeated
      }                          # between sampled frames)
    }

On error:
    {"ok": false, "error": "...message...", "traceback": "..."}

Memory management
─────────────────
Two levels of downscaling to keep SAM2 within ~4 GB of CPU RAM:

  1. Spatial   — Frames are extracted at max max_dim on the longest side
                 (default 1024).  SAM2 tracks shapes not pixels so quality
                 is essentially unchanged.  Picker points and output boxes are
                 scaled accordingly.

  2. Temporal  — Only every STRIDE-th frame is sent to SAM2 so that at most
                 MAX_SAM2_FRAMES frames load simultaneously.  For each original
                 frame not sampled, the box from the preceding sampled frame is
                 repeated.  This is transparent to the Kalman loop which just
                 reads boxes[str(fi)] for every fi.
"""

import sys, os, json, time, math, tempfile, argparse, traceback
from pathlib import Path

import numpy as np
import cv2
import torch

# ── Memory budget ──────────────────────────────────────────────────────────────
# SAM2 Hiera-tiny stores ~13-15 MB of feature tensors per frame in CPU RAM.
#   300 frames → ~4 GB   (safe on 8 GB machines)
#   600 frames → ~8 GB   (tight on 8 GB machines)
#  1350 frames → ~17 GB  (the "Invalid buffer size" crash)
MAX_SAM2_FRAMES = 100


def mask_to_bbox(mask: np.ndarray):
    """Return [x1,y1,x2,y2] from a boolean mask, or None if empty."""
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",   required=True)
    ap.add_argument("--my_pt",   required=True, help="x,y of ME boxer centre (original coords)")
    ap.add_argument("--op_pt",   required=True, help="x,y of OP boxer centre (original coords)")
    ap.add_argument("--model",   default="tiny", choices=["tiny", "small"])
    ap.add_argument("--max_dim", default=1024,   type=int,
                    help="Downscale longest side to this before SAM2 (default 1024)")
    ap.add_argument("--out",     required=True,  help="output JSON path")
    args = ap.parse_args()

    try:
        _run(args)
    except Exception as e:
        err = {"ok": False, "error": str(e), "traceback": traceback.format_exc()}
        Path(args.out).write_text(json.dumps(err))
        print(f"[sam2_tracker] FAILED: {e}", file=sys.stderr)
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
        f"[sam2_tracker] device={device}  "
        f"torch={torch.__version__}  "
        f"mps_built={torch.backends.mps.is_built()}  "
        f"mps_available={torch.backends.mps.is_available()}  "
        f"cuda={torch.cuda.is_available()}  "
        f"MPS_FALLBACK={mps_fallback}",
        file=sys.stderr, flush=True,
    )
    n_threads = max(1, _os.cpu_count() or 4)
    torch.set_num_threads(n_threads)
    print(f"[sam2_tracker] CPU threads set to {n_threads}", file=sys.stderr, flush=True)

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
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}\nRun:  bash setup_sam2.sh")

    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(cfg, str(ckpt), device=device)
    if hasattr(predictor, "max_obj_ptrs_in_encoder"):
        predictor.max_obj_ptrs_in_encoder = 8   # 4 was too aggressive; 8 balances quality vs speed

    # ── Parse picker points (original video coordinates) ──────────────────────
    my_x_orig, my_y_orig = map(float, args.my_pt.split(","))
    op_x_orig, op_y_orig = map(float, args.op_pt.split(","))

    # ── Open video: measure dimensions + total frame count ────────────────────
    cap0      = cv2.VideoCapture(args.video)
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
    print(f"[sam2_tracker] spatial: {orig_fw}x{orig_fh} → {new_fw}x{new_fh}  "
          f"scale={scale:.3f}", file=sys.stderr)

    # Scale picker points into the downscaled spatial coordinate space
    my_x = my_x_orig * scale
    my_y = my_y_orig * scale
    op_x = op_x_orig * scale
    op_y = op_y_orig * scale

    # ── Multi-point seeding ────────────────────────────────────────────────────
    # 3-point vertical stack per boxer: head / chest (clicked) / abdomen.
    # This gives SAM2 an unambiguous person column rather than a single
    # ambiguous dot that can be confused with ring ropes or shadows.
    _H_OFF = 60   # px above click → head/neck
    _A_OFF = 45   # px below click → abdomen

    def _person_points(cx, cy, frame_w, frame_h):
        pts = [
            [cx,  max(4.0, cy - _H_OFF)],
            [cx,  cy],
            [cx,  min(frame_h - 4.0, cy + _A_OFF)],
        ]
        return (np.array(pts, dtype=np.float32),
                np.array([1, 1, 1], dtype=np.int32))

    # ── Temporal stride ────────────────────────────────────────────────────────
    total_est = max(total_raw, 1)
    stride    = max(1, math.ceil(total_est / MAX_SAM2_FRAMES))
    print(f"[sam2_tracker] temporal: total_est={total_est}  stride={stride}  "
          f"→ ~{math.ceil(total_est/stride)} SAM2 frames", file=sys.stderr)

    # ── Extract (spatially downscaled + temporally strided) frames ────────────
    tmp        = tempfile.mkdtemp(prefix="sam2_frames_")
    frames_dir = Path(tmp)
    cap        = cv2.VideoCapture(args.video)
    raw_fi     = 0    # raw frame index
    sam_fi     = 0    # SAM2 frame index (0-based in the temp dir)

    # Maps SAM2 frame index → original raw frame index (for box remapping)
    sam_to_raw = {}

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if raw_fi % stride == 0:
            if scale < 1.0:
                frame = cv2.resize(frame, (new_fw, new_fh), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(frames_dir / f"{sam_fi:05d}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            sam_to_raw[sam_fi] = raw_fi
            sam_fi += 1
        raw_fi += 1
    cap.release()

    total_raw_actual = raw_fi        # true frame count (may differ from container metadata)
    total_sam        = sam_fi        # number of frames SAM2 will see
    print(f"[sam2_tracker] extracted {total_sam} SAM2 frames "
          f"(raw={total_raw_actual}) at {new_fw}x{new_fh}", file=sys.stderr)

    if total_sam == 0:
        raise RuntimeError("No frames extracted — video may be unreadable.")

    inv_scale = 1.0 / scale if scale < 1.0 else 1.0

    try:
        # ── SAM2 propagation ──────────────────────────────────────────────────
        t0          = time.time()
        sam2_boxes  = {}   # sam_fi → {"my": bbox_orig_coords, "op": bbox_orig_coords}

        # bfloat16 is only reliable on CUDA; MPS supports float16 but not bfloat16;
        # CPU is fine without autocast (float32 by default).
        def _autocast():
            if device.type == "cuda":
                return torch.autocast("cuda", dtype=torch.bfloat16)
            return torch.inference_mode()   # no-op context for MPS / CPU

        with torch.inference_mode(), _autocast():
            state = predictor.init_state(video_path=str(frames_dir))
            predictor.reset_state(state)

            # Seed both trackers from SAM2 frame 0 using 3-point vertical stacks
            my_pts, my_lbs = _person_points(my_x, my_y, new_fw, new_fh)
            op_pts, op_lbs = _person_points(op_x, op_y, new_fw, new_fh)
            print(f"[sam2_tracker] ME seed points (scaled): {my_pts.tolist()}",
                  file=sys.stderr, flush=True)
            print(f"[sam2_tracker] OP seed points (scaled): {op_pts.tolist()}",
                  file=sys.stderr, flush=True)

            predictor.add_new_points_or_box(
                inference_state=state, frame_idx=0, obj_id=1,
                points=my_pts, labels=my_lbs,
            )
            predictor.add_new_points_or_box(
                inference_state=state, frame_idx=0, obj_id=2,
                points=op_pts, labels=op_lbs,
            )

            report_every = max(1, total_sam // 10)
            for out_fi, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
                entry = {"my": None, "op": None}
                for oi, obj_id in enumerate(out_obj_ids):
                    raw_mask = out_mask_logits[oi]
                    if raw_mask.ndim == 3:
                        raw_mask = raw_mask.squeeze(0)
                    mask         = (raw_mask > 0.0).cpu().numpy().astype(bool)
                    bbox_scaled  = mask_to_bbox(mask)

                    # Scale bbox back to original coordinate space
                    if bbox_scaled is not None and scale < 1.0:
                        bbox = [
                            int(bbox_scaled[0] * inv_scale),
                            int(bbox_scaled[1] * inv_scale),
                            int(bbox_scaled[2] * inv_scale),
                            int(bbox_scaled[3] * inv_scale),
                        ]
                    else:
                        bbox = bbox_scaled

                    if obj_id == 1:
                        entry["my"] = bbox
                    else:
                        entry["op"] = bbox

                sam2_boxes[out_fi] = entry

                if out_fi % report_every == 0 or out_fi == total_sam - 1:
                    pct = int(out_fi / max(total_sam - 1, 1) * 100)
                    elapsed_so_far = time.time() - t0
                    fps_so_far = (out_fi + 1) / elapsed_so_far if elapsed_so_far > 0 else 0
                    print(f"[sam2_tracker] {pct}%  sam_frame {out_fi+1}/{total_sam}  "
                          f"{fps_so_far:.2f} fps",
                          file=sys.stderr, flush=True)

        elapsed       = time.time() - t0
        fps_effective = total_sam / elapsed if elapsed > 0 else 0
        print(f"[sam2_tracker] propagation done: {total_sam} SAM2 frames  "
              f"(raw={total_raw_actual}) in {elapsed:.1f}s "
              f"({fps_effective:.1f} fps eff)", file=sys.stderr)

        # ── Expand SAM2 boxes back to every original frame ────────────────────
        # For each original frame, use the box from the nearest preceding
        # sampled frame (hold-last-value interpolation).  This is transparent
        # to the Kalman loop which reads boxes[str(fi)] for every raw fi.
        boxes = {}
        last_entry = {"my": None, "op": None}
        si = 0   # pointer into sam2_boxes keys (sorted)
        sorted_sam_fis = sorted(sam2_boxes.keys())

        for raw_fi_idx in range(total_raw_actual):
            # Advance sampled pointer if the next sampled frame has arrived
            while si < len(sorted_sam_fis) and sam_to_raw[sorted_sam_fis[si]] <= raw_fi_idx:
                last_entry = sam2_boxes[sorted_sam_fis[si]]
                si += 1
            boxes[str(raw_fi_idx)] = last_entry

        result = {
            "ok":            True,
            "model":         f"sam2.1_hiera_{args.model}",
            "device":        str(device),
            "frame_count":   total_raw_actual,
            "fps_effective": round(fps_effective, 2),
            "scale":         round(scale, 4),
            "stride":        stride,
            "boxes":         boxes,
        }

    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    Path(args.out).write_text(json.dumps(result))


if __name__ == "__main__":
    main()
