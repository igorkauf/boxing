"""
test_sam2.py  —  Smoke-test SAM2 tiny on your Mac before full integration.

Run with:  sam2_venv/bin/python test_sam2.py  [path/to/video.mp4]

What it does:
  1. Loads SAM2.1 tiny on MPS (Apple Silicon) or CPU
  2. Extracts the first 60 frames of the supplied video (or a synthetic one)
  3. Initialises two tracker objects from hardcoded centre points
     (simulating what the boxer picker will supply)
  4. Propagates through all 60 frames and reports per-frame bounding boxes
  5. Prints timing so you know if it's fast enough to be practical
"""

import sys, time, tempfile, os, json
from pathlib import Path

# ── 0. Check we're in the right venv ──────────────────────────────────────
try:
    from sam2.build_sam import build_sam2_video_predictor
except ImportError:
    print("✗  sam2 not found.  Run:  bash setup_sam2.sh  first.")
    sys.exit(1)

import torch
import numpy as np
import cv2

# ── 1. Device ──────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✓  Device: Apple Silicon MPS")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("✓  Device: CUDA GPU")
else:
    device = torch.device("cpu")
    print("ℹ  Device: CPU  (will be slower)")

# ── 2. Load model ──────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
CKPT_DIR   = SCRIPT_DIR / "sam2_checkpoints"
CKPT       = CKPT_DIR / "sam2.1_hiera_tiny.pt"

if not CKPT.exists():
    print(f"✗  Checkpoint not found: {CKPT}")
    print("   Run:  bash setup_sam2.sh")
    sys.exit(1)

print(f"→  Loading SAM2.1 tiny from {CKPT} ...")
t0 = time.time()
predictor = build_sam2_video_predictor(
    "configs/sam2.1/sam2.1_hiera_t.yaml",
    str(CKPT),
    device=device,
)
print(f"✓  Model loaded in {time.time()-t0:.1f}s")

# ── 3. Get video frames ────────────────────────────────────────────────────
video_path = sys.argv[1] if len(sys.argv) > 1 else None

with tempfile.TemporaryDirectory() as tmpdir:
    frames_dir = Path(tmpdir) / "frames"
    frames_dir.mkdir()

    if video_path:
        video_path = str(Path(video_path).resolve())
        if not Path(video_path).exists():
            print(f"✗  File not found: {video_path}")
            print("   Provide an absolute path, e.g.:")
            print("   sam2_venv/bin/python test_sam2.py /full/path/to/video.mp4")
            sys.exit(1)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"✗  OpenCV cannot open: {video_path}")
            sys.exit(1)
        fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"→  Video: {fw}×{fh}  —  extracting first 60 frames ...")
        fi  = 0
        while fi < 60:
            ok, frame = cap.read()
            if not ok: break
            cv2.imwrite(str(frames_dir / f"{fi:05d}.jpg"), frame)
            fi += 1
        cap.release()
        if fi == 0:
            print("✗  No frames extracted — is this a valid video file?")
            sys.exit(1)
        total = fi
        # Two centre points from the first frame — simulate picker selection
        # (left-third and right-third of frame, vertically centred)
        my_pt = [[fw * 0.30, fh * 0.50]]   # "ME"  boxer
        op_pt = [[fw * 0.70, fh * 0.50]]   # "OP"  boxer
    else:
        print("→  No video supplied — generating 60 synthetic frames (640×480) ...")
        fw, fh, total = 640, 480, 60
        for fi in range(total):
            img = np.zeros((fh, fw, 3), dtype=np.uint8)
            # Draw two moving rectangles to give SAM2 something to track
            x_my = int(80  + fi * 2) % fw
            x_op = int(460 - fi * 2) % fw
            cv2.rectangle(img, (x_my-30, 160), (x_my+30, 320), (0, 100, 200), -1)
            cv2.rectangle(img, (x_op-30, 160), (x_op+30, 320), (200, 80,   0), -1)
            cv2.imwrite(str(frames_dir / f"{fi:05d}.jpg"), img)
        my_pt = [[110.0, 240.0]]
        op_pt = [[430.0, 240.0]]

    # ── 4. Initialise and propagate ───────────────────────────────────────
    print(f"→  Initialising SAM2 with {total} frames ...")
    t1 = time.time()

    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
        state = predictor.init_state(video_path=str(frames_dir))
        predictor.reset_state(state)

        # Add ME boxer (obj_id=1)
        _, _, _ = predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,
            obj_id=1,
            points=np.array(my_pt, dtype=np.float32),
            labels=np.array([1], dtype=np.int32),   # 1 = foreground
        )
        # Add OP boxer (obj_id=2)
        _, _, _ = predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,
            obj_id=2,
            points=np.array(op_pt, dtype=np.float32),
            labels=np.array([1], dtype=np.int32),
        )

        print(f"→  Propagating through {total} frames ...")
        results = {}   # frame_idx → {obj_id: [x1,y1,x2,y2]}

        for out_fi, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
            results[out_fi] = {}
            for oi, obj_id in enumerate(out_obj_ids):
                mask = (out_mask_logits[oi] > 0.0).squeeze().cpu().numpy().astype(bool)
                if not mask.any():
                    results[out_fi][int(obj_id)] = None
                    continue
                ys, xs = np.where(mask)
                results[out_fi][int(obj_id)] = [
                    int(xs.min()), int(ys.min()),
                    int(xs.max()), int(ys.max())
                ]

    elapsed = time.time() - t1
    fps_eff  = total / elapsed

    # ── 5. Report ─────────────────────────────────────────────────────────
    print()
    print("─────────────────────────────────────────────────────")
    print(f"✓  Propagated {total} frames in {elapsed:.1f}s  ({fps_eff:.1f} fps effective)")
    print()

    tracked_my = sum(1 for f in results.values() if f.get(1) is not None)
    tracked_op = sum(1 for f in results.values() if f.get(2) is not None)
    print(f"  ME  tracked in {tracked_my}/{total} frames")
    print(f"  OP  tracked in {tracked_op}/{total} frames")
    print()

    # Show sample bboxes
    for fi in [0, total//4, total//2, total-1]:
        if fi in results:
            print(f"  Frame {fi:3d}:  ME={results[fi].get(1)}  OP={results[fi].get(2)}")

    print()
    if fps_eff >= 5:
        print("✓  Speed looks good for offline video analysis.")
    elif fps_eff >= 1:
        print("ℹ  Manageable speed — a 1-minute video will take ~60s extra.")
    else:
        print("⚠  Slow on this device.  Try the small model or CPU optimisations.")
    print("─────────────────────────────────────────────────────")
