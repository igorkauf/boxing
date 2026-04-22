"""
pipeline_cache.py — content-hash-keyed cache for expensive pipeline stages.

Layout:
    cache/videos/<sha256>/
        original<ext>               # ingested upload
        meta.json                   # width, height, fps, duration, etc.
        stages/
            compressed.v1.mp4       # downscaled video fed to SAM2
            scene_ref.v1.json       # tier + detected floor/ring geometry
            sam2.v1.json            # box tracks
            sam2.v1.masks.npz       # packed fighter masks (for stabilization)
            stabilization.v1.json   # per-frame background-motion warps
            pose.v1.json            # YOLO-pose (post-smoothing)
            reid.v1.json            # appearance-embedding distances
            metrics.v1.json         # final metric series
        diagnostics/
            overlay_<stage>.mp4     # per-stage diagnostic renders

Stages are versioned; bumping a stage's version in STAGE_VERSIONS invalidates
that stage plus everything downstream of it (via DEPENDENTS). Stages stay
cached across sessions — if the same video content is uploaded twice, all
completed stages are reused.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE  = Path(__file__).parent
CACHE = BASE / "cache" / "videos"
CACHE.mkdir(parents=True, exist_ok=True)


# ── Stage versions ────────────────────────────────────────────────────────────
# Bump a version to force that stage (and everything downstream) to recompute
# on next run. Upstream stages stay cached.
STAGE_VERSIONS = {
    "compressed":    1,
    "scene_ref":     1,
    "sam2":          1,
    "stabilization": 1,
    "pose":          1,
    "reid":          1,
    "metrics":       1,
}

# Downstream invalidation: if KEY recomputes, every stage in its value set must
# also be recomputed (its artifacts will be deleted on invalidate).
DEPENDENTS = {
    "compressed":    {"scene_ref", "sam2", "stabilization", "pose", "reid", "metrics"},
    "sam2":          {"stabilization", "pose", "reid", "metrics"},
    "stabilization": {"pose", "metrics"},
    "pose":          {"reid", "metrics"},
    "scene_ref":     {"metrics"},
    "reid":          {"metrics"},
}


# ── Hashing ───────────────────────────────────────────────────────────────────
def hash_file(path: Path | str, chunk: int = 1 << 20) -> str:
    """SHA256 of a file's bytes, hex digest."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


# ── Directory helpers ─────────────────────────────────────────────────────────
def cache_dir(video_hash: str) -> Path:
    d = CACHE / video_hash
    (d / "stages").mkdir(parents=True, exist_ok=True)
    (d / "diagnostics").mkdir(parents=True, exist_ok=True)
    return d


def video_meta_path(video_hash: str) -> Path:
    return cache_dir(video_hash) / "meta.json"


def read_video_meta(video_hash: str) -> Optional[dict]:
    p = video_meta_path(video_hash)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def write_video_meta(video_hash: str, meta: dict) -> None:
    video_meta_path(video_hash).write_text(json.dumps(meta, indent=2))


def original_path(video_hash: str) -> Optional[Path]:
    """Locate the cached original video file (unknown extension)."""
    d = cache_dir(video_hash)
    for p in d.iterdir():
        if p.is_file() and p.stem == "original":
            return p
    return None


# ── Stage artifacts ───────────────────────────────────────────────────────────
def stage_path(video_hash: str, stage: str, ext: str = "json") -> Path:
    v = STAGE_VERSIONS[stage]
    return cache_dir(video_hash) / "stages" / f"{stage}.v{v}.{ext}"


def has_stage(video_hash: str, stage: str, ext: str = "json") -> bool:
    return stage_path(video_hash, stage, ext).exists()


def invalidate(video_hash: str, stage: str) -> None:
    """Delete a stage's artifacts and all downstream dependents."""
    to_delete = {stage} | DEPENDENTS.get(stage, set())
    stages_dir = cache_dir(video_hash) / "stages"
    if not stages_dir.exists():
        return
    for s in to_delete:
        for p in stages_dir.glob(f"{s}.v*.*"):
            try:
                p.unlink()
            except OSError:
                pass


# ── Ingest an uploaded file ───────────────────────────────────────────────────
def ingest_upload(src_path: Path | str) -> str:
    """
    Move (or copy if already elsewhere) an uploaded video into the cache keyed
    by content hash. Safe to call with a file already inside the cache.
    Returns the hex content hash.
    """
    src = Path(src_path)
    h = hash_file(src)
    dest_dir = cache_dir(h)
    dest = dest_dir / f"original{src.suffix.lower()}"

    if dest.exists():
        # Already cached — safe to drop the source copy if caller wants.
        return h

    # Prefer a rename (fast, same filesystem) and fall back to copy.
    try:
        os.replace(src, dest)
    except OSError:
        shutil.copy2(src, dest)
    return h


# ── Symlink cached artifacts into a session directory ─────────────────────────
def link_into_session(video_hash: str, session_dir: Path) -> None:
    """
    Create symlinks from the session directory to the cached original +
    compressed video so existing code paths (which read from session_dir)
    keep working. If the compressed stage isn't cached yet, only the original
    is linked — the compressed symlink is created later via refresh_session_links().
    """
    session_dir.mkdir(parents=True, exist_ok=True)
    orig = original_path(video_hash)
    if orig is not None:
        _link(orig, session_dir / f"original{orig.suffix.lower()}")
    comp = stage_path(video_hash, "compressed", "mp4")
    if comp.exists():
        _link(comp, session_dir / "lab_compressed.mp4")


def refresh_session_links(video_hash: str, session_dir: Path) -> None:
    """Re-create any stage symlinks whose versions may have changed."""
    comp = stage_path(video_hash, "compressed", "mp4")
    link_target = session_dir / "lab_compressed.mp4"
    if comp.exists():
        _link(comp, link_target)


def _link(src: Path, dst: Path) -> None:
    """Create or replace a symlink dst → src. Handles stale symlinks."""
    try:
        if dst.is_symlink() or dst.exists():
            dst.unlink()
    except OSError:
        pass
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        # Symlinks not supported on this filesystem — fall back to copy.
        shutil.copy2(src, dst)
