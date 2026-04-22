#!/usr/bin/env python3
"""
test_harness.py — regression-testing harness for the arena-metrics pipeline.

Runs `metrics.compute()` against every processed Lab session on disk
(sessions_data/) and writes a CSV + a one-line summary per clip. Reuses
the cached `sam2_enriched.json` + `arena.json` produced by the Flask Lab
pipeline, so a single invocation takes seconds regardless of clip length
— perfect for A/B'ing a threshold change by running once before and once
after and diffing the CSVs.

Usage:
    ./venv/bin/python test_harness.py
    ./venv/bin/python test_harness.py --out snapshots/v0.2.csv
    ./venv/bin/python test_harness.py --diff snapshots/v0.1.csv

If you pass --diff, the harness runs a fresh pass AND diffs it against
the named baseline CSV, printing per-session deltas for each metric so
you can instantly see whether a change helped or hurt.

To generate sessions in the first place, use the Flask Lab UI. Adding
fully-automated SAM2 runs to the harness is possible later but not
necessary for metric-layer iteration, which is what this is for.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path


BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

import metrics  # noqa: E402


METRIC_FIELDS = [
    "my_ring", "op_ring",
    "my_aggression", "op_aggression",
    "my_movement", "op_movement",
    "my_volume", "op_volume",
    "my_defense", "op_defense",
    "my_guard", "op_guard",
    "my_punches", "op_punches",
]

CSV_FIELDS = [
    "session", "filename", "duration_s",
    "tier", "tier_weight",
    "my_visibility", "op_visibility",
    *METRIC_FIELDS,
    "compute_ms",
]


def scan_sessions(sessions_dir: Path):
    """Yield (session_dir, meta_dict) for every lab-mode session with
    a present sam2_enriched.json (required by metrics.compute)."""
    if not sessions_dir.exists():
        return
    for d in sorted(sessions_dir.iterdir()):
        if not d.is_dir():
            continue
        meta_path = d / "meta.json"
        enriched  = d / "sam2_enriched.json"
        if not (meta_path.exists() and enriched.exists()):
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            continue
        if meta.get("lab_mode"):
            yield d, meta


def run(sessions_dir: Path):
    """Compute metrics for every session. Returns list of CSV rows."""
    rows = []
    for sess_dir, meta in scan_sessions(sessions_dir):
        t0 = time.perf_counter()
        result = metrics.compute(sess_dir)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if not result.get("ok"):
            print(f"[skip] {sess_dir.name:12}  {result.get('error', 'unknown error')}")
            continue

        rows.append({
            "session":       sess_dir.name,
            "filename":      meta.get("filename", ""),
            "duration_s":    result["duration_s"],
            "tier":          result["tier"],
            "tier_weight":   result["tier_weight"],
            "my_visibility": round(result["my_visibility"], 3),
            "op_visibility": round(result["op_visibility"], 3),
            **{k: result[k] for k in METRIC_FIELDS},
            "compute_ms":    round(elapsed_ms, 1),
        })

        print(f"[ok]   {sess_dir.name:12}  "
              f"punches {result['my_punches']:3}/{result['op_punches']:3}  "
              f"ring {result['my_ring']:3}/{result['op_ring']:3}  "
              f"aggr {result['my_aggression']:3}/{result['op_aggression']:3}  "
              f"mov {result['my_movement']:3}/{result['op_movement']:3}  "
              f"vol {result['my_volume']:3}/{result['op_volume']:3}  "
              f"tier {result['tier']:6}  "
              f"{elapsed_ms:6.0f}ms  "
              f"{meta.get('filename', '')}")
    return rows


def write_csv(rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path):
    with open(path) as f:
        return {row["session"]: row for row in csv.DictReader(f)}


def diff_against(cur_rows, baseline_path: Path):
    """Print a per-session, per-metric delta table vs. an earlier snapshot."""
    baseline = read_csv(baseline_path)
    cur_by_sess = {r["session"]: r for r in cur_rows}

    all_sessions = sorted(set(baseline) | set(cur_by_sess))
    print()
    print(f"{'session':<12}  {'metric':<14}  {'baseline':>8}  {'current':>8}  {'delta':>8}")
    print("-" * 60)
    any_change = False
    for s in all_sessions:
        b = baseline.get(s)
        c = cur_by_sess.get(s)
        if b is None:
            print(f"{s:<12}  (new in current — not in baseline)")
            any_change = True
            continue
        if c is None:
            print(f"{s:<12}  (removed from current — was in baseline)")
            any_change = True
            continue
        for k in METRIC_FIELDS:
            try:
                bv = float(b[k]); cv = float(c[k])
            except (KeyError, ValueError, TypeError):
                continue
            d = cv - bv
            if abs(d) < 0.01:
                continue
            any_change = True
            arrow = "↑" if d > 0 else "↓"
            print(f"{s:<12}  {k:<14}  {bv:>8.1f}  {cv:>8.1f}  {arrow}{abs(d):>7.1f}")
    if not any_change:
        print("(no metric moved by more than 0.01 vs baseline)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out", default="snapshots/latest.csv",
                    help="CSV output path (default: snapshots/latest.csv)")
    ap.add_argument("--sessions-dir", default="sessions_data",
                    help="Directory containing processed sessions")
    ap.add_argument("--diff", default=None,
                    help="Baseline CSV to diff current run against")
    args = ap.parse_args()

    sessions_dir = BASE / args.sessions_dir
    rows = run(sessions_dir)
    if not rows:
        print("\nNo sessions to process — run the Lab pipeline first.")
        return 1

    out_path = BASE / args.out
    write_csv(rows, out_path)
    print(f"\n{len(rows)} sessions → {out_path}")

    if args.diff:
        diff_path = BASE / args.diff
        if not diff_path.exists():
            print(f"\n[warn] baseline {diff_path} not found; skipping diff")
        else:
            print(f"\nDiff vs {diff_path.name}:")
            diff_against(rows, diff_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
