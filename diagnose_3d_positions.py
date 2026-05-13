"""
diagnose_3d_positions.py
------------------------
Reads existing pipeline output files (no GPU / model loading required) and
produces:

  1. A per-object table:
       object_id | label | img_idx | bbox_center_px | midas_depth
       | manifest_depth | manifest_pos_3d | depth_flag

  2. A failure-mode summary that classifies the cluster observed in the 3-D
     spatial audio plot into one of three cases:
       A) Fallback bug       – position_3d was never written (depth == 0.0)
       B) Depth collapse     – depth == 5.0 constant (depth_maps=None fallback)
       C) Camera frame error – real depths but wrong world coords

  3. A MiDaS depth sanity check: samples each saved .npy at every object's
     bbox centre and reports per-image depth statistics.

  4. Two matplotlib 3-D scatter plots:
       • Current manifest positions  (showing the cluster)
       • Corrected positions          (MiDaS depth + DUSt3R w2c from manifest
                                       listener data, or identity if unavailable)

Usage
-----
  python diagnose_3d_positions.py [--csv out.csv] [--no-plot]

All paths are relative to the project root; run from there.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------------
SEG_PATH   = Path("data/segmentation/all_segmentations.json")
DEPTH_DIR  = Path("data/reconstruction/depth")
MANIFEST   = Path("data/output/audio_manifest.json")
DUST3R_CAM = Path("data/output/audio_manifest.json")   # listener_positions carry w2c info


# ---------------------------------------------------------------------------
# Helpers – mirror src/utils.py (no import needed if run standalone)
# ---------------------------------------------------------------------------
def compute_bounding_box_center(bbox: List[float]) -> Tuple[float, float]:
    """Return (cx_norm, cy_norm) of a normalised [x1,y1,x2,y2] bbox."""
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def unproject_2d_to_3d(
    px: float, py: float, depth: float,
    fx: float, fy: float, cx: float, cy: float,
    pose: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Pinhole unproject (u,v,depth) → 3-D world point."""
    x = (px - cx) * depth / fx
    y = (py - cy) * depth / fy
    z = depth
    p = np.array([x, y, z, 1.0])
    if pose is not None:
        p = pose @ p
    return p[:3]


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_segmentation(path: Path) -> List[dict]:
    if not path.exists():
        sys.exit(f"ERROR: segmentation file not found: {path}")
    with open(path) as f:
        return json.load(f)


def load_depth_maps(depth_dir: Path) -> Dict[int, np.ndarray]:
    maps: Dict[int, np.ndarray] = {}
    for p in sorted(depth_dir.glob("depth_*.npy")):
        try:
            idx = int(p.stem.split("_")[1])
            maps[idx] = np.load(p)
        except (ValueError, IndexError):
            pass
    return maps


def load_manifest(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"ERROR: manifest file not found: {path}")
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Build per-object rows
# ---------------------------------------------------------------------------
def build_rows(
    seg: List[dict],
    depth_maps: Dict[int, np.ndarray],
    pos_by_id: Dict[str, dict],
) -> List[dict]:
    rows = []
    obj_counter: Dict[int, int] = {}   # img_idx → running count within that image

    for img_result in seg:
        img_idx = img_result.get("image_index", 0)
        h, w = img_result.get("image_shape", [480, 640])[:2]
        dmap = depth_maps.get(img_idx)

        local_count = obj_counter.get(img_idx, 0)

        for obj in img_result.get("objects", []):
            obj_id = f"obj_{img_idx:03d}_{local_count:03d}"
            local_count += 1

            bbox = obj.get("bbox", [0, 0, 1, 1])
            cx_n, cy_n = compute_bounding_box_center(bbox)
            px = cx_n * w
            py = cy_n * h

            # Depth from saved MiDaS map
            if dmap is not None:
                dy = int(np.clip(py, 0, h - 1))
                dx = int(np.clip(px, 0, w - 1))
                midas_d: Optional[float] = float(dmap[dy, dx])
            else:
                midas_d = None

            # What ended up in the manifest
            entry = pos_by_id.get(obj_id, {})
            manifest_depth  = entry.get("depth")     # None if not in manifest
            manifest_pos    = entry.get("position_3d")

            # Classify depth flag
            if manifest_depth is None:
                depth_flag = "NOT_IN_MANIFEST"
            elif manifest_depth == 0.0 and (manifest_pos is None or manifest_pos == [0.0, 0.0, 5.0]):
                depth_flag = "A_FALLBACK_NO_3D"
            elif manifest_depth == 5.0:
                depth_flag = "B_DEPTH_MAPS_NONE"
            elif manifest_depth == 0.0:
                depth_flag = "A_FALLBACK_NO_3D"
            else:
                depth_flag = "C_REAL_DEPTH"

            rows.append({
                "id":             obj_id,
                "label":          obj.get("label", "?"),
                "img_idx":        img_idx,
                "bbox":           bbox,
                "bbox_center_px": (round(px, 1), round(py, 1)),
                "img_wh":         (w, h),
                "midas_depth":    midas_d,
                "manifest_depth": manifest_depth,
                "manifest_pos":   manifest_pos,
                "depth_flag":     depth_flag,
                "confidence":     obj.get("confidence", 0.0),
            })

        obj_counter[img_idx] = local_count

    return rows


# ---------------------------------------------------------------------------
# Print table
# ---------------------------------------------------------------------------
def print_table(rows: List[dict]) -> None:
    try:
        import pandas as pd  # noqa: PLC0415
        df = pd.DataFrame([{
            "id":            r["id"],
            "label":         r["label"][:22],
            "img":           r["img_idx"],
            "px_center":     r["bbox_center_px"],
            "midas_d":       f"{r['midas_depth']:.2f}" if r["midas_depth"] is not None else "N/A",
            "manif_d":       f"{r['manifest_depth']:.2f}" if r["manifest_depth"] is not None else "N/A",
            "pos_3d":        str([round(v, 2) for v in r["manifest_pos"]]) if r["manifest_pos"] else "N/A",
            "flag":          r["depth_flag"],
        } for r in rows])
        with pd.option_context("display.max_rows", None, "display.max_colwidth", 30,
                               "display.width", 160):
            print(df.to_string(index=False))
    except Exception:
        # Fallback: plain text (catches ImportError, ValueError ABI mismatch, etc.)
        hdr = f"{'id':20s} {'label':22s} {'img':4s} {'midas_d':8s} {'manif_d':8s} {'pos_3d':30s} {'flag'}"
        print(hdr)
        print("-" * len(hdr))
        for r in rows:
            pos_str = str([round(v, 2) for v in r["manifest_pos"]]) if r["manifest_pos"] else "N/A"
            md = f"{r['midas_depth']:.2f}" if r["midas_depth"] is not None else "N/A"
            print(f"{r['id']:20s} {r['label'][:22]:22s} {r['img_idx']:4d} "
                  f"{md:8s} "
                  f"{str(r['manifest_depth']):8s} "
                  f"{pos_str:30s} {r['depth_flag']}")


# ---------------------------------------------------------------------------
# Failure-mode summary
# ---------------------------------------------------------------------------
def print_summary(rows: List[dict]) -> str:
    total = len(rows)
    counts = {}
    for r in rows:
        counts[r["depth_flag"]] = counts.get(r["depth_flag"], 0) + 1

    print("\n" + "=" * 60)
    print("FAILURE MODE SUMMARY")
    print("=" * 60)
    print(f"Total objects: {total}")
    for flag, n in sorted(counts.items(), key=lambda x: -x[1]):
        pct = 100.0 * n / total if total else 0
        print(f"  {flag:30s}: {n:4d}  ({pct:.1f}%)")

    dominant = max(counts, key=counts.get) if counts else ""
    print()
    if "A_FALLBACK_NO_3D" in dominant:
        diagnosis = (
            "DIAGNOSIS: Fallback bug (Mode A)\n"
            "  project_objects_to_3d output was NOT stored in state['objects_3d']\n"
            "  before audio generation ran. Positions default to [0,0,5] via\n"
            "  create_audio_manifest() fallback.\n"
            "  -> Fix: ensure project_objects_to_3d() runs and its output is\n"
            "     assigned to state['objects_3d'] before generate_audio_batch()."
        )
    elif "B_DEPTH_MAPS_NONE" in dominant:
        diagnosis = (
            "DIAGNOSIS: Depth collapse (Mode B)\n"
            "  project_objects_to_3d was called but depth_maps=None, so every\n"
            "  object gets depth=5.0, all landing at (0,0,5) in camera frame.\n"
            "  -> Fix: preserve MiDaS depth_maps in reconstruct_scene() after\n"
            "     results.update(dust3r_results) overwrites them."
        )
    elif "C_REAL_DEPTH" in dominant:
        diagnosis = (
            "DIAGNOSIS: Camera frame mismatch (Mode C)\n"
            "  Objects have real depths but world positions look wrong.\n"
            "  The extrinsic matrix direction (world->cam vs cam->world) may\n"
            "  be applied incorrectly in unproject_2d_to_3d()."
        )
    else:
        diagnosis = "DIAGNOSIS: Mixed or unclear - inspect individual rows above."

    print(diagnosis)
    print()

    # MiDaS depth sanity
    real_midas = [r for r in rows if r["midas_depth"] is not None]
    if real_midas:
        depths = np.array([r["midas_depth"] for r in real_midas])
        print(f"MiDaS depth sanity (n={len(depths)} samples with .npy files):")
        print(f"  min={depths.min():.2f}  max={depths.max():.2f}  "
              f"mean={depths.mean():.2f}  std={depths.std():.2f}")
        if depths.std() > 0.5:
            print("  -> MiDaS values have spread: depth collapse is NOT the issue.")
        else:
            print("  -> MiDaS values nearly constant: possible depth model failure.")
    else:
        print("MiDaS depth sanity: no .npy files matched - depth maps unavailable.")

    return dominant


# ---------------------------------------------------------------------------
# Corrected 3-D positions using MiDaS + identity/synthetic cameras
# ---------------------------------------------------------------------------
def compute_corrected_positions(rows: List[dict]) -> List[Optional[np.ndarray]]:
    """
    Re-unproject each object using its saved MiDaS depth and a synthetic
    pinhole (fx=fy=0.7*W, cx=W/2, cy=H/2, identity extrinsic).

    This gives the best estimate of what positions WOULD look like if the
    depth_maps overwrite bug were fixed (depth-only mode, no DUSt3R pose).
    """
    corrected = []
    for r in rows:
        if r["midas_depth"] is None or r["midas_depth"] <= 0:
            corrected.append(None)
            continue
        w, h   = r["img_wh"]
        fx = fy = 0.7 * w
        cx, cy = w / 2.0, h / 2.0
        px, py = r["bbox_center_px"]
        pos = unproject_2d_to_3d(px, py, r["midas_depth"], fx, fy, cx, cy)
        corrected.append(pos)
    return corrected


# ---------------------------------------------------------------------------
# 3-D scatter plots
# ---------------------------------------------------------------------------
def plot_positions(
    rows: List[dict],
    corrected: List[Optional[np.ndarray]],
    save_path: Optional[Path] = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
    except ImportError:
        print("matplotlib not available — skipping plots.")
        return

    fig = plt.figure(figsize=(16, 7))

    # ── Panel 1: current manifest positions ──────────────────────────────
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.set_title("Current manifest positions\n(cluster = bug)", fontsize=11)

    manifest_pts = [r["manifest_pos"] for r in rows if r["manifest_pos"]]
    if manifest_pts:
        xs, ys, zs = zip(*manifest_pts)
        ax1.scatter(xs, ys, zs, c="navy", s=20, alpha=0.6, label="objects")

    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    ax1.view_init(elev=25, azim=45)

    # ── Panel 2: corrected positions (MiDaS depth, identity camera) ──────
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.set_title("Corrected positions\n(MiDaS depth + identity cam)", fontsize=11)

    corr_pts  = [c for c in corrected if c is not None]
    corr_rows = [r for r, c in zip(rows, corrected) if c is not None]

    if corr_pts:
        xs2, ys2, zs2 = zip(*[(p[0], p[1], p[2]) for p in corr_pts])
        depths2 = [r["midas_depth"] for r in corr_rows]
        sc = ax2.scatter(xs2, ys2, zs2, c=depths2, cmap="plasma", s=20, alpha=0.7)
        fig.colorbar(sc, ax=ax2, label="MiDaS depth", shrink=0.6)

    ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")
    ax2.view_init(elev=25, azim=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose 3-D spatial audio positions.")
    parser.add_argument("--csv",     metavar="PATH", help="Save table to CSV file.")
    parser.add_argument("--plot",    metavar="PATH", help="Save plot PNG instead of showing.")
    parser.add_argument("--no-plot", action="store_true", help="Skip the matplotlib scatter.")
    args = parser.parse_args()

    print("Loading segmentation results …")
    seg = load_segmentation(SEG_PATH)

    print("Loading MiDaS depth maps …")
    depth_maps = load_depth_maps(DEPTH_DIR)
    print(f"  Found {len(depth_maps)} depth map(s): indices {sorted(depth_maps.keys())}")

    print("Loading audio manifest …")
    manifest  = load_manifest(MANIFEST)
    pos_by_id = {s["id"]: s for s in manifest.get("audio_sources", [])}
    print(f"  {len(pos_by_id)} audio sources in manifest.")

    print("\nBuilding per-object rows …")
    rows = build_rows(seg, depth_maps, pos_by_id)
    print(f"  {len(rows)} objects across all segmentation results.\n")

    print_table(rows)
    print_summary(rows)

    # Optional CSV export
    if args.csv:
        try:
            import pandas as pd
            pd.DataFrame(rows).to_csv(args.csv, index=False)
            print(f"Table saved to CSV: {args.csv}")
        except Exception:
            import csv
            with open(args.csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print(f"Table saved to CSV (csv module): {args.csv}")

    # Corrected positions + plots
    corrected = compute_corrected_positions(rows)
    n_corrected = sum(1 for c in corrected if c is not None)
    print(f"\nCorrected positions computed for {n_corrected} / {len(rows)} objects "
          f"(those with a saved MiDaS .npy and depth > 0).")

    if not args.no_plot:
        save_path = Path(args.plot) if args.plot else None
        plot_positions(rows, corrected, save_path)


if __name__ == "__main__":
    main()
