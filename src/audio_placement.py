"""Audio placement helpers for the 3DGS scene.

Provides the ``AudioPlacementCandidate`` dataclass, YAML persistence, and
functions to composite audio-source markers into an existing Gaussian PLY.

Typical workflow (Stage 8 in pipeline.ipynb)
--------------------------------------------
1.  ``candidates = build_auto_candidates(state, gs_ply, audio_dir, out_dir)``
    Builds candidates from depth+segmentation, auto-discovers audio files,
    and snaps every position to the nearest Gaussian in the 3DGS PLY so
    markers land precisely on reconstructed geometry.
2.  User enables/disables candidates with the checkbox cull UI.
3.  ``save_placements(candidates, path)``       → writes data/output/audio_placement.yaml
4.  ``compose_gs_scene(candidates, gs_ply, out_ply)``   → PLY + JSON sidecar
5.  ``export_for_unity_gs(candidates, out_json)``        → Unity-ready JSON
"""

from __future__ import annotations

import json
import logging
import os
import re
import struct
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PLY I/O helpers (no open3d required)
# ---------------------------------------------------------------------------

def _read_ply_xyz_numpy(path: str) -> Optional[np.ndarray]:
    """Read XYZ coordinates from an ASCII or binary PLY file using only numpy/stdlib.

    Returns an (N, 3) float64 array, or ``None`` if the file is absent / unreadable.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as fh:
            fmt = "ascii"
            n_verts = 0
            props: List[Tuple[str, str]] = []
            in_vertex = False
            while True:
                raw = fh.readline()
                line = raw.decode("ascii", errors="ignore").strip()
                if line.startswith("format"):
                    if "binary_little_endian" in line:
                        fmt = "binary_little"
                    elif "binary_big_endian" in line:
                        fmt = "binary_big"
                    else:
                        fmt = "ascii"
                elif line.startswith("element vertex"):
                    n_verts = int(line.split()[-1])
                    in_vertex = True
                elif line.startswith("element") and in_vertex:
                    in_vertex = False
                elif line.startswith("property") and in_vertex:
                    parts = line.split()
                    props.append((parts[1], parts[2]))  # (type_str, name)
                elif line == "end_header":
                    break

            pnames = [p[1] for p in props]
            if "x" not in pnames:
                return None
            xi, yi, zi = pnames.index("x"), pnames.index("y"), pnames.index("z")

            if fmt == "ascii":
                pts: List[List[float]] = []
                for _ in range(n_verts):
                    row = fh.readline().decode("ascii", errors="ignore").strip().split()
                    if len(row) > max(xi, yi, zi):
                        pts.append([float(row[xi]), float(row[yi]), float(row[zi])])
                return np.array(pts, dtype=np.float64) if pts else None
            else:
                _SIZES = {
                    "float": 4, "float32": 4, "double": 8, "float64": 8,
                    "uchar": 1, "uint8": 1, "char": 1, "int8": 1,
                    "short": 2, "ushort": 2, "int16": 2, "uint16": 2,
                    "int": 4, "uint": 4, "int32": 4, "uint32": 4,
                }
                _FMTS = {
                    "float": "f", "float32": "f", "double": "d", "float64": "d",
                    "uchar": "B", "uint8": "B", "char": "b", "int8": "b",
                    "short": "h", "ushort": "H", "int16": "h", "uint16": "H",
                    "int": "i", "uint": "I", "int32": "i", "uint32": "I",
                }
                endian = "<" if fmt == "binary_little" else ">"
                offsets = []
                stride = 0
                for typ, _ in props:
                    offsets.append(stride)
                    stride += _SIZES.get(typ, 4)
                data = fh.read(stride * n_verts)
                n_actual = len(data) // stride
                pts_arr = np.zeros((n_actual, 3), dtype=np.float64)
                for ax, col in ((xi, 0), (yi, 1), (zi, 2)):
                    typ = props[ax][0]
                    sf = endian + _FMTS.get(typ, "f")
                    sz = _SIZES.get(typ, 4)
                    offs = offsets[ax]
                    vals = struct.unpack_from(
                        endian + str(n_actual) + _FMTS.get(typ, "f"),
                        data,
                        # stride-based slicing via comprehension
                    ) if False else [
                        struct.unpack_from(sf, data, i * stride + offs)[0]
                        for i in range(n_actual)
                    ]
                    pts_arr[:, col] = vals
                return pts_arr
    except Exception as exc:
        logger.warning(f"_read_ply_xyz_numpy: failed to read {path}: {exc}")
        return None


def _snap_to_points_numpy(
    candidates: "List[AudioPlacementCandidate]",
    pts_ply: np.ndarray,
) -> None:
    """Snap each candidate's ``position_3d`` to the nearest point in *pts_ply*.

    Uses ``scipy.spatial.KDTree`` when available, falls back to a brute-force
    nearest-neighbour search otherwise.  Modifies candidates in-place.
    """
    if pts_ply is None or len(pts_ply) == 0:
        return
    pts_ply = np.asarray(pts_ply, dtype=np.float64)
    try:
        from scipy.spatial import KDTree  # type: ignore
        tree = KDTree(pts_ply)
        positions = np.array([c.position_3d for c in candidates], dtype=np.float64)
        _, idxs = tree.query(positions)
        for c, idx in zip(candidates, idxs):
            c.position_3d = pts_ply[idx].tolist()
    except ImportError:
        # Brute-force fallback (slow for large clouds — only hits if scipy absent)
        logger.warning("scipy not available; using brute-force nearest-neighbour snap.")
        for c in candidates:
            q = np.array(c.position_3d, dtype=np.float64)
            dists = np.sum((pts_ply - q) ** 2, axis=1)
            c.position_3d = pts_ply[int(np.argmin(dists))].tolist()


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class AudioPlacementCandidate:
    """Represents one candidate audio source for 3DGS placement."""

    id: str
    label: str
    audio_file: str
    position_3d: List[float]        # [x, y, z] in 3DGS PLY frame (after snapping)
    depth: float = 0.0
    confidence: float = 0.0
    image_index: int = 0
    enabled: bool = False           # whether this source is placed in the scene
    dx: float = 0.0                 # user nudge X
    dy: float = 0.0                 # user nudge Y
    dz: float = 0.0                 # user nudge Z
    intensity_override: float = -1.0  # -1 = auto, 0–1 = manual

    @property
    def final_position(self) -> List[float]:
        """``position_3d`` plus the user nudge offsets."""
        x, y, z = self.position_3d
        return [x + self.dx, y + self.dy, z + self.dz]

    @property
    def final_intensity(self) -> float:
        """Resolved intensity (manual override or heuristic from depth)."""
        if self.intensity_override >= 0:
            return float(np.clip(self.intensity_override, 0.0, 1.0))
        # Depth-based heuristic: closer = louder (normalised 0→1)
        d = max(self.depth, 0.01)
        return float(np.clip(1.0 / (1.0 + d), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Build candidates from pipeline state
# ---------------------------------------------------------------------------

def _label_to_slug(label: str) -> str:
    """Convert an object label to a filesystem-safe slug.

    Mirrors the logic in ``audio_generation._sanitize_label_for_audio_filename``.
    """
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def _build_audio_index(audio_dir: Optional[str]) -> Dict[str, str]:
    """Return a dict mapping slug → absolute wav path for every .wav in *audio_dir*."""
    if not audio_dir:
        return {}
    root = Path(audio_dir)
    if not root.is_dir():
        return {}
    return {p.stem.lower(): str(p) for p in root.glob("*.wav")}


def build_candidates(
    state: Dict[str, Any],
    audio_dir: Optional[str] = None,
) -> List[AudioPlacementCandidate]:
    """Merge ``state['objects_with_audio']`` and ``state['objects_3d']``.

    Mirrors the position-attachment logic in pipeline.ipynb Stage 6 so the
    world-space positions are consistent.

    If ``audio_dir`` is provided, any candidate whose ``audio_file`` is
    missing or points to a non-existent path is matched against .wav files
    in that directory by label slug so audio is discovered automatically.

    Args:
        state:     Pipeline state dict; expects keys ``objects_with_audio``
                   and (optionally) ``objects_3d``.
        audio_dir: Directory to scan for .wav files when ``audio_file`` is
                   absent or stale (e.g. ``'data/audio'``).

    Returns:
        List of ``AudioPlacementCandidate`` (all disabled by default).
        Positions are in the DUSt3R world frame at this stage; call
        ``snap_positions_to_gs_ply`` afterwards to convert to PLY frame.
    """
    objects_with_audio: List[Dict] = state.get("objects_with_audio", [])
    objects_3d: List[Dict] = state.get("objects_3d", [])

    pos_by_id: Dict[str, List[float]] = {
        o["id"]: o["position_3d"] for o in objects_3d if "position_3d" in o
    }
    depth_by_id: Dict[str, float] = {
        o["id"]: float(o.get("depth", 0.0)) for o in objects_3d
    }

    # Build a slug → wav path index for auto-discovery
    audio_index = _build_audio_index(audio_dir)

    candidates: List[AudioPlacementCandidate] = []
    for obj in objects_with_audio:
        audio_file = obj.get("audio_file", "")

        # Auto-discover audio if the stored path is missing or invalid
        if not audio_file or not Path(str(audio_file)).exists():
            slug = _label_to_slug(obj.get("label", ""))
            # Try exact slug match first, then prefix match
            discovered = audio_index.get(slug)
            if discovered is None:
                for stem, wav_path in audio_index.items():
                    if stem.startswith(slug) or slug.startswith(stem):
                        discovered = wav_path
                        break
            if discovered:
                logger.info(
                    f"  Auto-discovered audio for '{obj.get('label')}': {discovered}"
                )
                audio_file = discovered
            else:
                logger.debug(
                    f"  No audio file found for '{obj.get('label')}' (slug={slug!r}) — skipping."
                )
                continue

        obj_id = obj.get("id", "")
        pos = pos_by_id.get(obj_id, obj.get("position_3d", [0.0, 0.0, 5.0]))
        depth = depth_by_id.get(obj_id, float(obj.get("depth", 0.0)))

        candidates.append(
            AudioPlacementCandidate(
                id=obj_id,
                label=obj.get("label", "unknown"),
                audio_file=str(audio_file),
                position_3d=list(pos),
                depth=depth,
                confidence=float(obj.get("confidence", 0.0)),
                image_index=int(obj.get("image_index", 0)),
            )
        )

    logger.info(f"Built {len(candidates)} placement candidates.")
    return candidates


# ---------------------------------------------------------------------------
# PLY-frame position snapping
# ---------------------------------------------------------------------------

def _find_dataparser_transforms(out_dir: str) -> Optional[str]:
    """Return the path of the most recently modified ``dataparser_transforms.json``
    anywhere under *out_dir* (mirrors ``_find_nerfstudio_config`` in gs_train).
    """
    found = sorted(
        Path(out_dir).rglob("dataparser_transforms.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if not found:
        return None
    return str(found[-1])


def _load_dataparser_transforms(path: Optional[str]):
    """Load R (3×3), t (3,), scale from ``dataparser_transforms.json``.

    Returns identity transform (R=I, t=0, scale=1) if the file is absent.
    """
    R = np.eye(3, dtype=np.float64)
    t = np.zeros(3, dtype=np.float64)
    scale = 1.0
    if path and Path(path).exists():
        with open(path, encoding="utf-8") as fh:
            dt = json.load(fh)
        mat = np.array(dt["transform"], dtype=np.float64)   # (3, 4)
        R = mat[:, :3]
        t = mat[:, 3]
        scale = float(dt["scale"])
    return R, t, scale


def _compute_positions_from_pointcloud(
    objects_with_audio: List[Dict],
    seg_results: List[Dict],
    transforms_json_path: str,
    pointcloud_path: str,
    R_dp: np.ndarray,
    t_dp: np.ndarray,
    scale_dp: float,
) -> Dict[str, tuple]:
    """Project a DUSt3R point cloud into each camera view to find per-object 3D positions.

    For each detected object (image_index + normalised bbox from segmentation),
    the DUSt3R point cloud is projected into the corresponding camera.  The
    median 3D position of the points that land inside the bbox is used as the
    object's world position.  That position is then converted to the Nerfstudio
    PLY frame via ``scale_dp * (R_dp @ pos + t_dp)``.

    This method avoids MiDaS depth maps entirely (which are in a relative,
    non-metric scale incompatible with the DUSt3R camera poses).

    No open3d dependency — uses only numpy and stdlib.

    Args:
        objects_with_audio: List of object dicts with ``id`` and ``image_index``.
        seg_results:        Segmentation results list (for normalised bboxes).
        transforms_json_path: Path to ``data/reconstruction/3dgs/data/transforms.json``.
        pointcloud_path:    Path to ``dust3r_points.ply`` or ``sparse_pc.ply``
                            (DUSt3R world frame).
        R_dp, t_dp, scale_dp: Dataparser transform components.

    Returns:
        Dict mapping ``obj_id`` → ``(position_ply [x,y,z], depth_camera_units)``.
        Objects with no point-cloud coverage are omitted.
    """
    if not Path(transforms_json_path).exists():
        logger.warning(f"transforms.json not found at {transforms_json_path!r}")
        return {}

    # ── Load point cloud (no open3d needed) ───────────────────────────────
    pts = _read_ply_xyz_numpy(pointcloud_path)
    if pts is None or len(pts) == 0:
        logger.warning(f"Point cloud not found or empty at {pointcloud_path!r}")
        return {}
    pts = pts.astype(np.float64)
    logger.info(f"  Loaded {len(pts):,} points from {pointcloud_path}")

    # ── Load camera info from transforms.json ─────────────────────────────
    with open(transforms_json_path, encoding="utf-8") as fh:
        tf = json.load(fh)

    # Global intrinsics (may also be per-frame; we use global if present)
    fl_x = float(tf.get("fl_x") or tf["frames"][0].get("fl_x", 1000))
    fl_y = float(tf.get("fl_y") or tf.get("fl_x") or fl_x)
    cx   = float(tf.get("cx") or tf["frames"][0].get("cx", 0))
    cy   = float(tf.get("cy") or tf["frames"][0].get("cy", 0))
    w_img = float(tf.get("w") or tf["frames"][0].get("w", 1))
    h_img = float(tf.get("h") or tf["frames"][0].get("h", 1))

    c2w_list = [
        np.array(fr["transform_matrix"], dtype=np.float64)
        for fr in tf.get("frames", [])
    ]
    w2c_list = [np.linalg.inv(m) for m in c2w_list]

    # Homogeneous form for batch projection: (4, N)
    pts_h = np.vstack([pts.T, np.ones((1, len(pts)))])

    # ── Build bbox lookup from segmentation results ────────────────────────
    bbox_by_id: Dict[str, tuple] = {}
    for seg in seg_results:
        img_idx = int(seg.get("image_index", 0))
        for j, obj in enumerate(seg.get("objects", [])):
            oid = f"obj_{img_idx:03d}_{j:03d}"
            bbox_by_id[oid] = (obj.get("bbox"), img_idx)

    result: Dict[str, tuple] = {}

    for obj in objects_with_audio:
        obj_id = obj.get("id", "")
        info = bbox_by_id.get(obj_id)
        if not info:
            continue
        bbox, img_idx = info
        if not bbox or img_idx >= len(w2c_list):
            continue

        # Normalised bbox [x1, y1, x2, y2] → pixel coords
        x1, y1, x2, y2 = (
            bbox[0] * w_img, bbox[1] * h_img,
            bbox[2] * w_img, bbox[3] * h_img,
        )

        # Project all points into this camera (OpenGL: camera looks in -Z)
        p_cam = w2c_list[img_idx] @ pts_h   # (4, N)
        depth_vals = -p_cam[2]              # positive = in front of camera

        in_front = depth_vals > 0.01
        # Guard against divide-by-zero
        safe_z = np.where(in_front, -p_cam[2], 1.0)
        x_proj = fl_x * p_cam[0] / safe_z + cx
        y_proj = fl_y * p_cam[1] / safe_z + cy

        in_bbox = in_front & (x_proj >= x1) & (x_proj <= x2) & (y_proj >= y1) & (y_proj <= y2)

        if in_bbox.sum() == 0:
            # Expand bbox by 30 % and retry
            dw = (x2 - x1) * 0.3
            dh = (y2 - y1) * 0.3
            in_bbox = in_front & (
                (x_proj >= x1 - dw) & (x_proj <= x2 + dw) &
                (y_proj >= y1 - dh) & (y_proj <= y2 + dh)
            )

        if in_bbox.sum() == 0:
            continue   # no point cloud coverage for this bbox

        matched_pts  = pts[in_bbox]            # (K, 3) world frame
        matched_dept = depth_vals[in_bbox]     # (K,)   camera-space depth

        # Use median to suppress outliers
        p_world   = np.median(matched_pts, axis=0)
        depth_val = float(np.median(matched_dept))

        # Convert DUSt3R world → Nerfstudio PLY frame
        p_ply = scale_dp * (R_dp @ p_world + t_dp)

        result[obj_id] = (p_ply.tolist(), depth_val)

    logger.info(
        f"  Point-cloud projection: {len(result)}/{len(objects_with_audio)} objects resolved."
    )
    return result


def snap_to_nearest_gaussian(
    candidates: List[AudioPlacementCandidate],
    gs_ply: str,
) -> List[AudioPlacementCandidate]:
    """Snap each candidate's ``position_3d`` to the nearest Gaussian centre.

    Both the candidate positions and the Gaussian PLY are expected to be in
    the *same* Nerfstudio PLY frame (no further coordinate conversion is done
    here).  Call ``_compute_positions_from_pointcloud`` first to obtain
    correctly-converted positions.

    Uses scipy KDTree (no open3d required).

    Args:
        candidates: List of ``AudioPlacementCandidate`` with ``position_3d``
                    already in PLY frame.
        gs_ply:     Path to the trained / cleaned Gaussian PLY.

    Returns:
        The same list with ``position_3d`` snapped in-place.
    """
    if not candidates or not Path(gs_ply).exists():
        return candidates

    gs_pts = _read_ply_xyz_numpy(gs_ply)
    if gs_pts is None or len(gs_pts) == 0:
        logger.warning(f"  snap_to_nearest_gaussian: could not read {gs_ply!r}")
        return candidates

    logger.info(f"  Snapping {len(candidates)} positions to {len(gs_pts):,} Gaussians in {gs_ply}")
    _snap_to_points_numpy(candidates, gs_pts)
    logger.info("  Snap complete.")
    return candidates


def build_auto_candidates(
    state: Dict[str, Any],
    gs_ply: Optional[str] = None,
    audio_dir: str = "data/audio",
    out_dir: str = "data/reconstruction/3dgs",
) -> List[AudioPlacementCandidate]:
    """Full auto-placement pipeline for Stage 8a.

    Position computation strategy (in order of preference):

    1. **Point-cloud projection** — projects ``dust3r_points.ply`` (or
       ``sparse_pc.ply``) into each camera view using the cameras from
       ``transforms.json`` and finds the 3D points that fall inside the
       object's bounding box.  This is geometry-based and does **not** use
       MiDaS depth maps (which are relative-scale and incompatible with
       DUSt3R metric camera poses).

    2. **``state['objects_3d']`` fallback** — if the point-cloud approach
       yields no position for an object, the previously-computed DUSt3R
       world position (if available in state) is used after applying the
       dataparser transform.

    3. **Snap to nearest Gaussian** — all positions (regardless of how they
       were obtained) are snapped to the nearest Gaussian centre in the
       trained PLY so audio markers sit precisely on reconstructed geometry.

    4. Sorts by ``confidence`` descending and enables all by default
       (opt-out cull UI in Stage 8a).

    Args:
        state:     Pipeline state dict; may contain ``objects_with_audio``,
                   ``objects_3d``, and ``segmentation_results``.
        gs_ply:    Path to the trained Gaussian PLY (``gs_scene_clean.ply``
                   or ``gs_scene.ply``).
        audio_dir: Directory containing generated .wav files
                   (default ``data/audio``).
        out_dir:   Nerfstudio output root used to locate
                   ``dataparser_transforms.json``
                   (default ``data/reconstruction/3dgs``).

    Returns:
        List of ``AudioPlacementCandidate`` sorted by confidence.
        All candidates start **enabled=True**.
    """
    # Step 1 — basic merge + audio auto-discovery (positions still default here)
    candidates = build_candidates(state, audio_dir=audio_dir)
    if not candidates:
        return candidates

    # Step 2 — load dataparser transforms (needed for both strategies)
    dp_path = _find_dataparser_transforms(out_dir)
    if dp_path:
        logger.info(f"Using dataparser transforms: {dp_path}")
    else:
        logger.warning(
            f"No dataparser_transforms.json found under {out_dir!r}. "
            "Positions will be in raw DUSt3R world frame."
        )
    R_dp, t_dp, scale_dp = _load_dataparser_transforms(dp_path)

    # Step 3 — geometry-based position computation via point-cloud projection
    tf_json = str(
        Path(out_dir) / "data" / "transforms.json"
    )
    # Find the best available point cloud (DUSt3R full > sparse_pc)
    pc_candidates = [
        "data/reconstruction/dust3r/dust3r_points.ply",
        "data/reconstruction/3dgs/data/sparse_pc.ply",
    ]
    pc_path: Optional[str] = None
    for p in pc_candidates:
        if Path(p).exists():
            pc_path = p
            logger.info(f"Using point cloud for position computation: {p}")
            break

    seg_results: List[Dict] = state.get("segmentation_results", [])
    # If not in state, try loading from disk
    if not seg_results:
        seg_disk = "data/segmentation/all_segmentations.json"
        if Path(seg_disk).exists():
            import json as _json_module
            with open(seg_disk, encoding="utf-8") as fh:
                seg_results = _json_module.load(fh)
            logger.info(f"Loaded {len(seg_results)} segmentation results from disk.")

    pos_map: Dict[str, tuple] = {}
    if pc_path and seg_results and Path(tf_json).exists():
        objects_with_audio: List[Dict] = state.get("objects_with_audio", [])
        # Rebuild objects_with_audio from scratch if missing — include ALL
        # detected objects so the point-cloud projection can resolve positions
        # for every bbox.  Audio auto-discovery in build_candidates will match
        # label slugs to the actual .wav files later.
        if not objects_with_audio:
            objects_with_audio = []
            for seg in seg_results:
                img_idx = int(seg.get("image_index", 0))
                for j, obj in enumerate(seg.get("objects", [])):
                    oid = f"obj_{img_idx:03d}_{j:03d}"
                    objects_with_audio.append({
                        "id": oid,
                        "label": obj.get("label", "unknown"),
                        "image_index": img_idx,
                    })
                    _global_idx += 1

        pos_map = _compute_positions_from_pointcloud(
            objects_with_audio=objects_with_audio,
            seg_results=seg_results,
            transforms_json_path=tf_json,
            pointcloud_path=pc_path,
            R_dp=R_dp,
            t_dp=t_dp,
            scale_dp=scale_dp,
        )
    else:
        logger.warning(
            "Skipping point-cloud projection "
            f"(pc_path={pc_path!r}, seg_results={len(seg_results)}, "
            f"transforms={Path(tf_json).exists()})."
        )

    # Step 4 — fall back to state['objects_3d'] for any unresolved object
    objects_3d: List[Dict] = state.get("objects_3d", [])
    pos_by_id_state = {
        o["id"]: o["position_3d"]
        for o in objects_3d
        if "position_3d" in o
    }

    # Load camera centres in PLY frame for the per-camera fallback
    cam_centers_ply: List[np.ndarray] = []
    if Path(tf_json).exists():
        with open(tf_json, encoding="utf-8") as fh:
            _tf = json.load(fh)
        for fr in _tf.get("frames", []):
            m = np.array(fr["transform_matrix"], dtype=np.float64)
            cam_pos_world = m[:3, 3]
            cam_pos_ply   = scale_dp * (R_dp @ cam_pos_world + t_dp)
            cam_centers_ply.append(cam_pos_ply)

    n_from_pc = 0
    n_from_state = 0
    n_from_cam = 0
    n_default = 0
    for c in candidates:
        if c.id in pos_map:
            c.position_3d, c.depth = pos_map[c.id][0], pos_map[c.id][1]
            n_from_pc += 1
        elif c.id in pos_by_id_state:
            # Convert state position (raw world) → PLY frame
            p = np.array(pos_by_id_state[c.id], dtype=np.float64)
            c.position_3d = (scale_dp * (R_dp @ p + t_dp)).tolist()
            n_from_state += 1
        elif cam_centers_ply:
            # Fallback: use the nearest available camera centre in PLY frame;
            # snapping will pull it to the nearest geometry point from there
            cam_idx = min(c.image_index, len(cam_centers_ply) - 1)
            c.position_3d = cam_centers_ply[cam_idx].tolist()
            n_from_cam += 1
        else:
            n_default += 1

    logger.info(
        f"Positions: {n_from_pc} from point-cloud, "
        f"{n_from_state} from state, "
        f"{n_from_cam} from camera-centre fallback, "
        f"{n_default} fully defaulted."
    )

    # Step 5 — snap every position to the nearest reconstructed geometry point.
    # Priority: GS PLY first (actual rendered scene), then DUSt3R point cloud.
    # Uses numpy + scipy — no open3d required.
    snapped = False

    # Try GS PLY first — this is the geometry the viewer actually renders
    _gs_snap_path = str(gs_ply) if gs_ply else None
    if not _gs_snap_path or not Path(_gs_snap_path).exists():
        # Prefer clean PLY; fall back to raw
        for _candidate_ply in [
            "data/reconstruction/3dgs/gs_scene_clean.ply",
            "data/reconstruction/3dgs/gs_scene.ply",
        ]:
            if Path(_candidate_ply).exists():
                _gs_snap_path = _candidate_ply
                break

    if _gs_snap_path and Path(_gs_snap_path).exists():
        _gs_pts = _read_ply_xyz_numpy(_gs_snap_path)
        if _gs_pts is not None and len(_gs_pts) > 0:
            logger.info(
                f"  Snapping {len(candidates)} positions to "
                f"{len(_gs_pts):,} GS points in {_gs_snap_path}"
            )
            _snap_to_points_numpy(candidates, _gs_pts)
            logger.info("  Snap to GS PLY complete.")
            snapped = True

    # Fallback: DUSt3R point cloud converted to PLY frame
    if not snapped and pc_path and Path(pc_path).exists():
        _pts_w = _read_ply_xyz_numpy(pc_path)
        if _pts_w is not None and len(_pts_w) > 0:
            _pts_w = _pts_w.astype(np.float64)
            _pts_ply = (scale_dp * (R_dp @ _pts_w.T + t_dp[:, None])).T
            logger.info(
                f"  Snapping {len(candidates)} positions to "
                f"{len(_pts_ply):,} DUSt3R points (PLY frame)"
            )
            _snap_to_points_numpy(candidates, _pts_ply)
            logger.info("  Snap to DUSt3R complete.")
            snapped = True

    if not snapped:
        logger.warning(
            "No point cloud available for snapping — positions remain as-is."
        )

    # Step 6 — enable all (opt-out model) and sort by confidence
    for c in candidates:
        c.enabled = True
    candidates.sort(key=lambda c: c.confidence, reverse=True)

    logger.info(
        f"build_auto_candidates: {len(candidates)} candidates, all enabled."
    )
    return candidates


# ---------------------------------------------------------------------------
# YAML persistence
# ---------------------------------------------------------------------------

def save_placements(candidates: List[AudioPlacementCandidate], path: str) -> None:
    """Serialise ``candidates`` to a YAML file at ``path``.

    Args:
        candidates: List of ``AudioPlacementCandidate``.
        path:       Destination path (parent dirs created automatically).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(c) for c in candidates]
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False)
    enabled = sum(1 for c in candidates if c.enabled)
    logger.info(f"Saved {len(candidates)} candidates ({enabled} enabled) → {path}")


def load_placements(path: str) -> List[AudioPlacementCandidate]:
    """Load ``AudioPlacementCandidate`` list from a YAML file.

    Args:
        path: Path written by ``save_placements``.

    Returns:
        List of ``AudioPlacementCandidate`` (enabled state preserved).
    """
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or []
    candidates = []
    for item in raw:
        try:
            candidates.append(AudioPlacementCandidate(**item))
        except TypeError as exc:
            logger.warning(f"Skipping malformed placement entry: {exc}")
    logger.info(f"Loaded {len(candidates)} candidates from {path}")
    return candidates


# ---------------------------------------------------------------------------
# Composite: Gaussian PLY + audio-source markers
# ---------------------------------------------------------------------------

def _sample_sphere_points(
    center: np.ndarray,
    radius: float,
    color: np.ndarray,
    n_points: int = 300,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (pts, cols) for a sphere sampled with numpy (no open3d needed)."""
    rng = np.random.default_rng(seed=int(abs(center[0]) * 1e6) % (2**31))
    theta = rng.uniform(0, 2 * np.pi, n_points)
    phi   = np.arccos(rng.uniform(-1, 1, n_points))
    pts = center + radius * np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi),
    ])
    cols = np.tile(np.clip(color, 0, 1), (n_points, 1))
    return pts.astype(np.float32), cols.astype(np.float32)


def _write_ply_xyzrgb(path: str, pts: np.ndarray, cols: np.ndarray) -> None:
    """Write an ASCII PLY with XYZ + RGB (uint8) columns."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = len(pts)
    rgb = np.clip(cols * 255, 0, 255).astype(np.uint8)
    with open(path, "w", encoding="ascii") as fh:
        fh.write(
            f"ply\nformat ascii 1.0\nelement vertex {n}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        )
        for i in range(n):
            fh.write(
                f"{pts[i,0]:.6f} {pts[i,1]:.6f} {pts[i,2]:.6f} "
                f"{rgb[i,0]} {rgb[i,1]} {rgb[i,2]}\n"
            )


def compose_gs_scene(
    candidates: List[AudioPlacementCandidate],
    gs_ply: str,
    out_ply: str,
    marker_radius: float = 0.1,
) -> Tuple_str:
    """Add coloured sphere markers for enabled candidates into the GS PLY.

    The GS point cloud and the audio-source markers are merged into a single
    combined PLY (same world frame — no transformation needed).  A JSON
    sidecar ``<out_ply>.json`` is written alongside listing each enabled
    source with its final position, intensity, and audio path.

    Works without open3d — uses numpy for sphere sampling and writes ASCII PLY.

    Args:
        candidates:    List from ``build_candidates`` / ``load_placements``.
        gs_ply:        Path to the trained Gaussian PLY.
        out_ply:       Output PLY path for the composite scene.
        marker_radius: Sphere radius for audio-source markers (world units).

    Returns:
        Path to the written composite PLY.
    """
    enabled = [c for c in candidates if c.enabled]
    logger.info(
        f"Compositing {len(enabled)} enabled audio sources into {gs_ply} → {out_ply}"
    )

    # Load base point cloud XYZ+RGB (no open3d needed)
    base_pts: Optional[np.ndarray] = None
    base_cols: Optional[np.ndarray] = None

    if gs_ply and Path(gs_ply).exists():
        try:
            # Try open3d first (preserves colour) …
            import open3d as o3d  # type: ignore
            pcd = o3d.io.read_point_cloud(gs_ply)
            base_pts  = np.asarray(pcd.points,  dtype=np.float32)
            base_cols = (
                np.asarray(pcd.colors, dtype=np.float32)
                if pcd.has_colors()
                else np.full((len(base_pts), 3), 0.5, dtype=np.float32)
            )
            logger.info(f"  Loaded {len(base_pts):,} GS points via open3d")
        except Exception:
            # … fall back to our numpy reader (XYZ only; grey colour)
            base_pts = _read_ply_xyz_numpy(gs_ply)
            if base_pts is not None:
                base_pts  = base_pts.astype(np.float32)
                base_cols = np.full((len(base_pts), 3), 0.5, dtype=np.float32)
                logger.info(f"  Loaded {len(base_pts):,} GS points via numpy reader")
            else:
                logger.warning(f"GS PLY not readable at {gs_ply!r} — markers only.")

    # Build sphere marker points for each enabled candidate
    marker_pts_list:  List[np.ndarray] = []
    marker_cols_list: List[np.ndarray] = []
    for c in enabled:
        pos       = np.array(c.final_position, dtype=np.float64)
        intensity = c.final_intensity
        color     = np.array([intensity, 0.2, 1.0 - intensity], dtype=np.float32)
        mp, mc    = _sample_sphere_points(pos, marker_radius, color, n_points=300)
        marker_pts_list.append(mp)
        marker_cols_list.append(mc)

    # Merge base + markers
    all_pts_parts  = []
    all_cols_parts = []
    if base_pts is not None:
        all_pts_parts.append(base_pts)
        all_cols_parts.append(base_cols)
    if marker_pts_list:
        all_pts_parts.append(np.vstack(marker_pts_list))
        all_cols_parts.append(np.vstack(marker_cols_list))

    if not all_pts_parts:
        logger.warning("compose_gs_scene: nothing to write.")
        return out_ply

    merged_pts  = np.vstack(all_pts_parts)
    merged_cols = np.vstack(all_cols_parts)

    _write_ply_xyzrgb(out_ply, merged_pts, merged_cols)
    logger.info(f"Composite PLY written: {out_ply} ({len(merged_pts):,} points)")

    # JSON sidecar
    sidecar_path = str(out_ply) + ".json"
    sidecar = {
        "audio_sources": [
            {
                "id": c.id,
                "label": c.label,
                "audio_file": c.audio_file,
                "position": c.final_position,
                "intensity": c.final_intensity,
                "depth": c.depth,
                "confidence": c.confidence,
            }
            for c in enabled
        ]
    }
    with open(sidecar_path, "w", encoding="utf-8") as fh:
        json.dump(sidecar, fh, indent=2)
    logger.info(f"Sidecar JSON written: {sidecar_path}")

    return out_ply


# ---------------------------------------------------------------------------
# Unity export
# ---------------------------------------------------------------------------

def export_for_unity_gs(
    candidates: List[AudioPlacementCandidate],
    out_json: str,
) -> str:
    """Export enabled placements as a Unity-compatible JSON.

    Args:
        candidates: List of ``AudioPlacementCandidate``.
        out_json:   Output path for the JSON file.

    Returns:
        Path to the written JSON.
    """
    enabled = [c for c in candidates if c.enabled]
    logger.info(f"Exporting {len(enabled)} enabled sources for Unity → {out_json}")

    unity_data = {
        "audioSources": [
            {
                "name": c.label,
                "audioClipPath": c.audio_file,
                "position": {
                    "x": c.final_position[0],
                    "y": c.final_position[1],
                    "z": c.final_position[2],
                },
                "volume": c.final_intensity,
                "spatialBlend": 1.0,
                "minDistance": 0.5,
                "maxDistance": 10.0,
            }
            for c in enabled
        ]
    }

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(unity_data, fh, indent=2)
    logger.info(f"Unity JSON written: {out_json}")
    return out_json


# ---------------------------------------------------------------------------
# Type alias used in compose_gs_scene return type
# ---------------------------------------------------------------------------

Tuple_str = str  # simple alias so the function annotation parses without 'from typing'
