"""3D Gaussian Splatting training module.

Bridges DUSt3R outputs (camera_data + PLY point cloud) to Nerfstudio's
splatfacto trainer, then exports the trained Gaussians as a PLY for
downstream audio placement.

Quick-start
-----------
1.  Create the nerfstudio conda env (once, outside MVTSG):
        conda create -n nerfstudio python=3.10 -y
        conda activate nerfstudio
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
        pip install nerfstudio
        ns-install-cli
2.  Set  config.yaml  gaussian_splatting.enabled: true
3.  Run  Stage 7  in pipeline.ipynb

Fallback
--------
If Nerfstudio is not available, set  gaussian_splatting.env_name: null.
The bridge still writes transforms.json so you can train manually later.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Nerfstudio / transforms.json bridge
# ---------------------------------------------------------------------------

def export_dust3r_to_nerfstudio(
    camera_data: Dict,
    image_paths: List[str],
    sparse_ply: Optional[str],
    out_dir: str,
) -> str:
    """Write a Nerfstudio-compatible ``transforms.json`` from DUSt3R camera data.

    Args:
        camera_data:  Dict from ``DUSt3RReconstructor.reconstruct()`` with keys
                      ``intrinsics``, ``extrinsics``, ``camera_centers``.
        image_paths:  Ordered list of image file paths (same order as images
                      fed to DUSt3R, i.e. ``data/Nano_banana_output_images``).
        sparse_ply:   Path to ``dust3r_points.ply``; copied to
                      ``<out_dir>/sparse_pc.ply`` for Gaussian seed init.
        out_dir:      Root output directory (e.g. ``data/reconstruction/3dgs/data``).

    Returns:
        Path to the written ``transforms.json``.
    """
    out_dir = Path(out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    intrinsics: Dict = camera_data.get("intrinsics", {})
    extrinsics: Dict = camera_data.get("extrinsics", {})

    if not intrinsics:
        raise ValueError("camera_data has no intrinsics — run DUSt3R first.")

    frames = []
    for i, src_path in enumerate(image_paths):
        if i not in intrinsics:
            logger.warning(f"No camera data for index {i}, skipping frame.")
            continue

        K: np.ndarray = intrinsics[i]          # 3×3
        w2c: np.ndarray = extrinsics[i]        # 4×4 world→cam

        # Nerfstudio expects cam-to-world (c2w) in OpenGL convention
        # (Y-up, Z-back).  DUSt3R outputs poses in OpenCV convention
        # (Y-down, Z-forward), so flip columns 1 and 2 of the rotation
        # block — the same transform used by ns-process-data / instant-ngp.
        # The translation column (camera position) is unchanged.
        c2w = np.linalg.inv(w2c)               # 4×4
        c2w[:3, 1] *= -1                        # flip Y axis
        c2w[:3, 2] *= -1                        # flip Z axis

        # Copy image into out_dir/images/
        dst_name = f"frame_{i:04d}{Path(src_path).suffix}"
        dst_path = images_dir / dst_name
        if Path(src_path).exists():
            shutil.copy2(src_path, dst_path)
        else:
            logger.warning(f"Image not found: {src_path}")

        # Determine W/H from image file if available
        try:
            from PIL import Image as _PILImage
            with _PILImage.open(src_path) as im:
                w_px, h_px = im.size
        except Exception:
            # Fall back to principal point × 2
            w_px = int(K[0, 2] * 2)
            h_px = int(K[1, 2] * 2)

        frame = {
            "file_path": f"images/{dst_name}",
            "transform_matrix": c2w.tolist(),
            "fl_x": float(K[0, 0]),
            "fl_y": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
            "w": w_px,
            "h": h_px,
        }
        frames.append(frame)

    # Build global intrinsic block (from camera 0; splatfacto also reads per-frame)
    K0: np.ndarray = intrinsics[0]
    try:
        from PIL import Image as _PILImage
        with _PILImage.open(image_paths[0]) as im:
            W0, H0 = im.size
    except Exception:
        W0 = int(K0[0, 2] * 2)
        H0 = int(K0[1, 2] * 2)

    transforms = {
        "fl_x": float(K0[0, 0]),
        "fl_y": float(K0[1, 1]),
        "cx": float(K0[0, 2]),
        "cy": float(K0[1, 2]),
        "w": W0,
        "h": H0,
        "camera_model": "OPENCV",
        "frames": frames,
    }

    transforms_path = out_dir / "transforms.json"
    with open(transforms_path, "w") as f:
        json.dump(transforms, f, indent=2)
    logger.info(f"Wrote {len(frames)}-frame transforms.json → {transforms_path}")

    # Copy sparse PLY (used by splatfacto for Gaussian seed initialisation)
    if sparse_ply and Path(sparse_ply).exists():
        dst_ply = out_dir / "sparse_pc.ply"
        shutil.copy2(sparse_ply, dst_ply)
        logger.info(f"Copied sparse PLY → {dst_ply}")
    else:
        logger.warning("sparse_ply not provided or missing; Gaussians will be randomly init.")

    return str(transforms_path)


# ---------------------------------------------------------------------------
# Windows-safe conda executable resolver
# ---------------------------------------------------------------------------

def _resolve_conda_exe() -> str:
    """Return the full path to the conda executable (handles Windows .bat).

    Search order:
    1. CONDA_EXE env var (set by conda activate).
    2. ``conda`` on PATH (works on Linux/macOS).
    3. Common Anaconda/Miniconda install locations on Windows.
    """
    import os
    import shutil as _shutil

    # 1. CONDA_EXE is set by conda itself when an env is active
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe and Path(conda_exe).exists():
        return conda_exe

    # 2. Plain 'conda' on PATH
    found = _shutil.which("conda")
    if found:
        return found

    # 3. Windows-specific search
    if sys.platform == "win32":
        home = Path.home()
        candidates = [
            home / "anaconda3" / "Scripts" / "conda.exe",
            home / "miniconda3" / "Scripts" / "conda.exe",
            home / "AppData" / "Local" / "anaconda3" / "Scripts" / "conda.exe",
            home / "AppData" / "Local" / "miniconda3" / "Scripts" / "conda.exe",
            Path("C:/ProgramData/anaconda3/Scripts/conda.exe"),
            Path("C:/ProgramData/miniconda3/Scripts/conda.exe"),
        ]
        for p in candidates:
            if p.exists():
                return str(p)

    raise FileNotFoundError(
        "Could not locate the conda executable. "
        "Ensure conda is installed and either CONDA_EXE is set or conda is on PATH."
    )


def _conda_env_exists(env_name: str) -> bool:
    """Return True if a conda env named ``env_name`` exists."""
    try:
        conda = _resolve_conda_exe()
        result = subprocess.run(
            [conda, "env", "list"],
            capture_output=True, text=True,
        )
        return any(
            part == env_name
            for line in result.stdout.splitlines()
            for part in line.split()
            if not line.startswith("#")
        )
    except Exception:
        return False


def _run_conda_cmd(args: List[str]) -> subprocess.CompletedProcess:
    """Run a conda command, streaming output line-by-line so it appears in Jupyter.

    ``subprocess.run(capture_output=False)`` sends child stdout/stderr to the
    *server* terminal, not the notebook cell.  Using ``Popen`` with piped streams
    and re-printing each line forces the output into the calling Python process's
    stdout, which Jupyter does capture.
    """
    import io
    import threading

    conda = _resolve_conda_exe()
    cmd = [conda] + args
    logger.info(f"Running: {' '.join(str(a) for a in cmd)}")

    captured_lines: List[str] = []

    def _drain(stream, tag: str) -> None:
        for raw in iter(stream.readline, b""):
            line = raw.decode("utf-8", errors="replace").rstrip()
            captured_lines.append(f"[{tag}] {line}")
            print(f"[{tag}] {line}", flush=True)
        stream.close()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    t_out = threading.Thread(target=_drain, args=(proc.stdout, "ns-train"))
    t_err = threading.Thread(target=_drain, args=(proc.stderr, "ns-train"))
    t_out.start(); t_err.start()
    proc.wait()
    t_out.join(); t_err.join()

    # Attach captured output to a fake CompletedProcess for error reporting
    result = subprocess.CompletedProcess(
        args=cmd,
        returncode=proc.returncode,
        stdout="\n".join(l for l in captured_lines if l.startswith("[ns-train]")),
        stderr="",
    )
    return result


# ---------------------------------------------------------------------------
# Splatfacto trainer (via subprocess into nerfstudio conda env)
# ---------------------------------------------------------------------------

def train_splatfacto(
    data_dir: str,
    out_dir: str,
    env_name: str = "nerfstudio",
    max_iters: int = 15000,
    use_sparse_init: bool = True,
    extra_args: Optional[List[str]] = None,
) -> Optional[str]:
    """Train a 3DGS scene with Nerfstudio's splatfacto.

    Spawns the training in the ``env_name`` conda env via ``conda run``.
    After training, exports the Gaussian PLY and returns its path.

    Args:
        data_dir:         Path to the folder containing ``transforms.json``
                          (e.g. ``data/reconstruction/3dgs/data``).
        out_dir:          Root output directory for Nerfstudio artefacts.
        env_name:         Conda environment containing Nerfstudio.
                          Pass ``None`` to skip training (bridge only).
        max_iters:        Training iteration count (15 000 ≈ 5–10 min on RTX 4060).
        use_sparse_init:  If ``sparse_pc.ply`` exists, seed Gaussians from it.
        extra_args:       Additional CLI args forwarded to ``ns-train``.

    Returns:
        Path to the exported Gaussian PLY, or ``None`` if training was skipped.
    """
    if env_name is None:
        logger.info("env_name is None — skipping splatfacto training.")
        return None

    # Pre-flight: check the target conda env actually exists
    if not _conda_env_exists(env_name):
        raise RuntimeError(
            f"Conda environment '{env_name}' does not exist.\n\n"
            "Create it with:\n"
            f"  conda create -n {env_name} python=3.10 -y\n"
            f"  conda activate {env_name}\n"
            "  pip install torch==2.1.2+cu118 torchvision --index-url https://download.pytorch.org/whl/cu118\n"
            "  pip install nerfstudio && ns-install-cli\n\n"
            "Alternatively, set  gaussian_splatting.env_name: null  in config.yaml\n"
            "to skip training and only generate the transforms.json bridge file."
        )

    # Pre-flight: check ns-train is callable inside the env
    conda = _resolve_conda_exe()
    _check = subprocess.run(
        [conda, "run", "--no-capture-output", "-n", env_name, "ns-train", "--help"],
        capture_output=True, text=True,
    )
    if _check.returncode != 0:
        raise RuntimeError(
            f"'ns-train' not found in conda env '{env_name}'.\n"
            "Install Nerfstudio with:\n"
            f"  conda activate {env_name}\n"
            "  pip install nerfstudio && ns-install-cli\n\n"
            f"stderr: {_check.stderr[:400]}"
        )

    data_dir = Path(data_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    conda_args = [
        "run", "-n", env_name,
        "ns-train", "splatfacto",
        "--data", str(data_dir),
        "--max-num-iterations", str(max_iters),
        "--output-dir", str(out_dir),
        "--vis", "tensorboard",
        # random-init False = use SFM / sparse_pc points (already the default,
        # but stated explicitly to be safe)
        "--pipeline.model.random-init", "False",
    ]

    # sparse_pc.ply is automatically discovered by the nerfstudio-data parser
    # when placed in the data_dir; no extra flag needed.

    if extra_args:
        conda_args += extra_args

    result = _run_conda_cmd(conda_args)

    if result.returncode != 0:
        tail = "\n".join(result.stdout.splitlines()[-30:]) if result.stdout else "(no output captured)"
        raise RuntimeError(
            f"splatfacto training failed (exit code {result.returncode}).\n"
            "Last 30 lines of output:\n"
            f"{tail}\n\n"
            "Common causes: invalid --data path, out of VRAM, or a missing "
            "nerfstudio dependency.  Check the [ns-train] lines printed above."
        )

    # Locate the config.yml written by Nerfstudio (newest run inside out_dir)
    config_yml = _find_nerfstudio_config(out_dir)
    if config_yml is None:
        logger.warning("Could not find Nerfstudio config.yml — skipping PLY export.")
        return None

    exported_ply = _export_gaussian_ply(config_yml, out_dir, env_name)
    return exported_ply


def _find_nerfstudio_config(out_dir: Path) -> Optional[Path]:
    """Return the most recently created config.yml under ``out_dir``."""
    candidates = sorted(out_dir.rglob("config.yml"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return None
    cfg = candidates[-1]
    logger.info(f"Using Nerfstudio config: {cfg}")
    return cfg


def _export_gaussian_ply(
    config_yml: Path,
    out_dir: Path,
    env_name: str,
) -> Optional[str]:
    """Run ``ns-export gaussian-splat`` and return the PLY path."""
    ply_out = out_dir / "gs_scene.ply"
    result = _run_conda_cmd([
        "run", "--no-capture-output", "-n", env_name,
        "ns-export", "gaussian-splat",
        "--load-config", str(config_yml),
        "--output-dir", str(out_dir),
    ])
    if result.returncode != 0:
        logger.warning("ns-export failed; trying to locate splat.ply in output tree.")

    # ns-export writes 'splat.ply' inside the output dir; rename to gs_scene.ply
    for candidate in out_dir.rglob("splat.ply"):
        shutil.move(str(candidate), str(ply_out))
        logger.info(f"Gaussian PLY exported → {ply_out}")
        return str(ply_out)

    # Some versions write 'point_cloud.ply'
    for candidate in out_dir.rglob("point_cloud.ply"):
        shutil.copy2(str(candidate), str(ply_out))
        logger.info(f"Gaussian PLY (point_cloud) copied → {ply_out}")
        return str(ply_out)

    logger.warning(f"Could not locate exported Gaussian PLY in {out_dir}")
    return None


# ---------------------------------------------------------------------------
# Floater removal / PLY cleanup
# ---------------------------------------------------------------------------

def clean_gaussian_ply(
    ply_path: str,
    out_path: Optional[str] = None,
    min_opacity: float = 0.05,
    max_scale: float = 0.5,
    outlier_nb_neighbors: int = 20,
    outlier_std_ratio: float = 2.0,
    camera_centers: Optional[np.ndarray] = None,
    bbox_margin: float = 1.5,
) -> str:
    """Remove floater Gaussians from a trained 3DGS PLY.

    Applies four filter passes in sequence:

    1. **Opacity threshold** — removes Gaussians whose sigmoid(opacity) is
       below ``min_opacity``.  Floaters tend to be semi-transparent.
    2. **Scale threshold** — removes Gaussians that have grown larger than
       ``max_scale`` world units on their longest axis.  Floaters often blow
       up to cover large regions of free space.
    3. **Statistical outlier removal** — runs Open3D
       ``remove_statistical_outlier`` on the surviving XYZ positions to
       eliminate isolated clusters far from the main surface density.
    4. **Bounding-box crop** (optional) — if ``camera_centers`` is provided,
       computes the axis-aligned bounding box of the camera rig, expands it
       by ``bbox_margin`` in each direction, and discards Gaussians outside
       that volume.

    The filtered PLY preserves the full binary layout (all Gaussian
    properties) so it remains valid for splatfacto rendering.

    Args:
        ply_path:            Path to the raw exported Gaussian PLY.
        out_path:            Output path.  Defaults to
                             ``<ply_path stem>_clean.ply`` next to the input.
        min_opacity:         Minimum sigmoid-opacity to keep (0–1).
        max_scale:           Maximum Gaussian extent in world units.
        outlier_nb_neighbors: Neighbourhood size for statistical outlier test.
        outlier_std_ratio:   Standard-deviation multiplier for outlier test.
        camera_centers:      (N, 3) array of camera world positions for the
                             optional bounding-box crop.
        bbox_margin:         Multiplicative expansion applied to the camera
                             hull before cropping.

    Returns:
        Path to the cleaned PLY file.
    """
    import struct

    ply_path = Path(ply_path)
    if out_path is None:
        out_path = str(ply_path.parent / (ply_path.stem + "_clean.ply"))

    # ------------------------------------------------------------------
    # 1. Read the full binary PLY
    # ------------------------------------------------------------------
    with open(ply_path, "rb") as fh:
        header_bytes: List[bytes] = []
        while True:
            line = fh.readline()
            header_bytes.append(line)
            if line.strip() == b"end_header":
                break
        body = fh.read()

    header_text = b"".join(header_bytes).decode("ascii", errors="ignore")

    # Parse element vertex count and property list
    num_verts = 0
    props: List[str] = []          # property names in order
    prop_types: List[str] = []     # corresponding C types
    _type_sizes = {"float": 4, "double": 8, "int": 4, "uint": 4,
                   "short": 2, "ushort": 2, "char": 1, "uchar": 1}

    for line in header_text.splitlines():
        line = line.strip()
        if line.startswith("element vertex"):
            num_verts = int(line.split()[-1])
        elif line.startswith("property "):
            parts = line.split()
            prop_types.append(parts[1])
            props.append(parts[2])

    is_binary_le = "format binary_little_endian" in header_text

    if not is_binary_le or not props:
        logger.warning(
            "clean_gaussian_ply: PLY is not binary-little-endian or has no "
            "properties — skipping cleanup."
        )
        return str(ply_path)

    # Build a numpy structured dtype for one row
    _np_map = {
        "float": np.float32, "double": np.float64,
        "int": np.int32, "uint": np.uint32,
        "short": np.int16, "ushort": np.uint16,
        "char": np.int8, "uchar": np.uint8,
    }
    dtype = np.dtype([(p, _np_map.get(t, np.float32))
                      for p, t in zip(props, prop_types)])
    stride = dtype.itemsize
    expected_bytes = num_verts * stride

    if len(body) < expected_bytes:
        logger.warning(
            f"clean_gaussian_ply: body has {len(body)} bytes but expected "
            f"{expected_bytes} — skipping cleanup."
        )
        return str(ply_path)

    data = np.frombuffer(body[:expected_bytes], dtype=dtype).copy()
    logger.info(f"Loaded {num_verts:,} Gaussians from {ply_path}")

    # ------------------------------------------------------------------
    # Helper: column index by name
    # ------------------------------------------------------------------
    def _col(name: str) -> Optional[int]:
        return props.index(name) if name in props else None

    # ------------------------------------------------------------------
    # Filter A — opacity threshold
    # ------------------------------------------------------------------
    opc_col = _col("opacity")
    if opc_col is not None:
        raw_opacity = data["opacity"].astype(np.float64)
        alpha = 1.0 / (1.0 + np.exp(-raw_opacity))   # sigmoid
        mask_opac = alpha >= min_opacity
        removed = int((~mask_opac).sum())
        data = data[mask_opac]
        logger.info(
            f"  opacity filter (alpha >= {min_opacity}): "
            f"removed {removed:,}, kept {len(data):,}"
        )
    else:
        logger.warning("  opacity property not found — skipping opacity filter")

    # ------------------------------------------------------------------
    # Filter B — scale threshold
    # ------------------------------------------------------------------
    scale_cols = [n for n in ("scale_0", "scale_1", "scale_2") if n in props]
    if scale_cols:
        scales = np.stack([data[n].astype(np.float64) for n in scale_cols], axis=1)
        max_axis_scale = np.exp(scales).max(axis=1)   # log-scale → world units
        mask_scale = max_axis_scale <= max_scale
        removed = int((~mask_scale).sum())
        data = data[mask_scale]
        logger.info(
            f"  scale filter (max_scale <= {max_scale}): "
            f"removed {removed:,}, kept {len(data):,}"
        )
    else:
        logger.warning("  scale_* properties not found — skipping scale filter")

    # ------------------------------------------------------------------
    # Filter C — bounding-box crop (camera rig hull)
    # ------------------------------------------------------------------
    if camera_centers is not None and len(camera_centers) > 0:
        centers = np.array(camera_centers, dtype=np.float64)
        c_min = centers.min(axis=0)
        c_max = centers.max(axis=0)
        # Expand by bbox_margin on each side
        half_ext = (c_max - c_min) / 2.0
        # Guarantee at least a small absolute margin so single-point clusters
        # don't collapse the box to zero size
        half_ext = np.maximum(half_ext, 0.1)
        mid = (c_min + c_max) / 2.0
        lo = mid - half_ext * bbox_margin
        hi = mid + half_ext * bbox_margin

        xyz = np.stack([data["x"].astype(np.float64),
                        data["y"].astype(np.float64),
                        data["z"].astype(np.float64)], axis=1)
        mask_bb = np.all((xyz >= lo) & (xyz <= hi), axis=1)
        removed = int((~mask_bb).sum())
        data = data[mask_bb]
        logger.info(
            f"  bbox crop (margin {bbox_margin}×): "
            f"removed {removed:,}, kept {len(data):,}"
        )

    # ------------------------------------------------------------------
    # Filter D — statistical outlier removal (Open3D on XYZ)
    # ------------------------------------------------------------------
    try:
        import open3d as o3d

        if len(data) > outlier_nb_neighbors:
            xyz = np.stack([data["x"].astype(np.float64),
                            data["y"].astype(np.float64),
                            data["z"].astype(np.float64)], axis=1)
            pcd_tmp = o3d.geometry.PointCloud()
            pcd_tmp.points = o3d.utility.Vector3dVector(xyz)
            _, inlier_idx = pcd_tmp.remove_statistical_outlier(
                nb_neighbors=outlier_nb_neighbors,
                std_ratio=outlier_std_ratio,
            )
            mask_sor = np.zeros(len(data), dtype=bool)
            mask_sor[inlier_idx] = True
            removed = int((~mask_sor).sum())
            data = data[mask_sor]
            logger.info(
                f"  statistical outlier removal "
                f"(nb={outlier_nb_neighbors}, std={outlier_std_ratio}): "
                f"removed {removed:,}, kept {len(data):,}"
            )
        else:
            logger.info("  skipping SOR: too few Gaussians remaining")
    except ImportError:
        logger.warning("  open3d not available — skipping statistical outlier removal")

    # ------------------------------------------------------------------
    # Rewrite PLY with filtered rows
    # ------------------------------------------------------------------
    # Rebuild header with updated vertex count
    new_header = header_text.encode("ascii")
    # Replace the original vertex count string
    old_vc_line = f"element vertex {num_verts}".encode("ascii")
    new_vc_line = f"element vertex {len(data)}".encode("ascii")
    new_header = new_header.replace(old_vc_line, new_vc_line, 1)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        fh.write(new_header)
        fh.write(data.tobytes())

    logger.info(
        f"Cleaned PLY written: {out_path} "
        f"({len(data):,} / {num_verts:,} Gaussians retained)"
    )
    return out_path


# ---------------------------------------------------------------------------
# PLY loader (for compositing + visualisation)
# ---------------------------------------------------------------------------

def load_gaussian_ply(path: str):
    """Load a Gaussian-splat PLY as an Open3D PointCloud.

    Reads XYZ + RGB from the PLY header.  Spherical-harmonic coefficients
    (``f_dc_*`` properties) are decoded as linear RGB when present; otherwise
    the reader falls back to the ``red``/``green``/``blue`` properties.

    Args:
        path: Path to the Gaussian PLY.

    Returns:
        ``open3d.geometry.PointCloud`` with points and colours set.
    """
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError("open3d is required: pip install open3d")

    pcd = o3d.io.read_point_cloud(path)
    if len(pcd.points) == 0:
        logger.warning(f"Loaded 0 points from {path} — trying manual header parse.")
        pcd = _load_gs_ply_manual(path)
    else:
        logger.info(f"Loaded Gaussian PLY: {len(pcd.points):,} points from {path}")

    return pcd


def _load_gs_ply_manual(path: str):
    """Parse a 3DGS PLY that Open3D may struggle with (f_dc_* SH props)."""
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError("open3d is required: pip install open3d")

    # Read raw bytes and extract xyz + f_dc (SH DC coefficient = mean colour)
    try:
        import struct

        with open(path, "rb") as fh:
            header_lines: List[bytes] = []
            while True:
                line = fh.readline()
                header_lines.append(line)
                if line.strip() == b"end_header":
                    break
            data = fh.read()

        # Parse property names
        props = []
        num_verts = 0
        binary_le = False
        for ln in header_lines:
            ln_s = ln.decode("ascii", errors="ignore").strip()
            if ln_s.startswith("element vertex"):
                num_verts = int(ln_s.split()[-1])
            if ln_s.startswith("format binary_little_endian"):
                binary_le = True
            if ln_s.startswith("property float"):
                props.append(ln_s.split()[-1])

        if not binary_le or not props:
            logger.warning("Manual PLY parse: unsupported format")
            return o3d.geometry.PointCloud()

        stride = len(props) * 4  # all float32
        xyz_idx = [props.index(n) for n in ("x", "y", "z") if n in props]
        sh_idx = [
            props.index(n) for n in ("f_dc_0", "f_dc_1", "f_dc_2")
            if n in props
        ]
        rgb_idx = [
            props.index(n) for n in ("red", "green", "blue")
            if n in props
        ]

        pts = np.zeros((num_verts, 3), dtype=np.float32)
        cols = np.ones((num_verts, 3), dtype=np.float32) * 0.5

        fmt = f"<{len(props)}f"
        for vi in range(num_verts):
            row = struct.unpack_from(fmt, data, vi * stride)
            if len(xyz_idx) == 3:
                pts[vi] = [row[xyz_idx[0]], row[xyz_idx[1]], row[xyz_idx[2]]]
            if len(sh_idx) == 3:
                # DC SH coefficient → linear colour (C0 = 0.28209479177387814)
                C0 = 0.28209479177387814
                r = np.clip(0.5 + C0 * row[sh_idx[0]], 0, 1)
                g = np.clip(0.5 + C0 * row[sh_idx[1]], 0, 1)
                b = np.clip(0.5 + C0 * row[sh_idx[2]], 0, 1)
                cols[vi] = [r, g, b]
            elif len(rgb_idx) == 3:
                cols[vi] = [row[rgb_idx[0]] / 255.0, row[rgb_idx[1]] / 255.0, row[rgb_idx[2]] / 255.0]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
        logger.info(f"Manually parsed {num_verts:,} Gaussians from {path}")
        return pcd

    except Exception as e:
        logger.error(f"Manual PLY parse failed: {e}")
        return o3d.geometry.PointCloud()


# ---------------------------------------------------------------------------
# Convenience wrapper called from pipeline.ipynb Stage 7
# ---------------------------------------------------------------------------

def run_gs_pipeline(
    camera_data: Optional[Dict],
    image_paths: List[str],
    sparse_ply: Optional[str],
    cfg: Dict,
) -> Optional[str]:
    """Full Stage 7 entry point: bridge → train → export.

    Args:
        camera_data:  DUSt3R camera_data from ``state['reconstruction_results']``.
                      May be ``None`` — the function will attempt to load it from
                      the JSON written by Stage 3 (``data/reconstruction/camera_data.json``).
        image_paths:  Ordered list of view image paths.
        sparse_ply:   Path to ``dust3r_points.ply``.
        cfg:          Full pipeline ``config.yaml`` dict (reads
                      ``cfg['gaussian_splatting']``).

    Returns:
        Path to ``gs_scene.ply``, or ``None`` if disabled / training failed.
    """
    gs_cfg = cfg.get("gaussian_splatting", {})
    if not gs_cfg.get("enabled", True):
        logger.info("gaussian_splatting.enabled=false — skipping Stage 7.")
        return None

    # If camera_data wasn't passed in-memory, try the JSON saved by Stage 3
    if camera_data is None:
        _cam_json_candidates = [
            "data/reconstruction/camera_data.json",
            "data/reconstruction/dust3r/camera_data.json",
        ]
        for _p in _cam_json_candidates:
            if Path(_p).exists():
                try:
                    from src.reconstruction import load_camera_data
                    camera_data = load_camera_data(_p)
                    logger.info(f"Loaded camera_data from disk: {_p}")
                    break
                except Exception as _e:
                    logger.warning(f"Could not load camera_data from {_p}: {_e}")

    if camera_data is None:
        raise RuntimeError(
            "camera_data is not available in memory or on disk.\n"
            "Run Stage 3 (3-D Reconstruction) at least once to generate:\n"
            "  data/reconstruction/camera_data.json"
        )

    data_dir = gs_cfg.get("data_dir", "data/reconstruction/3dgs/data")
    out_dir  = gs_cfg.get("output_dir", "data/reconstruction/3dgs")
    env_name = gs_cfg.get("env_name", "nerfstudio")
    max_iters = int(gs_cfg.get("max_iters", 15000))
    exported_ply_cfg = gs_cfg.get("exported_ply", "data/reconstruction/3dgs/gs_scene.ply")
    extra_args: Optional[List[str]] = gs_cfg.get("extra_args", None)
    fresh_start: bool = bool(gs_cfg.get("fresh_start", False))
    cleanup_cfg: Dict = gs_cfg.get("cleanup", {})

    # Optional: wipe previous splatfacto checkpoints so training always starts
    # from scratch rather than resuming a stale (possibly wrong-convention) run.
    if fresh_start:
        splatfacto_dir = Path(out_dir) / "splatfacto"
        if splatfacto_dir.exists():
            import shutil as _shutil
            _shutil.rmtree(splatfacto_dir)
            logger.info(f"fresh_start=true — removed old splatfacto dir: {splatfacto_dir}")

    # Step 1 — export transforms.json
    export_dust3r_to_nerfstudio(camera_data, image_paths, sparse_ply, data_dir)

    # Step 2 — train (returns None if env_name is None)
    exported_ply = train_splatfacto(
        data_dir=data_dir,
        out_dir=out_dir,
        env_name=env_name if env_name else None,
        max_iters=max_iters,
        extra_args=extra_args,
    )

    if exported_ply is None and Path(exported_ply_cfg).exists():
        exported_ply = exported_ply_cfg
        logger.info(f"Using cached gs_scene.ply: {exported_ply}")

    # Step 3 — optional floater cleanup
    if exported_ply and cleanup_cfg.get("enabled", True):
        cam_centers: Optional[np.ndarray] = None
        if camera_data and camera_data.get("camera_centers"):
            try:
                cam_centers = np.array(
                    list(camera_data["camera_centers"].values()), dtype=np.float64
                )
            except Exception:
                cam_centers = None

        clean_out = str(Path(exported_ply).parent / (Path(exported_ply).stem + "_clean.ply"))
        try:
            exported_ply = clean_gaussian_ply(
                ply_path=exported_ply,
                out_path=clean_out,
                min_opacity=float(cleanup_cfg.get("min_opacity", 0.05)),
                max_scale=float(cleanup_cfg.get("max_scale", 0.5)),
                outlier_nb_neighbors=int(cleanup_cfg.get("outlier_nb_neighbors", 20)),
                outlier_std_ratio=float(cleanup_cfg.get("outlier_std_ratio", 2.0)),
                camera_centers=cam_centers,
                bbox_margin=float(cleanup_cfg.get("bbox_margin", 1.5)),
            )
        except Exception as _ce:
            logger.warning(f"PLY cleanup failed (non-fatal): {_ce}")

    return exported_ply
