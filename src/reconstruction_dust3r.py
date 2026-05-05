"""DUSt3R-based 3D reconstruction.

Replaces COLMAP for scenes containing AI-generated images, which lack the
photometric consistency required by SIFT feature matching.  DUSt3R uses a
transformer to directly predict dense 3D point maps and relative camera poses
from image pairs, requiring no handcrafted feature detector.

The public API mirrors COLMAPReconstructor.get_camera_data() so the rest of
the pipeline (project_objects_to_3d, spatial audio) can consume either output
without changes.

Reference: https://github.com/naver/dust3r
License:   CC BY-NC-SA 4.0 (non-commercial use only)
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path setup — DUSt3R lives in <project_root>/dust3r/
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DUST3R_ROOT  = _PROJECT_ROOT / "dust3r"

if str(_DUST3R_ROOT) not in sys.path:
    sys.path.insert(0, str(_DUST3R_ROOT))

# HuggingFace model id for automatic weight download
_DEFAULT_MODEL = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"


def _lazy_imports():
    """Import DUSt3R modules lazily to avoid slowing down the whole pipeline."""
    try:
        from dust3r.inference import inference
        from dust3r.model import AsymmetricCroCo3DStereo
        from dust3r.utils.image import load_images
        from dust3r.image_pairs import make_pairs
        from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
        return inference, AsymmetricCroCo3DStereo, load_images, make_pairs, global_aligner, GlobalAlignerMode
    except ImportError as e:
        raise ImportError(
            f"DUSt3R not available: {e}. "
            "Make sure the dust3r submodule is cloned: "
            "git clone --recursive https://github.com/naver/dust3r.git"
        ) from e


class DUSt3RReconstructor:
    """Estimate camera poses and a dense point cloud using DUSt3R.

    Usage
    -----
    rec = DUSt3RReconstructor()
    rec.load_model()                        # downloads weights on first run
    camera_data = rec.reconstruct(images)   # list of H×W×3 uint8 numpy arrays
    ply_path    = rec.save_point_cloud('data/reconstruction/dust3r/points.ply')
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: str = "cuda",
        output_dir: str = "data/reconstruction/dust3r",
        image_size: int = 512,
        niter: int = 300,
        schedule: str = "cosine",
        lr: float = 0.01,
        min_conf_thr: float = 1.5,
    ):
        """
        Args:
            model_name:    HuggingFace model id or local checkpoint path.
            device:        "cuda" or "cpu".
            output_dir:    Where to save the point cloud and camera JSON.
            image_size:    Resolution fed to DUSt3R (224 or 512).
            niter:         Global alignment iterations (more = better, slower).
            schedule:      LR schedule for global alignment ("cosine" or "linear").
            lr:            Learning rate for global alignment.
            min_conf_thr:  Confidence threshold for filtering point cloud.
        """
        import torch
        self.model_name   = model_name
        self.device       = device if torch.cuda.is_available() else "cpu"
        self.output_dir   = output_dir
        self.image_size   = image_size
        self.niter        = niter
        self.schedule     = schedule
        self.lr           = lr
        self.min_conf_thr = min_conf_thr
        self._model       = None
        self._scene       = None   # last global alignment result

        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load (or download) the DUSt3R model weights."""
        if self._model is not None:
            return
        _, AsymmetricCroCo3DStereo, *_ = _lazy_imports()
        logger.info(f"Loading DUSt3R model: {self.model_name} on {self.device}")
        self._model = (
            AsymmetricCroCo3DStereo.from_pretrained(self.model_name)
            .to(self.device)
        )
        self._model.eval()
        logger.info("DUSt3R model loaded")

    # ------------------------------------------------------------------
    # Core reconstruction
    # ------------------------------------------------------------------

    def reconstruct(
        self,
        images: List[np.ndarray],
        batch_size: int = 1,
    ) -> Dict:
        """Run DUSt3R on a list of images and return camera_data.

        The returned dict has the same keys as COLMAPReconstructor.get_camera_data():
            intrinsics      {idx: 3×3 K matrix}
            extrinsics      {idx: 4×4 [R|t] camera-from-world}
            image_names     {idx: str}
            camera_centers  {idx: 3-vec world-space camera centre}
            dust3r          True   (flag for downstream code)

        It also stores pts3d and confidence internally for save_point_cloud().

        Args:
            images:     List of H×W×3 uint8 numpy arrays.
            batch_size: Inference batch size (reduce to 1 if VRAM is tight).

        Returns:
            camera_data dict compatible with project_objects_to_3d().
        """
        import torch
        inference, _, load_images, make_pairs, global_aligner, GlobalAlignerMode = _lazy_imports()

        self.load_model()

        # --- Save images to temp JPEG files so dust3r.utils.image.load_images
        #     can read them (it expects file paths, not numpy arrays) ----------
        tmp_dir = os.path.join(self.output_dir, "_tmp_imgs")
        os.makedirs(tmp_dir, exist_ok=True)
        img_paths = []
        for i, img in enumerate(images):
            p = os.path.join(tmp_dir, f"img_{i:03d}.jpg")
            img_u8 = img if img.dtype == np.uint8 else (img * 255).astype(np.uint8)
            Image.fromarray(img_u8).save(p, format="JPEG", quality=95)
            img_paths.append(p)

        logger.info(f"Running DUSt3R inference on {len(img_paths)} images")

        dust3r_imgs = load_images(img_paths, size=self.image_size)

        # Complete graph of image pairs (symmetrised)
        pairs = make_pairs(
            dust3r_imgs,
            scene_graph="complete",
            prefilter=None,
            symmetrize=True,
        )
        logger.info(f"Created {len(pairs)} image pairs for inference")

        output = inference(pairs, self._model, self.device, batch_size=batch_size)

        # Global alignment
        mode = (
            GlobalAlignerMode.PairViewer
            if len(images) == 2
            else GlobalAlignerMode.PointCloudOptimizer
        )
        scene = global_aligner(output, device=self.device, mode=mode)

        if mode == GlobalAlignerMode.PointCloudOptimizer:
            logger.info(
                f"Running global alignment: niter={self.niter}, "
                f"schedule={self.schedule}, lr={self.lr}"
            )
            scene.compute_global_alignment(
                init="mst",
                niter=self.niter,
                schedule=self.schedule,
                lr=self.lr,
            )

        self._scene = scene

        # --- Extract camera parameters ------------------------------------
        focals    = scene.get_focals()          # (N,) tensor
        poses     = scene.get_im_poses()        # (N, 4, 4) cam-to-world
        pts3d     = scene.get_pts3d()           # list of (H, W, 3) tensors
        conf_masks = scene.get_masks()          # list of (H, W) bool tensors
        imgs       = scene.imgs                 # list of (H, W, 3) float32

        camera_data: Dict = {
            "intrinsics":     {},
            "extrinsics":     {},
            "image_names":    {},
            "camera_centers": {},
            "dust3r":         True,
        }

        N = len(images)
        for i in range(N):
            h, w = images[i].shape[:2]
            f    = float(focals[i].item())
            cx, cy = w / 2.0, h / 2.0

            K = np.array([
                [f,  0, cx],
                [0,  f, cy],
                [0,  0,  1],
            ], dtype=np.float64)

            # DUSt3R gives cam-to-world; we need world-to-cam (extrinsic)
            c2w = poses[i].detach().cpu().numpy()   # 4×4
            w2c = np.linalg.inv(c2w)                # world-to-cam

            cam_centre = c2w[:3, 3]                 # world-space camera position

            camera_data["intrinsics"][i]     = K
            camera_data["extrinsics"][i]     = w2c
            camera_data["image_names"][i]    = f"image_{i:03d}"
            camera_data["camera_centers"][i] = cam_centre

        # Store for point cloud export
        self._pts3d      = pts3d
        self._conf_masks = conf_masks
        self._imgs_float = imgs
        self._N          = N

        logger.info(
            f"DUSt3R reconstruction complete: "
            f"{N} cameras, confidence threshold={self.min_conf_thr}"
        )
        return camera_data

    # ------------------------------------------------------------------
    # Point cloud export
    # ------------------------------------------------------------------

    def save_point_cloud(self, output_path: Optional[str] = None) -> str:
        """Save the dense coloured point cloud as a PLY file.

        Args:
            output_path: Full path for the PLY. Defaults to
                         <output_dir>/dust3r_points.ply

        Returns:
            Path to the saved PLY file.
        """
        if self._scene is None:
            raise RuntimeError("Call reconstruct() before save_point_cloud()")

        if output_path is None:
            output_path = os.path.join(self.output_dir, "dust3r_points.ply")

        import torch

        all_pts   = []
        all_cols  = []
        conf_masks = self._conf_masks
        pts3d      = self._pts3d
        imgs_float = self._imgs_float

        for i in range(self._N):
            mask = conf_masks[i].cpu().numpy() if hasattr(conf_masks[i], "cpu") else conf_masks[i]
            pts  = pts3d[i].detach().cpu().numpy() if hasattr(pts3d[i], "detach") else pts3d[i]
            col  = imgs_float[i]

            pts_sel  = pts[mask]
            col_sel  = (col[mask] * 255).clip(0, 255).astype(np.uint8)

            all_pts.append(pts_sel)
            all_cols.append(col_sel)

        all_pts  = np.concatenate(all_pts,  axis=0)
        all_cols = np.concatenate(all_cols, axis=0)

        _write_ply(output_path, all_pts, all_cols)
        logger.info(
            f"Point cloud saved: {output_path} "
            f"({len(all_pts):,} points)"
        )
        return output_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_ply(path: str, points: np.ndarray, colors: np.ndarray) -> None:
    """Write an ASCII PLY point cloud."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n = len(points)
    with open(path, "w") as f:
        f.write(
            f"ply\nformat ascii 1.0\n"
            f"element vertex {n}\n"
            f"property float x\nproperty float y\nproperty float z\n"
            f"property uchar red\nproperty uchar green\nproperty uchar blue\n"
            f"end_header\n"
        )
        for pt, col in zip(points, colors):
            f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} "
                    f"{col[0]} {col[1]} {col[2]}\n")


def reconstruct_with_dust3r(
    images: List[np.ndarray],
    config: Dict = None,
    output_dir: str = "data/reconstruction",
) -> Dict:
    """Convenience wrapper called by reconstruct_scene().

    Args:
        images:     List of H×W×3 uint8 numpy arrays.
        config:     Pipeline config dict (reconstruction section).
        output_dir: Root reconstruction directory.

    Returns:
        results dict with keys: camera_data, dense_point_cloud, depth_maps (None).
    """
    if config is None:
        config = {}

    dust3r_dir = os.path.join(output_dir, "dust3r")
    rec = DUSt3RReconstructor(
        model_name   = config.get("dust3r_model",    _DEFAULT_MODEL),
        device       = "cuda",
        output_dir   = dust3r_dir,
        image_size   = config.get("dust3r_img_size", 512),
        niter        = config.get("dust3r_niter",    300),
        min_conf_thr = config.get("dust3r_min_conf", 1.5),
    )

    camera_data = rec.reconstruct(images)

    ply_path = rec.save_point_cloud(
        os.path.join(dust3r_dir, "dust3r_points.ply")
    )

    return {
        "camera_data":      camera_data,
        "dense_point_cloud": ply_path,
        "sparse_reconstruction": None,  # not applicable for DUSt3R
        "depth_maps":        None,      # MiDaS runs separately
    }
