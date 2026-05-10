"""3D Reconstruction using VGGT (Visual Geometry Grounded Transformer).

CVPR 2025 Best Paper.  VGGT processes all images simultaneously in a single
feed-forward pass (< 1 second for 16 frames) and directly outputs:

  - Per-image extrinsic matrices  (3×4, camera-from-world, OpenCV convention)
  - Per-image intrinsic matrices  (3×3)
  - Per-image depth maps with per-pixel confidence scores
  - Dense 3-D point maps (unprojected from depth + cameras)

This replaces DUSt3R's pairwise + iterative global-alignment approach and
also solves the prior limitation where DUSt3R returned ``depth_maps: None``
in the results dict — VGGT natively provides per-image depth, which the
downstream spatial-audio step uses directly.

Installation
------------
pip install git+https://github.com/facebookresearch/vggt.git

Model weights are auto-downloaded from HuggingFace on first use:
  facebook/VGGT-1B           (non-commercial, default)
  facebook/VGGT-1B-Commercial (commercial license, same performance)
"""

import os
import logging
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_DEFAULT_MODEL    = "facebook/VGGT-1B"
_VGGT_RESOLUTION  = 518   # fixed internal inference resolution (hard-coded in VGGT)
_LOAD_RESOLUTION  = 1024  # images are loaded at this size before downscaling to 518


# ---------------------------------------------------------------------------
# Lazy imports — avoids hard failure when VGGT is not installed
# ---------------------------------------------------------------------------

def _lazy_imports():
    """Import VGGT modules; raise a helpful error if the package is missing."""
    try:
        import torch
        import torch.nn.functional as F
        from vggt.models.vggt import VGGT as _VGGT
        from vggt.utils.load_fn import load_and_preprocess_images_square
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        return (
            torch, F, _VGGT,
            load_and_preprocess_images_square,
            pose_encoding_to_extri_intri,
            unproject_depth_map_to_point_map,
        )
    except ImportError as exc:
        raise ImportError(
            f"VGGT is not installed ({exc}). "
            "Run:  pip install git+https://github.com/facebookresearch/vggt.git"
        ) from exc


# ---------------------------------------------------------------------------
# PLY helper (mirrors reconstruction_dust3r._write_ply)
# ---------------------------------------------------------------------------

def _write_ply(path: str, points: np.ndarray, colors: np.ndarray) -> str:
    """Write a coloured point cloud to an ASCII PLY file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if colors.dtype != np.uint8:
        colors = (np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8)
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
            f.write(
                f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} "
                f"{int(col[0])} {int(col[1])} {int(col[2])}\n"
            )
    return path


# ---------------------------------------------------------------------------
# Main reconstructor class
# ---------------------------------------------------------------------------

class VGGTReconstructor:
    """Run VGGT reconstruction on a set of images captured in the same scene.

    Typical use
    -----------
    rec = VGGTReconstructor()
    camera_data = rec.reconstruct(images)         # list of H×W×3 uint8 arrays
    ply_path    = rec.save_point_cloud("out.ply")
    depth_maps  = rec.depth_maps                  # List[np.ndarray (H, W)]
    """

    def __init__(
        self,
        model_name:      str   = _DEFAULT_MODEL,
        device:          str   = "cuda",
        output_dir:      str   = "data/reconstruction/vggt",
        conf_threshold:  float = 5.0,
        max_images:      int   = 24,
    ):
        """
        Args:
            model_name:     HuggingFace model id or local checkpoint path.
            device:         "cuda" — VGGT requires a GPU; CPU would take many hours.
            output_dir:     Directory for intermediate outputs.
            conf_threshold: Depth-confidence threshold for point-cloud filtering
                            (same as ``conf_thres_value`` in VGGT's demo_colmap.py).
            max_images:     Hard cap on the number of images passed to the forward
                            pass.  Excess frames are uniformly subsampled.  24 is
                            safe for a 12 GB card (3080 Ti); lower to 16 for 8 GB.
        """
        torch, *_ = _lazy_imports()

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available in this environment. VGGT-1B requires a GPU — "
                "running on CPU would take many hours (640+ minutes observed). "
                "Check that the Jupyter kernel is using the scene-audio conda env "
                "with PyTorch+CUDA installed."
            )
        self.device         = device
        self.model_name     = model_name
        self.output_dir     = output_dir
        self.conf_threshold = conf_threshold
        self.max_images     = max_images

        # bfloat16 on Ampere+ (CC ≥ 8.0), float16 on older GPUs, float32 on CPU
        if device == "cuda":
            cc = torch.cuda.get_device_capability()[0]
            self.dtype = torch.bfloat16 if cc >= 8 else torch.float16
        else:
            self.dtype = torch.float32
        logger.info(
            f"VGGTReconstructor: device={device}, dtype={self.dtype}, "
            f"model={model_name}, conf_threshold={conf_threshold}, "
            f"max_images={max_images}"
        )

        self._model:      Optional[object]           = None
        self._points3d:   Optional[np.ndarray]       = None
        self._colors:     Optional[np.ndarray]       = None
        self._depth_maps: Optional[List[np.ndarray]] = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Download and cache VGGT weights (runs once; subsequent calls are no-ops)."""
        if self._model is not None:
            return
        torch, _, _VGGT, *_ = _lazy_imports()
        print(f"[VGGT] Loading model '{self.model_name}' → target dtype: {self.dtype}")
        # VGGT is a plain nn.Module (not a HuggingFace PreTrainedModel), so
        # torch_dtype= is not supported. Use .to(dtype=) after loading instead.
        self._model = (
            _VGGT.from_pretrained(self.model_name)
            .to(dtype=self.dtype)
            .to(self.device)
        )
        self._model.eval()

        # Confirm actual dtype of the loaded weights — visible in notebook output.
        actual_dtype = next(self._model.parameters()).dtype
        print(f"[VGGT] Model weights dtype: {actual_dtype}  "
              f"({'OK — bfloat16 active' if actual_dtype == torch.bfloat16 else 'WARNING — not bfloat16!'})")
        if self.device == "cuda":
            allocated_gb = torch.cuda.memory_allocated() / 1e9
            free_gb      = torch.cuda.mem_get_info()[0] / 1e9
            total_gb     = torch.cuda.mem_get_info()[1] / 1e9
            print(f"[VGGT] VRAM after model load: {allocated_gb:.2f} GB allocated  "
                  f"| {free_gb:.1f} GB free / {total_gb:.1f} GB total")

    # ------------------------------------------------------------------
    # Core reconstruction
    # ------------------------------------------------------------------

    def reconstruct(self, images: List[np.ndarray]) -> Dict:
        """Run VGGT on all images in a single forward pass.

        Args:
            images: List of H×W×3 uint8 numpy arrays (all views of one scene).

        Returns:
            ``camera_data`` dict with keys:
              intrinsics, extrinsics, camera_centers, image_names, vggt=True.
            Also populates ``self._points3d``, ``self._colors``, and
            ``self._depth_maps`` for point-cloud export and downstream use.
        """
        self.load_model()
        torch, F, _, load_and_preprocess_images_square, \
            pose_encoding_to_extri_intri, unproject_depth_map_to_point_map = _lazy_imports()

        # ── 0. Enforce max_images cap via uniform subsampling ─────────────────
        original_count = len(images)
        if original_count > self.max_images:
            indices = np.linspace(0, original_count - 1, self.max_images, dtype=int)
            images = [images[i] for i in indices]
            logger.warning(
                f"Subsampled from {original_count} → {self.max_images} images "
                f"(vggt_max_images cap). Lower vggt_max_images for less VRAM."
            )

        n = len(images)

        # ── 0b. Normalise all images to the same (H, W) ───────────────────────
        # Input images from the user's camera may be at a different resolution
        # than the generated views (e.g. 2880×2160 vs 1440×1080).  VGGT down-
        # scales everything to 518×518 internally, but orig_shapes is used later
        # for intrinsic rescaling and depth-map resizing — mixed sizes would
        # produce inconsistent camera models.  Resize to the most common size.
        from collections import Counter as _Counter
        shape_counts = _Counter(img.shape[:2] for img in images)  # (H, W) counts
        target_hw = shape_counts.most_common(1)[0][0]             # majority shape
        target_h, target_w = target_hw
        resized_any = False
        normalised_images = []
        for img in images:
            h, w = img.shape[:2]
            if (h, w) != (target_h, target_w):
                pil = Image.fromarray(img if img.dtype == np.uint8
                                      else (img * 255).astype(np.uint8))
                pil = pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
                normalised_images.append(np.array(pil))
                resized_any = True
            else:
                normalised_images.append(img)
        if resized_any:
            print(f"[VGGT] Normalised input images → {target_w}×{target_h} "
                  f"(majority size; {shape_counts.most_common(1)[0][1]}/{n} already matched)")
        images = normalised_images

        if self.device == "cuda":
            free_gb  = torch.cuda.mem_get_info()[0] / 1e9
            total_gb = torch.cuda.mem_get_info()[1] / 1e9
            print(f"[VGGT] Forward pass: {n} images — "
                  f"VRAM {free_gb:.1f} GB free / {total_gb:.1f} GB total before inference")
        else:
            print(f"[VGGT] Forward pass: {n} images (CPU)")

        # ── 1. Save numpy arrays to a temp dir so VGGT's loader can read them ─
        orig_shapes: List[Tuple[int, int]] = []  # (H, W) per image

        with tempfile.TemporaryDirectory() as tmp_dir:
            image_paths = []
            for i, img_arr in enumerate(images):
                img_u8 = (
                    img_arr if img_arr.dtype == np.uint8
                    else (img_arr * 255).astype(np.uint8)
                )
                orig_shapes.append(img_u8.shape[:2])  # (H, W)
                path = os.path.join(tmp_dir, f"img_{i:03d}.jpg")
                Image.fromarray(img_u8).save(path, format="JPEG", quality=95)
                image_paths.append(path)

            # Load images at 1024×1024 (VGGT convention); original_coords carries
            # the crop/scale info needed to map back to original pixel coordinates.
            vggt_tensors, original_coords = load_and_preprocess_images_square(
                image_paths, _LOAD_RESOLUTION
            )
        # tmp_dir cleaned up; tensors are already in memory

        vggt_tensors  = vggt_tensors.to(self.device)   # (N, 3, 1024, 1024)

        # ── 2. Downscale to VGGT's fixed 518×518 inference resolution ─────────
        imgs_518 = F.interpolate(
            vggt_tensors,
            size=(_VGGT_RESOLUTION, _VGGT_RESOLUTION),
            mode="bilinear",
            align_corners=False,
        )  # (N, 3, 518, 518)

        # ── 3. Forward pass ───────────────────────────────────────────────────
        with torch.no_grad():
            with torch.amp.autocast(device_type=self.device, dtype=self.dtype):
                imgs_batch = imgs_518[None]  # (1, N, 3, 518, 518)

                aggregated_tokens_list, ps_idx = self._model.aggregator(imgs_batch)

                # Camera prediction — extrinsic follows OpenCV convention
                # (camera-from-world, i.e. X_cam = R @ X_world + t)
                pose_enc = self._model.camera_head(aggregated_tokens_list)[-1]
                extrinsic, intrinsic = pose_encoding_to_extri_intri(
                    pose_enc, imgs_batch.shape[-2:]
                )

                # Depth prediction
                depth_map, depth_conf = self._model.depth_head(
                    aggregated_tokens_list, imgs_batch, ps_idx
                )

        # ── 4. CPU numpy ──────────────────────────────────────────────────────
        extrinsic  = extrinsic.squeeze(0).cpu().numpy()   # (N, 3, 4)
        intrinsic  = intrinsic.squeeze(0).cpu().numpy()   # (N, 3, 3) at 518×518
        depth_map  = depth_map.squeeze(0).cpu().numpy()   # (N, 518, 518) or (N, 518, 518, 1)
        depth_conf = depth_conf.squeeze(0).cpu().numpy()  # (N, 518, 518) or (N, 518, 518, 1)
        imgs_np    = imgs_518.cpu().numpy().transpose(0, 2, 3, 1)  # (N, 518, 518, 3)

        # unproject_depth_map_to_point_map internally calls .squeeze(-1) on each
        # frame, so it requires shape (N, H, W, 1).  Ensure that trailing dim
        # exists regardless of whether the depth head returned it or not.
        depth_map_4d = depth_map if depth_map.ndim == 4 else depth_map[..., np.newaxis]

        # 2D form (N, H, W) — used for confidence masking and per-image resize.
        depth_map_2d  = depth_map_4d[..., 0]
        depth_conf_2d = depth_conf[..., 0] if depth_conf.ndim == 4 else depth_conf

        # ── 5. Dense 3-D point cloud (confidence-filtered) ───────────────────
        points3d_map = unproject_depth_map_to_point_map(depth_map_4d, extrinsic, intrinsic)
        # → (N, 518, 518, 3)

        # Diagnostic: show confidence score distribution so the threshold can be tuned.
        print(f"[VGGT] Depth confidence stats — "
              f"min: {depth_conf_2d.min():.4f}  "
              f"mean: {depth_conf_2d.mean():.4f}  "
              f"max: {depth_conf_2d.max():.4f}  "
              f"(threshold: {self.conf_threshold})")

        conf_mask       = depth_conf_2d >= self.conf_threshold
        kept_pct        = conf_mask.mean() * 100
        print(f"[VGGT] Points passing threshold: {conf_mask.sum():,} / "
              f"{conf_mask.size:,} ({kept_pct:.1f}%)")

        self._points3d  = points3d_map[conf_mask]  # (M, 3)
        self._colors    = imgs_np[conf_mask]        # (M, 3) float [0, 1]

        # ── 6. Build camera_data matching the downstream schema ───────────────
        camera_data: Dict = {
            "intrinsics":     {},
            "extrinsics":     {},
            "image_names":    {},
            "camera_centers": {},
            "vggt":           True,
        }

        self._depth_maps = []

        for i in range(n):
            orig_h, orig_w = orig_shapes[i]

            # Rescale intrinsics from inference resolution to original image size
            K = intrinsic[i].copy()  # 3×3
            K[0, :] *= orig_w / _VGGT_RESOLUTION  # fx, cx
            K[1, :] *= orig_h / _VGGT_RESOLUTION  # fy, cy

            # Pad 3×4 extrinsic to 4×4 (world-to-cam homogeneous)
            ext_4x4 = np.eye(4, dtype=np.float64)
            ext_4x4[:3, :] = extrinsic[i]

            R = ext_4x4[:3, :3]
            t = ext_4x4[:3,  3]
            cam_center = -R.T @ t  # world-space camera position

            camera_data["intrinsics"][i]     = K
            camera_data["extrinsics"][i]     = ext_4x4
            camera_data["image_names"][i]    = f"image_{i:03d}"
            camera_data["camera_centers"][i] = cam_center

            # Resize depth map from 518×518 to original image dimensions
            dm_tensor = (
                torch.from_numpy(depth_map_2d[i])
                .unsqueeze(0).unsqueeze(0)
                .float()
            )
            dm_resized = F.interpolate(
                dm_tensor,
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze().numpy()
            self._depth_maps.append(dm_resized)

        logger.info(
            f"VGGT reconstruction complete: {n} cameras, "
            f"{len(self._points3d):,} 3D points"
        )
        return camera_data

    # ------------------------------------------------------------------
    # Point-cloud export
    # ------------------------------------------------------------------

    def save_point_cloud(self, path: str) -> Optional[str]:
        """Write the confident point cloud to a PLY file.

        Returns the path on success, or None if no point cloud is available.
        """
        if self._points3d is None or len(self._points3d) == 0:
            logger.warning("No point cloud data available; skipping PLY export.")
            return None
        _write_ply(path, self._points3d, self._colors)
        logger.info(f"Saved point cloud ({len(self._points3d):,} pts) → {path}")
        return path

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def depth_maps(self) -> Optional[List[np.ndarray]]:
        """Per-image depth maps at original image resolution, or None."""
        return self._depth_maps


# ---------------------------------------------------------------------------
# Convenience wrapper (mirrors reconstruct_with_dust3r)
# ---------------------------------------------------------------------------

def reconstruct_with_vggt(
    images:     List[np.ndarray],
    config:     Dict = None,
    output_dir: str  = "data/reconstruction",
) -> Dict:
    """Convenience wrapper called by ``reconstruct_scene()``.

    Args:
        images:     List of H×W×3 uint8 numpy arrays.
        config:     Pipeline config dict (reconstruction section).
        output_dir: Root reconstruction directory.

    Returns:
        Dict with keys:
          camera_data        — intrinsics / extrinsics / camera_centers / image_names
          dense_point_cloud  — path to PLY, or None
          sparse_reconstruction — None (VGGT does not produce a COLMAP sparse model)
          depth_maps         — List[np.ndarray (H, W)], one per input image
    """
    if config is None:
        config = {}

    vggt_dir = os.path.join(output_dir, "vggt")
    os.makedirs(vggt_dir, exist_ok=True)

    rec = VGGTReconstructor(
        model_name     = config.get("vggt_model",          _DEFAULT_MODEL),
        device         = "cuda",
        output_dir     = vggt_dir,
        conf_threshold = config.get("vggt_conf_threshold", 5.0),
        max_images     = config.get("vggt_max_images",     24),
    )

    camera_data = rec.reconstruct(images)

    ply_path = rec.save_point_cloud(
        os.path.join(vggt_dir, "vggt_points.ply")
    )

    return {
        "camera_data":           camera_data,
        "dense_point_cloud":     ply_path,
        "sparse_reconstruction": None,   # VGGT feed-forward; no COLMAP sparse model
        "depth_maps":            rec.depth_maps,
    }
