"""3D Reconstruction Module using COLMAP and Depth Estimation"""

import os
import logging
from typing import List, Tuple, Dict, Optional
import numpy as np
from PIL import Image
import shutil
from pathlib import Path

try:
    import pycolmap
except ImportError:
    raise ImportError("Please install pycolmap: pip install pycolmap")

try:
    import torch
except ImportError:
    raise ImportError("Please install torch: pip install torch")

try:
    import open3d as o3d
except ImportError:
    raise ImportError("Please install open3d: pip install open3d")

logger = logging.getLogger(__name__)


class DepthEstimator:
    """Depth estimation using MiDaS or Depth-Anything models"""
    
    def __init__(self, model_name: str = "DPT_Large", device: str = "cuda"):
        """
        Initialize depth estimator
        
        Args:
            model_name: Model to use. Current MiDaS hub names:
                        "DPT_Large" (best quality), "DPT_Hybrid", "MiDaS", "MiDaS_small"
            device: Device to run on (cuda, cpu, mps)
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        logger.info(f"Loading depth estimation model: {model_name} on {self.device}")
        
        # Load MiDaS model from torch hub.
        # The GitHub org moved from intel-isl → isl-org; force_reload=True
        # discards any stale cached clone of the old repo.
        _MIDAS_REPO = "isl-org/MiDaS"
        try:
            self.model = torch.hub.load(
                _MIDAS_REPO, model_name, force_reload=True
            )
            self.model.to(self.device)
            self.model.eval()

            # Load transforms — case-insensitive check for DPT variants
            midas_transforms = torch.hub.load(_MIDAS_REPO, "transforms")
            if "dpt" in model_name.lower():
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.small_transform

            logger.info("Depth estimation model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading depth model: {str(e)}")
            raise
    
    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth map for a single image
        
        Args:
            image: Input image as numpy array (H, W, 3)
        
        Returns:
            Depth map as numpy array (H, W)
        """
        # MiDaS transforms expect a numpy RGB array, not a PIL Image
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        img_np = np.array(image)  # ensure it's a plain numpy array

        # Apply transforms
        input_batch = self.transform(img_np).to(self.device)
        
        # Predict depth
        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        
        return depth_map
    
    def estimate_depth_batch(
        self,
        images: List[np.ndarray],
        output_dir: str = "data/reconstruction/depth"
    ) -> List[np.ndarray]:
        """
        Estimate depth maps for multiple images
        
        Args:
            images: List of input images
            output_dir: Directory to save depth maps
        
        Returns:
            List of depth maps
        """
        logger.info(f"Estimating depth for {len(images)} images")
        os.makedirs(output_dir, exist_ok=True)
        
        depth_maps = []
        for i, image in enumerate(images):
            depth_map = self.estimate_depth(image)
            depth_maps.append(depth_map)
            
            # Save depth map visualization
            depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
            depth_colored = (depth_normalized * 255).astype(np.uint8)
            depth_img = Image.fromarray(depth_colored, mode='L')
            depth_img.save(os.path.join(output_dir, f"depth_{i:03d}.png"))
            
            # Save raw depth as numpy
            np.save(os.path.join(output_dir, f"depth_{i:03d}.npy"), depth_map)
            
            logger.info(f"Saved depth map {i}: shape {depth_map.shape}, range [{depth_map.min():.2f}, {depth_map.max():.2f}]")
        
        return depth_maps


class COLMAPReconstructor:
    """3D reconstruction using COLMAP"""
    
    def __init__(self, output_dir: str = "data/reconstruction/colmap"):
        """
        Initialize COLMAP reconstructor
        
        Args:
            output_dir: Directory to store COLMAP outputs
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.database_path = os.path.join(output_dir, "database.db")
        self.image_dir = os.path.join(output_dir, "images")
        self.sparse_dir = os.path.join(output_dir, "sparse")
        self.dense_dir = os.path.join(output_dir, "dense")
        
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.sparse_dir, exist_ok=True)
        os.makedirs(self.dense_dir, exist_ok=True)
        
        logger.info(f"Initialized COLMAP reconstructor at: {output_dir}")
    
    def prepare_images(self, images: List[np.ndarray], image_paths: Optional[List[str]] = None) -> List[str]:
        """
        Prepare images for COLMAP by copying to working directory
        
        Args:
            images: List of images as numpy arrays
            image_paths: Optional list of original image paths for naming
        
        Returns:
            List of paths to prepared images
        """
        prepared_paths = []
        
        for i, image in enumerate(images):
            if image_paths and i < len(image_paths):
                stem = Path(image_paths[i]).stem
            else:
                stem = f"image_{i:03d}"
            # Always write JPEG — COLMAP is more reliable with JPEG than PNG
            filename = f"{stem}.jpg"

            output_path = os.path.join(self.image_dir, filename)

            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)

            Image.fromarray(image).save(output_path, format="JPEG", quality=95)
            prepared_paths.append(output_path)
        
        logger.info(f"Prepared {len(prepared_paths)} images for COLMAP")
        return prepared_paths
    
    def run_feature_extraction(self, feature_type: str = "SIFT") -> None:
        """
        Extract features from images
        
        Args:
            feature_type: Type of features to extract (SIFT, AKAZE, etc.)
        """
        logger.info(f"Extracting {feature_type} features")
        
        # pycolmap 4.x: FeatureExtractionOptions no longer exposes individual
        # SIFT params at the top level — pass with defaults (8192 features).
        pycolmap.extract_features(
            database_path=self.database_path,
            image_path=self.image_dir,
            extraction_options=pycolmap.FeatureExtractionOptions(),
        )
        
        logger.info("Feature extraction complete")
    
    def run_feature_matching(self, matching_method: str = "exhaustive") -> None:
        """
        Match features between images
        
        Args:
            matching_method: Matching method (exhaustive, sequential, vocab_tree)
        """
        logger.info(f"Matching features using {matching_method} method")

        # pycolmap 4.x renamed SiftMatchingOptions → FeatureMatchingOptions.
        if matching_method == "exhaustive":
            pycolmap.match_exhaustive(
                database_path=self.database_path,
                matching_options=pycolmap.FeatureMatchingOptions(),
            )
        else:
            logger.warning(f"Matching method {matching_method} not implemented, using exhaustive")
            pycolmap.match_exhaustive(database_path=self.database_path)
        
        logger.info("Feature matching complete")
    
    def run_sparse_reconstruction(self) -> pycolmap.Reconstruction:
        """
        Run sparse reconstruction (Structure from Motion)
        
        Returns:
            Sparse reconstruction object
        """
        logger.info("Running sparse reconstruction")
        
        # Create output directory for this reconstruction
        output_path = os.path.join(self.sparse_dir, "0")
        os.makedirs(output_path, exist_ok=True)
        
        # pycolmap 4.x moved per-image inlier/error thresholds into nested
        # sub-objects (mapper, triangulation). Only top-level attributes that
        # still exist in IncrementalPipelineOptions are used here; defaults
        # are suitable for a 10-image indoor scene.
        maps = pycolmap.incremental_mapping(
            database_path=self.database_path,
            image_path=self.image_dir,
            output_path=output_path,
            options=pycolmap.IncrementalPipelineOptions(
                min_num_matches=15,
            ),
        )
        
        if not maps:
            logger.error("Sparse reconstruction failed")
            raise RuntimeError("COLMAP sparse reconstruction failed")
        
        # Get the largest reconstruction
        reconstruction = maps[0]
        logger.info(f"Sparse reconstruction complete: {len(reconstruction.images)} images, "
                   f"{len(reconstruction.points3D)} 3D points")
        
        # Write reconstruction to disk
        reconstruction.write(output_path)
        
        return reconstruction
    
    def run_dense_reconstruction(self, max_image_size: int = 2000) -> str:
        """
        Run dense reconstruction (Multi-View Stereo) via COLMAP CLI.

        pycolmap's Python bindings do not expose patch_match_stereo or
        stereo_fusion directly, so we shell out to the COLMAP binary which
        must be on PATH.  The undistort step IS available via pycolmap.

        Args:
            max_image_size: Maximum image dimension for dense reconstruction

        Returns:
            Path to dense point cloud (PLY file)
        """
        import subprocess

        logger.info("Running dense reconstruction (this may take a while)")

        sparse_model_path = os.path.join(self.sparse_dir, "0")

        # Undistort images (available in pycolmap Python bindings)
        logger.info("Undistorting images")
        pycolmap.undistort_images(
            output_path=self.dense_dir,
            input_path=sparse_model_path,
            image_path=self.image_dir,
            max_image_size=max_image_size,
        )

        # Patch-match stereo (requires COLMAP binary)
        logger.info("Running patch match stereo (COLMAP binary)")
        subprocess.run(
            [
                "colmap", "patch_match_stereo",
                "--workspace_path", self.dense_dir,
                "--workspace_format", "COLMAP",
                "--PatchMatchStereo.geom_consistency", "true",
            ],
            check=True,
        )

        # Stereo fusion
        logger.info("Fusing depth maps (COLMAP binary)")
        output_ply = os.path.join(self.dense_dir, "fused.ply")
        subprocess.run(
            [
                "colmap", "stereo_fusion",
                "--workspace_path", self.dense_dir,
                "--workspace_format", "COLMAP",
                "--input_type", "geometric",
                "--output_path", output_ply,
            ],
            check=True,
        )

        logger.info(f"Dense reconstruction complete: {output_ply}")
        return output_ply
    
    def get_camera_data(self) -> Dict[str, np.ndarray]:
        """
        Extract camera intrinsics and extrinsics from reconstruction
        
        Returns:
            Dictionary with camera data
        """
        sparse_model_path = os.path.join(self.sparse_dir, "0")
        reconstruction = pycolmap.Reconstruction(sparse_model_path)
        
        camera_data = {
            'intrinsics': {},
            'extrinsics': {},
            'image_names': {}
        }
        
        for image_id, image in reconstruction.images.items():
            camera = reconstruction.cameras[image.camera_id]

            # Intrinsic matrix — SIMPLE_RADIAL / RADIAL / PINHOLE all store
            # (fx, fy, cx, cy) in the first four params; SIMPLE_PINHOLE only
            # has (f, cx, cy) so we handle both.
            params = camera.params
            if camera.model.name in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            else:  # PINHOLE, RADIAL, OPENCV, …
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]

            K = np.array([
                [fx,  0, cx],
                [ 0, fy, cy],
                [ 0,  0,  1],
            ])

            # COLMAP extrinsic: X_cam = R @ X_world + t
            # pycolmap 4.x: image.cam_from_world is a Rigid3d object.
            # rotation.matrix() gives the 3×3 rotation matrix.
            # translation gives the tvec.
            cam_from_world = image.cam_from_world
            R = cam_from_world.rotation.matrix()
            t = cam_from_world.translation
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = R
            extrinsic[:3,  3] = t  # camera-space transform (for unprojection)

            camera_data['intrinsics'][image_id]  = K
            camera_data['extrinsics'][image_id]  = extrinsic
            camera_data['image_names'][image_id] = image.name
            # Also store the world-space camera centre for spatial audio
            camera_data.setdefault('camera_centers', {})[image_id] = (-R.T @ t)
        
        logger.info(f"Extracted camera data for {len(camera_data['intrinsics'])} cameras")
        return camera_data


def reconstruct_scene(
    images: List[np.ndarray],
    image_paths: Optional[List[str]] = None,
    config: Dict = None,
    output_dir: str = "data/reconstruction"
) -> Dict:
    """Complete 3D reconstruction pipeline.

    The ``method`` key in config selects the reconstruction backend:

    * ``"dust3r"``     — DUSt3R transformer (default; works on AI-generated images,
                         no feature matching required).
    * ``"colmap"``     — Classic COLMAP SfM (requires real photos with photometric
                         consistency for SIFT to succeed).
    * ``"depth_only"`` — Skip camera pose estimation entirely; use synthetic
                         pinhole intrinsics from image dimensions.

    If DUSt3R or COLMAP fails, the pipeline falls back to depth-only mode so
    that spatial audio positioning still works.

    Args:
        images:      List of input images (H×W×3 uint8 numpy arrays).
        image_paths: Optional list of original image paths (used for filenames).
        config:      Configuration dictionary (reconstruction section).
        output_dir:  Root output directory.

    Returns:
        Dictionary with reconstruction results.
    """
    if config is None:
        config = {
            'method': 'dust3r',
            'use_depth_model': True,
            'depth_model': 'DPT_Large',
            'colmap': {
                'feature_type': 'SIFT',
                'matching_method': 'exhaustive',
                'dense_reconstruction': False,
            },
        }

    method = config.get('method', 'dust3r').lower()
    results = {}

    # ------------------------------------------------------------------
    # Step 1 — Depth estimation (always run; used for spatial audio even
    # when DUSt3R handles camera poses)
    # ------------------------------------------------------------------
    if config.get('use_depth_model', True):
        try:
            depth_estimator = DepthEstimator(
                model_name=config.get('depth_model', 'DPT_Large'),
                device='cuda' if torch.cuda.is_available() else 'cpu',
            )
            depth_maps = depth_estimator.estimate_depth_batch(
                images,
                output_dir=os.path.join(output_dir, "depth"),
            )
            results['depth_maps'] = depth_maps
            logger.info("Depth estimation completed successfully")
        except Exception as e:
            logger.error(f"Depth estimation failed: {str(e)}")
            results['depth_maps'] = None

    # ------------------------------------------------------------------
    # Step 2 — Camera pose estimation
    # ------------------------------------------------------------------

    if method == 'dust3r':
        # ── DUSt3R ─────────────────────────────────────────────────────
        try:
            from src.reconstruction_dust3r import reconstruct_with_dust3r
            logger.info("Running DUSt3R reconstruction")
            dust3r_results = reconstruct_with_dust3r(
                images,
                config=config,
                output_dir=output_dir,
            )
            # Preserve MiDaS depth maps: reconstruct_with_dust3r returns
            # depth_maps=None because DUSt3R doesn't compute them.
            # results.update() would overwrite the MiDaS list computed in
            # Step 1, so we save and restore it explicitly.
            _midas_depth_maps = results.get('depth_maps')
            results.update(dust3r_results)
            if _midas_depth_maps is not None:
                results['depth_maps'] = _midas_depth_maps
            logger.info("DUSt3R reconstruction completed successfully")
        except Exception as e:
            logger.error(f"DUSt3R reconstruction failed: {e}")
            results['camera_data'] = None

    elif method == 'colmap':
        # ── COLMAP ─────────────────────────────────────────────────────
        try:
            colmap = COLMAPReconstructor(output_dir=os.path.join(output_dir, "colmap"))
            colmap.prepare_images(images, image_paths)
            colmap.run_feature_extraction(
                feature_type=config.get('colmap', {}).get('feature_type', 'SIFT')
            )
            colmap.run_feature_matching(
                matching_method=config.get('colmap', {}).get('matching_method', 'exhaustive')
            )
            sparse_reconstruction = colmap.run_sparse_reconstruction()
            results['sparse_reconstruction'] = sparse_reconstruction
            camera_data = colmap.get_camera_data()
            results['camera_data'] = camera_data

            if config.get('colmap', {}).get('dense_reconstruction', False):
                try:
                    dense_ply = colmap.run_dense_reconstruction()
                    results['dense_point_cloud'] = dense_ply
                except Exception as e:
                    logger.warning(f"Dense reconstruction failed: {str(e)}")
                    results['dense_point_cloud'] = None

            logger.info("COLMAP reconstruction completed successfully")
        except Exception as e:
            logger.error(f"COLMAP reconstruction failed: {str(e)}")
            results['sparse_reconstruction'] = None
            results['camera_data'] = None

    else:
        # ── depth_only — skip pose estimation entirely ──────────────────
        logger.info("Reconstruction method=depth_only: skipping camera pose estimation")
        results['camera_data'] = None

    # Depth-only camera_data fallback
    # If COLMAP failed (camera_data is None) but we have depth maps, synthesise
    # a camera_data dict using pinhole intrinsics derived from image dimensions
    # and identity extrinsics.  This lets project_objects_to_3d produce
    # meaningful relative 3-D positions without any COLMAP output.
    if not results.get('camera_data') and results.get('depth_maps'):
        logger.info(
            "COLMAP unavailable — building synthetic camera_data from image "
            "shapes for depth-only 3-D projection"
        )
        synthetic = {
            'intrinsics': {},
            'extrinsics': {},
            'image_names': {},
            'camera_centers': {},
            'depth_only': True,
        }
        for idx, img in enumerate(images):
            h, w = img.shape[:2]
            fx = fy = 0.7 * w
            cx, cy = w / 2.0, h / 2.0
            K = np.array([
                [fx,  0, cx],
                [ 0, fy, cy],
                [ 0,  0,  1],
            ], dtype=np.float64)
            synthetic['intrinsics'][idx]  = K
            synthetic['extrinsics'][idx]  = np.eye(4)
            synthetic['image_names'][idx] = f"image_{idx:03d}"
            synthetic['camera_centers'][idx] = np.zeros(3)
        results['camera_data'] = synthetic
        logger.info(
            f"Synthetic camera_data created for {len(images)} images "
            "(fx=fy=0.7*W, identity extrinsics)"
        )

    # Persist camera_data to disk so Stage 7 can reload it without re-running Stage 3
    if results.get('camera_data'):
        _cam_json = os.path.join(output_dir, "camera_data.json")
        try:
            save_camera_data(results['camera_data'], _cam_json)
            logger.info(f"Camera data persisted to {_cam_json}")
        except Exception as _e:
            logger.warning(f"Could not save camera_data to disk: {_e}")

    return results


def save_camera_data(camera_data: Dict, path: str) -> None:
    """Serialize ``camera_data`` (numpy arrays, int keys) to JSON.

    Args:
        camera_data: Dict with keys ``intrinsics``, ``extrinsics``,
                     ``image_names``, ``camera_centers``, and optional flags.
        path:        Destination JSON path.
    """
    import json as _json

    def _to_serializable(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        return v

    def _dict_to_serial(d):
        return {str(k): _to_serializable(v) for k, v in d.items()}

    out = {
        "intrinsics":     _dict_to_serial(camera_data.get("intrinsics", {})),
        "extrinsics":     _dict_to_serial(camera_data.get("extrinsics", {})),
        "image_names":    _dict_to_serial(camera_data.get("image_names", {})),
        "camera_centers": _dict_to_serial(camera_data.get("camera_centers", {})),
        "dust3r":         bool(camera_data.get("dust3r", False)),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        _json.dump(out, fh, indent=2)


def load_camera_data(path: str) -> Dict:
    """Load ``camera_data`` saved by :func:`save_camera_data`.

    Returns a dict with the same structure as ``reconstruct_scene`` produces,
    with numpy array values and integer keys.

    Args:
        path: JSON path written by ``save_camera_data``.

    Returns:
        camera_data dict ready for ``export_dust3r_to_nerfstudio`` / ``project_objects_to_3d``.
    """
    import json as _json

    with open(path, encoding="utf-8") as fh:
        raw = _json.load(fh)

    def _to_array(v):
        return np.array(v, dtype=np.float64) if isinstance(v, list) else v

    def _parse_dict(d, value_fn=_to_array):
        return {int(k): value_fn(v) for k, v in d.items()}

    return {
        "intrinsics":     _parse_dict(raw.get("intrinsics", {})),
        "extrinsics":     _parse_dict(raw.get("extrinsics", {})),
        "image_names":    {int(k): v for k, v in raw.get("image_names", {}).items()},
        "camera_centers": _parse_dict(raw.get("camera_centers", {})),
        "dust3r":         raw.get("dust3r", False),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage — same folder as Stage 2 (`image_generation.output_dir` in config.yaml)
    from src.utils import load_pipeline_view_images
    
    images, paths = load_pipeline_view_images("data/Nano_banana_output_images")
    print(f"Loaded {len(images)} images")
    
    results = reconstruct_scene(images, paths)
    
    print("\nReconstruction Results:")
    print(f"- Depth maps: {len(results.get('depth_maps', [])) if results.get('depth_maps') else 'None'}")
    print(f"- Sparse reconstruction: {results.get('sparse_reconstruction') is not None}")
    print(f"- Camera data: {len(results.get('camera_data', {}).get('intrinsics', {})) if results.get('camera_data') else 0} cameras")
