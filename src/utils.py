"""Utility functions for the pipeline"""

import os
import yaml
import logging
from pathlib import Path
import re
from typing import Dict, Any, List, Tuple, Union
import numpy as np
from PIL import Image
import json


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('pipeline.log')
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save dictionary to JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> Dict[str, Any]:
    """Load dictionary from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


_VIEW_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


def _pipeline_view_sort_key(path: Path) -> Tuple[int, int, str]:
    """Non-generated files first (lexicographic), then ``generated_view_XX`` by index."""
    name_lower = path.name.lower()
    m = re.match(r'generated_view_(\d+)\.', name_lower)
    if m:
        return (1, int(m.group(1)), path.name)
    return (0, 0, path.name)


def load_pipeline_view_images(directory: str) -> Tuple[List[np.ndarray], List[str]]:
    """Load all RGB views from the unified synthesis folder (e.g. ``Nano_banana_output_images``).

    Used for reconstruction and downstream steps so every frame shares the same
    pixel resolution as written by the image-generation stage. Does not read
    ``data/input`` — only files inside ``directory``.
    """
    root = Path(directory)
    if not root.is_dir():
        return [], []

    paths = [p for p in root.iterdir() if p.suffix.lower() in _VIEW_IMAGE_EXTS]
    paths.sort(key=_pipeline_view_sort_key)

    images: List[np.ndarray] = []
    str_paths: List[str] = []
    for p in paths:
        with Image.open(p) as im:
            images.append(np.array(im.convert('RGB')))
        str_paths.append(str(p))

    return images, str_paths


def load_images_from_directory(directory: str) -> List[np.ndarray]:
    """Load all images from a directory"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    images = []
    image_paths = []
    
    directory = Path(directory)
    for filepath in sorted(directory.iterdir()):
        if filepath.suffix.lower() in image_extensions:
            img = Image.open(filepath)
            images.append(np.array(img))
            image_paths.append(str(filepath))
    
    return images, image_paths


def save_image(image: np.ndarray, filepath: str) -> None:
    """Save numpy array as image"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    Image.fromarray(image).save(filepath)


def save_rgb_jpeg_at_size(
    img_arr: np.ndarray,
    dest: Union[str, Path],
    size: Tuple[int, int],
    quality: int = 95,
) -> None:
    """Save ``H×W×3`` RGB as JPEG, resizing to ``size`` (width, height) if needed."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if img_arr.dtype != np.uint8:
        img_arr = (np.clip(img_arr, 0.0, 1.0) * 255).astype(np.uint8)
    pil = Image.fromarray(img_arr).convert("RGB")
    if pil.size != size:
        pil = pil.resize(size, Image.Resampling.LANCZOS)
    pil.save(dest, format="JPEG", quality=quality)


def create_output_directories(base_path: str = "data") -> Dict[str, str]:
    """Create all necessary output directories"""
    directories = {
        'input': os.path.join(base_path, 'input'),
        'generated': os.path.join(base_path, 'generated'),
        'reconstruction': os.path.join(base_path, 'reconstruction'),
        'segmentation': os.path.join(base_path, 'segmentation'),
        'audio': os.path.join(base_path, 'audio'),
        'output': os.path.join(base_path, 'output'),
    }
    
    for directory in directories.values():
        os.makedirs(directory, exist_ok=True)
    
    return directories


def normalize_depth_map(depth_map: np.ndarray) -> np.ndarray:
    """Normalize depth map to [0, 1] range"""
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    if depth_max - depth_min < 1e-6:
        return np.zeros_like(depth_map)
    return (depth_map - depth_min) / (depth_max - depth_min)


def unproject_2d_to_3d(
    point_2d: np.ndarray,
    depth: float,
    camera_matrix: np.ndarray,
    camera_pose: np.ndarray = None
) -> np.ndarray:
    """
    Unproject 2D image coordinates to 3D world coordinates
    
    Args:
        point_2d: [x, y] coordinates in image space
        depth: Depth value at that point
        camera_matrix: 3x3 camera intrinsic matrix
        camera_pose: 4x4 camera extrinsic matrix (optional)
    
    Returns:
        3D point in world coordinates [x, y, z]
    """
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    # Convert to camera coordinates
    x = (point_2d[0] - cx) * depth / fx
    y = (point_2d[1] - cy) * depth / fy
    z = depth
    
    point_3d = np.array([x, y, z, 1.0])
    
    # Transform to world coordinates if pose provided
    if camera_pose is not None:
        point_3d = camera_pose @ point_3d
    
    return point_3d[:3]


def compute_bounding_box_center(bbox: List[float]) -> np.ndarray:
    """
    Compute center of bounding box
    
    Args:
        bbox: [x_min, y_min, x_max, y_max]
    
    Returns:
        Center point [x, y]
    """
    x_center = (bbox[0] + bbox[2]) / 2
    y_center = (bbox[1] + bbox[3]) / 2
    return np.array([x_center, y_center])


def estimate_audio_intensity(
    object_label: str,
    distance: float,
    falloff: str = "inverse_square"
) -> float:
    """
    Estimate audio intensity based on object type and distance
    
    Args:
        object_label: Name/category of the object
        distance: Distance from listener
        falloff: Type of falloff ("inverse_square", "linear", "constant")
    
    Returns:
        Intensity value [0, 1]
    """
    # Base intensity by object category
    loud_objects = {'water', 'waterfall', 'fountain', 'speaker', 'machine', 'engine'}
    medium_objects = {'wind', 'leaves', 'footsteps', 'door', 'clock'}
    quiet_objects = {'plant', 'furniture', 'wall', 'floor'}
    
    base_intensity = 0.5
    for keyword in loud_objects:
        if keyword in object_label.lower():
            base_intensity = 0.9
            break
    for keyword in medium_objects:
        if keyword in object_label.lower():
            base_intensity = 0.6
            break
    for keyword in quiet_objects:
        if keyword in object_label.lower():
            base_intensity = 0.3
            break
    
    # Apply distance falloff
    if falloff == "inverse_square":
        intensity = base_intensity / (1 + distance ** 2)
    elif falloff == "linear":
        intensity = base_intensity * max(0, 1 - distance / 10)
    else:  # constant
        intensity = base_intensity
    
    return np.clip(intensity, 0.0, 1.0)
