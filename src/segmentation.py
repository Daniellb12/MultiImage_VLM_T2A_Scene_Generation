"""Scene Segmentation Module using Qwen2.5-VL"""

import os
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image
import json

try:
    import torch
    # The correct class name for Qwen2.5-VL is Qwen2_5_VLForConditionalGeneration
    # (Qwen2VLForConditionalGeneration is the older Qwen2-VL family)
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
except ImportError:
    raise ImportError(
        "Please install required packages:\n"
        "pip install transformers>=4.49.0 torch qwen-vl-utils[decord]"
    )

logger = logging.getLogger(__name__)


class SceneSegmenter:
    """Object detection and segmentation using Qwen2.5-VL"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
        quantization: str = "bf16",
        max_pixels: int = 1280,
        min_pixels: int = 256
    ):
        """
        Initialize the scene segmenter
        
        Args:
            model_name: Model name from HuggingFace
            device: Device to run on (cuda, cpu, mps)
            quantization: Quantization type (bf16, fp16, fp32, int8, int4)
            max_pixels: Maximum pixels for image processing
            min_pixels: Minimum pixels for image processing
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        
        logger.info(f"Loading Qwen2.5-VL model: {model_name} on {self.device}")
        logger.info(f"Quantization: {quantization}")
        
        # Determine dtype — int8/int4 are handled by BitsAndBytes, not as a
        # raw torch dtype, so we load in fp16 and let the quantization_config
        # handle further compression.
        if quantization == "bf16":
            torch_dtype = torch.bfloat16
            quantization_config = None
        elif quantization == "fp16":
            torch_dtype = torch.float16
            quantization_config = None
        elif quantization in ("int8", "int4"):
            torch_dtype = torch.float16
            try:
                from transformers import BitsAndBytesConfig
                load_in_4bit = quantization == "int4"
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=not load_in_4bit,
                )
            except ImportError:
                logger.warning("bitsandbytes not installed, falling back to bf16")
                torch_dtype = torch.bfloat16
                quantization_config = None
        else:
            torch_dtype = torch.float32
            quantization_config = None

        try:
            # Load model with the correct Qwen2.5-VL class
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                quantization_config=quantization_config,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
            )
            
            if device != "cuda":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def segment_image(
        self,
        image: np.ndarray,
        prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Segment objects in a single image
        
        Args:
            image: Input image as numpy array
            prompt: Optional custom prompt
        
        Returns:
            Dictionary with segmentation results
        """
        if prompt is None:
            prompt = """Analyze this image and identify all distinct objects and surfaces present.
For each object, provide:
1. Object name/label
2. Category (e.g., furniture, electronics, nature, etc.)
3. Approximate bounding box coordinates as [x_min, y_min, x_max, y_max] in normalized coordinates (0-1)
4. Confidence score (0-1)

Format the output as a JSON array of objects. Example:
[
  {
    "label": "wooden table",
    "category": "furniture",
    "bbox": [0.2, 0.5, 0.8, 0.9],
    "confidence": 0.95
  }
]

Provide ONLY the JSON array, no additional text."""
        
        # Convert to PIL Image
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(image)
        
        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": pil_image,
                        "resized_height": self.max_pixels,
                        "resized_width": self.max_pixels,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Prepare for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.1,
                top_p=0.9,
            )
        
        # Trim input tokens from output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # Decode output
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        # Parse JSON output
        try:
            # Try to extract JSON from the output
            json_start = output_text.find('[')
            json_end = output_text.rfind(']') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = output_text[json_start:json_end]
                objects = json.loads(json_str)
            else:
                logger.warning("Could not find JSON array in output")
                objects = []
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON output: {str(e)}")
            logger.debug(f"Raw output: {output_text}")
            objects = []
        
        return {
            'objects': objects,
            'raw_output': output_text,
            'image_shape': image.shape
        }
    
    def segment_batch(
        self,
        images: List[np.ndarray],
        output_dir: str = "data/segmentation"
    ) -> List[Dict[str, Any]]:
        """
        Segment objects in multiple images
        
        Args:
            images: List of input images
            output_dir: Directory to save segmentation results
        
        Returns:
            List of segmentation results for each image
        """
        logger.info(f"Segmenting {len(images)} images")
        os.makedirs(output_dir, exist_ok=True)
        
        all_results = []
        
        for i, image in enumerate(images):
            logger.info(f"Segmenting image {i+1}/{len(images)}")
            
            try:
                result = self.segment_image(image)
                result['image_index'] = i
                all_results.append(result)
                
                # Save results
                output_path = os.path.join(output_dir, f"segmentation_{i:03d}.json")
                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=2)
                
                logger.info(f"Found {len(result['objects'])} objects in image {i}")
                
                # Save visualization (optional)
                self._save_visualization(image, result, output_dir, i)
                
            except Exception as e:
                logger.error(f"Error segmenting image {i}: {str(e)}")
                all_results.append({
                    'objects': [],
                    'error': str(e),
                    'image_index': i,
                    'image_shape': image.shape,  # needed by project_objects_to_3d
                })
        
        # Save combined results
        combined_path = os.path.join(output_dir, "all_segmentations.json")
        with open(combined_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"Segmentation complete. Results saved to {output_dir}")
        return all_results
    
    def _save_visualization(
        self,
        image: np.ndarray,
        result: Dict[str, Any],
        output_dir: str,
        index: int
    ) -> None:
        """
        Save visualization of detected objects with bounding boxes
        
        Args:
            image: Original image
            result: Segmentation result
            output_dir: Output directory
            index: Image index
        """
        try:
            from PIL import ImageDraw, ImageFont

            img_u8 = image if image.dtype == np.uint8 else (image * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_u8)
            draw = ImageDraw.Draw(pil_image)

            h, w = img_u8.shape[:2]
            
            # Draw bounding boxes
            for obj in result['objects']:
                if 'bbox' in obj and len(obj['bbox']) == 4:
                    bbox = obj['bbox']
                    
                    # Convert normalized coordinates to pixel coordinates
                    x1 = int(bbox[0] * w)
                    y1 = int(bbox[1] * h)
                    x2 = int(bbox[2] * w)
                    y2 = int(bbox[3] * h)
                    
                    # Draw rectangle
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                    
                    # Draw label
                    label = obj.get('label', 'unknown')
                    confidence = obj.get('confidence', 0.0)
                    text = f"{label} ({confidence:.2f})"
                    
                    # Draw text background
                    try:
                        font = ImageFont.truetype("arial.ttf", 16)
                    except:
                        font = ImageFont.load_default()
                    
                    bbox_text = draw.textbbox((x1, y1 - 20), text, font=font)
                    draw.rectangle(bbox_text, fill="red")
                    draw.text((x1, y1 - 20), text, fill="white", font=font)
            
            # Save visualization
            vis_path = os.path.join(output_dir, f"visualization_{index:03d}.png")
            pil_image.save(vis_path)
            logger.info(f"Saved visualization to {vis_path}")
            
        except Exception as e:
            logger.warning(f"Could not save visualization: {str(e)}")
    
    def get_scene_description(self, image: np.ndarray) -> str:
        """
        Get a natural language description of the scene
        
        Args:
            image: Input image
        
        Returns:
            Scene description
        """
        prompt = """Provide a detailed description of this scene, including:
1. The type of environment (indoor/outdoor, room type, etc.)
2. Main objects and their spatial relationships
3. Colors, textures, and lighting
4. Any notable features or activities

Keep the description concise but informative."""
        
        img_u8 = image if image.dtype == np.uint8 else (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_u8)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return output_text


def _make_pinhole_intrinsic(w: int, h: int) -> np.ndarray:
    """
    Synthesise a pinhole intrinsic matrix from image dimensions.

    Assumes a 35 mm-equivalent focal length (~65° HFOV), which is a reasonable
    default for interior photographs captured without COLMAP.  The focal length
    in pixels is fx = fy = 0.7 * W (derived from tan(HFOV/2) = W/2 / fx).
    """
    fx = fy = 0.7 * w
    cx, cy = w / 2.0, h / 2.0
    return np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1],
    ], dtype=np.float64)


def project_objects_to_3d(
    segmentation_results: List[Dict[str, Any]],
    camera_data: Optional[Dict[str, np.ndarray]],
    depth_maps: Optional[List[np.ndarray]] = None
) -> List[Dict[str, Any]]:
    """
    Project 2D detected objects to 3D space.

    When ``camera_data`` is ``None`` (e.g. COLMAP failed) but ``depth_maps``
    are available, the function falls back to depth-only positioning:

    - Pinhole intrinsics are synthesised from the image dimensions
      (35 mm equivalent, ~65° HFOV: fx = fy = 0.7 * W).
    - Camera extrinsics are set to the identity matrix (camera-space = world-
      space), which is internally consistent for monocular depth even though
      the absolute metric scale is unknown.
    - MiDaS depth is relative, not metric, but the *ratios* between object
      distances are preserved, which is sufficient for spatial audio panning.

    Args:
        segmentation_results: List of segmentation results from segment_batch.
        camera_data: Camera intrinsics and extrinsics from COLMAP, or ``None``
            to use the depth-only fallback.
        depth_maps: Optional depth maps for each image (required for the
            depth-only fallback to produce meaningful positions).

    Returns:
        List of objects with 3D positions.
    """
    from src.utils import unproject_2d_to_3d, compute_bounding_box_center

    depth_only_mode = camera_data is None
    if depth_only_mode:
        logger.info(
            "project_objects_to_3d: no COLMAP camera_data — using depth-only "
            "pinhole fallback (fx=fy=0.7*W, identity extrinsics)"
        )

    objects_3d = []

    for result in segmentation_results:
        image_idx = result.get('image_index', 0)
        h, w = result.get('image_shape', [480, 640])[:2]

        # --- Determine intrinsic / extrinsic for this image ---
        if depth_only_mode:
            intrinsic = _make_pinhole_intrinsic(w, h)
            extrinsic = np.eye(4)          # identity: camera-space == world-space
        else:
            if image_idx not in camera_data.get('intrinsics', {}):
                logger.warning(f"No camera data for image {image_idx}, skipping")
                continue
            intrinsic = camera_data['intrinsics'][image_idx]
            extrinsic = camera_data['extrinsics'][image_idx]

        # Get depth map if available
        depth_map = depth_maps[image_idx] if depth_maps and image_idx < len(depth_maps) else None

        for obj in result.get('objects', []):
            if 'bbox' not in obj:
                continue

            bbox = obj['bbox']

            # Compute centre of bounding box (normalised → pixel)
            center_2d = compute_bounding_box_center(bbox)
            center_2d_pixel = np.array([center_2d[0] * w, center_2d[1] * h])

            # Get depth value at centre pixel
            if depth_map is not None:
                depth_y = int(np.clip(center_2d_pixel[1], 0, h - 1))
                depth_x = int(np.clip(center_2d_pixel[0], 0, w - 1))
                depth = float(depth_map[depth_y, depth_x])
            else:
                depth = 5.0
                logger.warning(
                    f"No depth map for image {image_idx}, using default depth 5.0"
                )

            # Unproject to 3D
            position_3d = unproject_2d_to_3d(
                center_2d_pixel,
                depth,
                intrinsic,
                extrinsic,
            )

            obj_3d = {
                'id': f"obj_{image_idx:03d}_{len(objects_3d):03d}",
                'label': obj.get('label', 'unknown'),
                'category': obj.get('category', 'unknown'),
                'confidence': obj.get('confidence', 0.0),
                'position_3d': position_3d.tolist(),
                'position_2d': center_2d_pixel.tolist(),
                'bbox': bbox,
                'image_index': image_idx,
                'depth': depth,
                'depth_only': depth_only_mode,
            }

            objects_3d.append(obj_3d)

    logger.info(f"Projected {len(objects_3d)} objects to 3D space")
    return objects_3d


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    from src.utils import load_images_from_directory
    
    images, paths = load_images_from_directory("data/input")
    print(f"Loaded {len(images)} images")
    
    # Initialize segmenter
    segmenter = SceneSegmenter(
        quantization="bf16",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Segment images
    results = segmenter.segment_batch(images)
    
    print(f"\nSegmentation complete:")
    for i, result in enumerate(results):
        print(f"Image {i}: {len(result['objects'])} objects detected")
