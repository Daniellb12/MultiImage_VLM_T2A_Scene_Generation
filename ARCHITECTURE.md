# System Architecture

## Overview Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Multi-Modal Scene Audio Pipeline                          │
└─────────────────────────────────────────────────────────────────────────────┘

INPUT: 4 Multi-View Images
    │
    ├─────────────────────────────────────────────────────────────────────┐
    │                                                                       │
    v                                                                       │
┌─────────────────────────────────────────────────────────────────────┐   │
│  STAGE 1: Image Generation (Gemini Nano Banana)                     │   │
│  - Scene analysis                                                    │   │
│  - Viewpoint synthesis (6 additional views)                         │   │
│  - Output: 10 total images                                          │   │
└─────────────────────────────────────────────────────────────────────┘   │
    │                                                                       │
    v                                                                       │
┌─────────────────────────────────────────────────────────────────────┐   │
│  STAGE 2: 3D Reconstruction                                         │   │
│  ┌──────────────────────┐  ┌──────────────────────┐               │   │
│  │ Track A: COLMAP      │  │ Track B: Depth Est.  │               │   │
│  │ - Feature extraction │  │ - MiDaS v3 DPT      │               │   │
│  │ - Feature matching   │  │ - Per-image depth    │               │   │
│  │ - Sparse SfM        │  │ - Fast preview       │               │   │
│  │ - Dense MVS         │  │                      │               │   │
│  └──────────────────────┘  └──────────────────────┘               │   │
│  Output: Point cloud + Camera poses + Depth maps                   │   │
└─────────────────────────────────────────────────────────────────────┘   │
    │                                                                       │
    v                                                                       │
┌─────────────────────────────────────────────────────────────────────┐   │
│  STAGE 3: Scene Segmentation (Qwen2.5-VL-7B)                       │   │
│  - Object detection                                                 │   │
│  - Bounding box regression                                          │   │
│  - Label & category classification                                 │   │
│  - 2D → 3D projection (using camera poses + depth)                 │   │
│  Output: List of objects with 3D positions                          │   │
└─────────────────────────────────────────────────────────────────────┘   │
    │                                                                       │
    v                                                                       │
┌─────────────────────────────────────────────────────────────────────┐   │
│  STAGE 4: Audio Generation (MMAudio)                                │   │
│  - Object → Sound prompt mapping                                    │   │
│  - Foley audio synthesis (per object)                              │   │
│  - Ambient scene audio                                              │   │
│  Output: WAV files for each object                                  │   │
└─────────────────────────────────────────────────────────────────────┘   │
    │                                                                       │
    v                                                                       │
┌─────────────────────────────────────────────────────────────────────┐   │
│  STAGE 5: Spatial Audio Positioning                                 │   │
│  - Intensity calculation (distance falloff)                         │   │
│  - Audio manifest generation                                        │   │
│  - 3D visualization                                                 │   │
│  - Export (Unity, HTML viewer)                                      │   │
└─────────────────────────────────────────────────────────────────────┘   │
    │                                                                       │
    v                                                                       │
OUTPUT:                                                                    │
  - audio_manifest.json (spatial positions)                               │
  - audio_visualization.html (interactive viewer)                         │
  - unity_scene.json (game engine export)                                │
  - Individual WAV files                                                  │
  - 3D point cloud (PLY)                                                 │
    │                                                                       │
    └───────────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
┌──────────────┐
│ Input Images │ (4 views, JPEG/PNG)
└──────┬───────┘
       │
       v
┌──────────────────────┐
│ ImageGenerator       │ 
│ - Gemini API client  │ → API Call → [Gemini Cloud]
│ - Prompt engineering │
└──────┬───────────────┘
       │ Generated images (6 views)
       v
┌──────────────────────┐       ┌──────────────────────┐
│ COLMAPReconstructor  │       │ DepthEstimator       │
│ - pycolmap bindings  │       │ - MiDaS model        │
│ - Feature matching   │       │ - torch inference    │
└──────┬───────────────┘       └──────┬───────────────┘
       │ Point cloud + cameras        │ Depth maps
       └──────────┬──────────────────┘
                  v
       ┌──────────────────────┐
       │ SceneSegmenter       │
       │ - Qwen2.5-VL model   │
       │ - Vision-Language    │
       │ - JSON parser        │
       └──────┬───────────────┘
              │ Objects with labels + 2D boxes
              v
       ┌──────────────────────┐
       │ project_objects_to_3d│
       │ - Unproject 2D → 3D  │
       │ - Depth lookup       │
       └──────┬───────────────┘
              │ Objects with 3D positions
              v
       ┌──────────────────────┐
       │ AudioGenerator       │
       │ - MMAudio API client │ → API Call → [MMAudio Space]
       │ - Prompt mapping     │
       └──────┬───────────────┘
              │ Audio files (WAV)
              v
       ┌──────────────────────────┐
       │ SpatialAudioPositioner   │
       │ - Intensity calculator   │
       │ - Manifest builder       │
       │ - Visualization          │
       └──────┬───────────────────┘
              │
              v
       ┌──────────────────────┐
       │ Final Outputs        │
       │ - JSON manifest      │
       │ - HTML viewer        │
       │ - Unity export       │
       └──────────────────────┘
```

## Module Dependencies

```
main.py
  │
  ├─→ src.utils
  │     ├─ setup_logging()
  │     ├─ load_config()
  │     ├─ create_output_directories()
  │     └─ load_images_from_directory()
  │
  ├─→ src.image_generation
  │     └─ ImageGenerator
  │           ├─ google.genai (external API)
  │           └─ PIL, numpy
  │
  ├─→ src.reconstruction
  │     ├─ DepthEstimator
  │     │     ├─ torch
  │     │     └─ torch.hub (MiDaS)
  │     │
  │     └─ COLMAPReconstructor
  │           ├─ pycolmap
  │           └─ open3d
  │
  ├─→ src.segmentation
  │     ├─ SceneSegmenter
  │     │     ├─ transformers (Qwen2VLForConditionalGeneration)
  │     │     ├─ torch
  │     │     └─ qwen_vl_utils
  │     │
  │     └─ project_objects_to_3d()
  │           └─ src.utils (unproject_2d_to_3d)
  │
  ├─→ src.audio_generation
  │     └─ AudioGenerator
  │           ├─ gradio_client (MMAudio API)
  │           └─ soundfile (optional)
  │
  └─→ src.spatial_audio
        ├─ SpatialAudioPositioner
        │     ├─ open3d
        │     └─ src.utils (estimate_audio_intensity)
        │
        └─ export_for_unity()
```

## API Integration Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    External Services                         │
└─────────────────────────────────────────────────────────────┘

┌──────────────────┐         ┌──────────────────┐
│  Gemini API      │         │ MMAudio Space    │
│  (Google)        │         │ (HuggingFace)    │
└────────┬─────────┘         └────────┬─────────┘
         │ HTTPS                      │ HTTPS
         │ REST                       │ Gradio Protocol
         v                            v
┌──────────────────┐         ┌──────────────────┐
│ google.genai     │         │ gradio_client    │
│ Client           │         │ Client           │
└────────┬─────────┘         └────────┬─────────┘
         │                            │
         v                            v
┌──────────────────┐         ┌──────────────────┐
│ ImageGenerator   │         │ AudioGenerator   │
└──────────────────┘         └──────────────────┘


┌─────────────────────────────────────────────────────────────┐
│                    Local Models                              │
└─────────────────────────────────────────────────────────────┘

┌──────────────────┐         ┌──────────────────┐
│ MiDaS v3 DPT     │         │ Qwen2.5-VL-7B   │
│ (torch.hub)      │         │ (HuggingFace)   │
└────────┬─────────┘         └────────┬─────────┘
         │                            │
         v                            v
┌──────────────────┐         ┌──────────────────┐
│ DepthEstimator   │         │ SceneSegmenter   │
└──────────────────┘         └──────────────────┘
```

## State Management

```
Pipeline.state = {
    'input_images': List[np.ndarray],           # Stage 1 input
    'generated_images': List[np.ndarray],       # Stage 1 output
    'all_images': List[np.ndarray],            # Combined images
    'scene_description': str,                   # Scene analysis
    
    'reconstruction_results': {                 # Stage 2 output
        'depth_maps': List[np.ndarray],
        'sparse_reconstruction': pycolmap.Reconstruction,
        'dense_point_cloud': str,              # Path to PLY
        'camera_data': {
            'intrinsics': Dict[int, np.ndarray],  # 3x3 matrices
            'extrinsics': Dict[int, np.ndarray],  # 4x4 matrices
            'image_names': Dict[int, str]
        }
    },
    
    'segmentation_results': [                   # Stage 3 output
        {
            'objects': [
                {
                    'label': str,
                    'category': str,
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float
                }
            ],
            'image_index': int
        }
    ],
    
    'objects_3d': [                            # Stage 3 projected
        {
            'id': str,
            'label': str,
            'category': str,
            'position_3d': [x, y, z],
            'position_2d': [x, y],
            'bbox': [x1, y1, x2, y2],
            'confidence': float,
            'depth': float
        }
    ],
    
    'objects_with_audio': [                    # Stage 4 output
        {
            ...objects_3d fields...,
            'audio_file': str,                 # Path to WAV
            'audio_duration': float
        }
    ],
    
    'audio_manifest': {                        # Stage 5 output
        'metadata': {...},
        'audio_sources': [
            {
                'id': str,
                'label': str,
                'audio_file': str,
                'position_3d': [x, y, z],
                'intensity': float
            }
        ],
        'listener_positions': [
            {
                'camera_id': int,
                'position': [x, y, z],
                'forward': [x, y, z]
            }
        ]
    }
}
```

## Error Handling Strategy

```
┌────────────────────────────────────────────────────────────┐
│               Graceful Degradation Hierarchy                │
└────────────────────────────────────────────────────────────┘

IF Image Generation FAILS:
    → Continue with input images only
    → Log warning: "Using input images only"

IF COLMAP FAILS:
    → Use depth estimation only
    → Continue with estimated camera poses
    → Log warning: "3D reconstruction degraded"

IF Depth Estimation FAILS:
    → Use default depth values (5.0m)
    → Log warning: "Using default depths"

IF Segmentation FAILS:
    → Pipeline cannot continue (critical failure)
    → Save intermediate results
    → Exit with error code 1

IF Audio Generation FAILS (single object):
    → Skip that object
    → Continue with remaining objects
    → Log error for failed object

IF Audio Generation FAILS (all objects):
    → Create manifest without audio files
    → Continue to visualization
    → Log warning: "No audio generated"

IF Spatial Positioning FAILS:
    → Save audio files without spatial data
    → Skip visualization
    → Log warning: "Spatial positioning failed"
```

## Performance Optimization Points

```
┌──────────────────────────────────────────────────────────┐
│              Bottlenecks & Optimizations                  │
└──────────────────────────────────────────────────────────┘

1. Image Generation (API-bound)
   Bottleneck: Network latency, API rate limits
   Optimization: Batch multiple prompts, cache results

2. COLMAP (CPU/Memory-bound)
   Bottleneck: Feature matching O(n²) with image count
   Optimization: Use GPU-accelerated COLMAP, reduce image resolution

3. Depth Estimation (GPU-bound)
   Bottleneck: Model inference time per image
   Optimization: Batch processing, use smaller model (DPT-Hybrid)

4. Segmentation (GPU-bound)
   Bottleneck: VLM inference, 7B parameters
   Optimization: Quantization (int8/int4), reduce max_pixels

5. Audio Generation (API-bound)
   Bottleneck: Sequential API calls, 1/sec rate limit
   Optimization: Limit object count, cache by label

6. I/O Operations (Disk-bound)
   Bottleneck: Saving large depth maps, point clouds
   Optimization: Use SSD, compress intermediate results
```

## Security Considerations

```
┌──────────────────────────────────────────────────────────┐
│                  Security Architecture                    │
└──────────────────────────────────────────────────────────┘

API Keys:
    ├─ Stored in .env file (not committed to git)
    ├─ Loaded via python-dotenv
    └─ Never logged or displayed

External API Calls:
    ├─ HTTPS only (verified certificates)
    ├─ No sensitive data in requests
    └─ Rate limiting respected

File System:
    ├─ All writes to designated data/ directory
    ├─ No arbitrary path execution
    └─ Input validation on file paths

Model Execution:
    ├─ Trusted models from official sources
    ├─ No code execution from user input
    └─ Sandboxed inference (torch no-grad mode)

Dependencies:
    ├─ Pinned versions in requirements.txt
    ├─ Regular security updates recommended
    └─ No known vulnerabilities in current stack
```

## Scalability Analysis

```
┌──────────────────────────────────────────────────────────┐
│              Scaling Characteristics                      │
└──────────────────────────────────────────────────────────┘

Horizontal Scaling (Multiple Scenes):
    ✓ Each scene is independent
    ✓ Can process N scenes in parallel
    ✓ No shared state between runs
    → Embarrassingly parallel

Vertical Scaling (Single Scene):
    Depth Estimation:     O(n) with image count
    COLMAP Feature Match: O(n²) with image count  ⚠
    Segmentation:         O(n) with image count
    Audio Generation:     O(m) with object count
    
    Limiting factor: COLMAP quadratic growth
    
    Recommendations:
    - Keep image count ≤ 20 for reasonable runtime
    - Use sequential matching for > 20 images
    - Consider distributed COLMAP for large scenes

GPU Memory Scaling:
    Qwen2.5-VL-7B:     6-8 GB (bf16)
    MiDaS DPT:         2-4 GB
    Peak usage:        8-10 GB (segmentation)
    
    For lower memory:
    - Use int8 quantization (4GB)
    - Process images sequentially
    - Reduce batch_size to 1

Storage Scaling:
    Per scene estimate:
    - Input images: 20-40 MB
    - Generated images: 40-60 MB
    - Depth maps: 100-200 MB
    - COLMAP sparse: 10-50 MB
    - COLMAP dense: 500-2000 MB (optional)
    - Segmentation: 5-10 MB
    - Audio files: 5-20 MB
    - Total: ~200 MB per scene (without dense)
```

## Extension Points

```
┌──────────────────────────────────────────────────────────┐
│              How to Extend the Pipeline                   │
└──────────────────────────────────────────────────────────┘

Add New Image Generation Backend:
    1. Subclass ImageGenerator
    2. Implement generate_viewpoint()
    3. Update config.yaml with new model name

Add New 3D Reconstruction Method:
    1. Create new class in reconstruction.py
    2. Implement common interface (get_camera_data, etc.)
    3. Add to reconstruct_scene() function

Add New Segmentation Model:
    1. Subclass SceneSegmenter
    2. Implement segment_image() with same output format
    3. Update config.yaml

Add New Audio Generator:
    1. Subclass AudioGenerator
    2. Implement generate_audio()
    3. Add local inference support

Add New Output Format:
    1. Add export function to spatial_audio.py
    2. Follow export_for_unity() pattern
    3. Create format-specific manifest

Add Pipeline Step:
    1. Create new module in src/
    2. Add to Pipeline class as method
    3. Update state dictionary
    4. Add to main.run() sequence
```

---

This architecture is designed for:
- **Modularity:** Each component can be swapped
- **Robustness:** Graceful degradation on failures
- **Extensibility:** Clear extension points
- **Performance:** Identified bottlenecks and optimizations
- **Scalability:** Analysis of scaling characteristics

For implementation details, see individual module documentation.
