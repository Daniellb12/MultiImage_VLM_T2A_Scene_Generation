# Quick Start Guide

Get up and running with the Multi-Modal Scene Audio Generation Pipeline in minutes.

## Prerequisites

- Python 3.9 or higher
- NVIDIA GPU with 8GB+ VRAM (recommended for Qwen2.5-VL)
- ~50GB free disk space
- Internet connection for API access

## Installation

### 1. Set up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** PyTorch installation may require specific commands for your CUDA version. Visit [pytorch.org](https://pytorch.org) for details.

### 3. Set up API Keys

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your API keys
# You can use any text editor:
notepad .env  # Windows
nano .env     # Linux/Mac
```

Required API keys:
- **GEMINI_API_KEY**: Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
- **HF_TOKEN** (optional): Get from [HuggingFace Settings](https://huggingface.co/settings/tokens)

### 4. Verify Setup

```bash
python test_setup.py
```

This will check:
- All dependencies are installed
- Directory structure is correct
- Environment variables are set
- CUDA is available (if applicable)

## Running the Pipeline

### Option 1: Command Line (Full Pipeline)

```bash
# 1. Place 4+ input images in data/input/
#    (JPEG, PNG, BMP, TIFF formats supported)

# 2. Run the full pipeline
python main.py

# 3. Find results in data/output/
#    - audio_manifest.json: Audio source positions
#    - audio_visualization.html: Interactive 3D viewer
#    - unity_scene.json: Unity-compatible export
```

### Option 2: Jupyter Notebook (Interactive)

```bash
# Start Jupyter
jupyter notebook

# Open: notebooks/pipeline_demo.ipynb
# Follow the step-by-step demo
```

### Option 3: Step-by-Step (Manual Control)

```bash
# Skip certain steps if needed:
python main.py --skip-generation      # Use input images only
python main.py --skip-reconstruction  # Skip 3D reconstruction
python main.py --skip-audio          # Skip audio generation

# Custom configuration:
python main.py --config custom_config.yaml

# CPU-only mode:
python main.py --device cpu
```

## Configuration

Edit `config.yaml` to customize:

```yaml
image_generation:
  num_additional_views: 6  # Number of views to generate

reconstruction:
  use_colmap: true         # Enable COLMAP reconstruction
  use_depth_model: true    # Enable depth estimation
  
segmentation:
  batch_size: 2            # Batch size for VLM
  quantization: "bf16"     # bf16, fp16, fp32, int8
  
audio:
  clip_duration: 4.0       # Audio clip length in seconds
  sample_rate: 44100       # Audio sample rate
```

## Example Workflow

### 1. Prepare Input Images

Place 4 images of your scene from different viewpoints in `data/input/`:
- `view1.jpg` - Front view
- `view2.jpg` - Right side
- `view3.jpg` - Left side  
- `view4.jpg` - Top/back view

**Tips:**
- Use consistent lighting
- Overlap between views helps reconstruction
- Avoid motion blur
- Higher resolution is better (but slower)

### 2. Run Pipeline

```bash
python main.py
```

The pipeline will:
1. Load your 4 images
2. Generate 6 additional viewpoints
3. Reconstruct the 3D scene
4. Detect and label objects
5. Generate foley audio for each object
6. Position audio in 3D space

**Estimated time:** 15-30 minutes depending on hardware and number of objects

### 3. View Results

Open in browser:
- `data/output/audio_visualization.html` - Interactive 3D viewer

Check files:
- `data/output/audio_manifest.json` - All audio source data
- `data/audio/*.wav` - Individual audio clips
- `data/segmentation/*.json` - Object detection results
- `data/reconstruction/` - 3D reconstruction data

## Troubleshooting

### "CUDA out of memory"

**Solution 1:** Reduce batch size in `config.yaml`:
```yaml
segmentation:
  batch_size: 1
```

**Solution 2:** Use CPU mode:
```bash
python main.py --device cpu
```

**Solution 3:** Use quantization:
```yaml
segmentation:
  quantization: "int8"  # More aggressive quantization
```

### "COLMAP reconstruction failed"

This can happen with:
- Too few images (< 4)
- Poor feature matching (low texture scenes)
- Inconsistent lighting

**Solution:** Pipeline will continue with depth estimation only. Results may be less accurate but still functional.

### "MMAudio API timeout"

The HuggingFace Space may be busy or down.

**Solution:** Wait and retry, or implement local MMAudio inference (see Advanced Usage).

### Import errors

**Solution:** Reinstall dependencies:
```bash
pip install --force-reinstall -r requirements.txt
```

## Performance Optimization

### For Faster Processing

1. **Reduce image resolution** in `config.yaml`:
   ```yaml
   image_generation:
     output_size: [512, 512]  # Default: [1024, 1024]
   ```

2. **Skip dense reconstruction**:
   ```yaml
   reconstruction:
     colmap:
       dense_reconstruction: false
   ```

3. **Limit objects processed**:
   Edit `main.py` to reduce `max_objects` in `generate_audio()` method

### For Better Quality

1. **Use more input views**: Place 6-8 images instead of 4

2. **Enable dense reconstruction**:
   ```yaml
   reconstruction:
     colmap:
       dense_reconstruction: true
   ```

3. **Use higher resolution**:
   ```yaml
   image_generation:
     output_size: [2048, 2048]
   ```

## Next Steps

- Check out `notebooks/pipeline_demo.ipynb` for detailed exploration
- Read `README.md` for architecture details
- Modify `config.yaml` to experiment with settings
- Export to Unity using `data/output/unity_scene.json`

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the full README.md
3. Check `pipeline.log` for detailed error messages
4. Verify setup with `python test_setup.py`

## License

MIT License - See LICENSE file for details
