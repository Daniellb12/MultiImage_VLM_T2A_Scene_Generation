# 🎵 Get Started with Multi-Modal Scene Audio Pipeline

Welcome! This guide will get you up and running in **under 10 minutes**.

---

## 📋 Prerequisites

- Python 3.9 or higher
- 8GB+ GPU (for best performance)
- ~50GB free disk space
- Internet connection

---

## 🚀 Quick Setup (3 Steps)

### Step 1: Install Dependencies

```bash
# Create and activate virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate

# Linux/Mac:
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### Step 2: Configure API Keys

```bash
# Copy environment template
copy .env.example .env   # Windows
# OR
cp .env.example .env     # Linux/Mac

# Edit .env and add your Gemini API key
# Get it from: https://makersuite.google.com/app/apikey
```

Your `.env` should look like:
```
GEMINI_API_KEY=your_actual_api_key_here
HF_TOKEN=optional_huggingface_token
DEVICE=cuda
```

### Step 3: Verify Setup

```bash
python test_setup.py
```

You should see all green checkmarks ✓

---

## 🎬 Run Your First Pipeline

### 1. Prepare Input Images

Place **4 or more images** of your scene in `data/input/`:
- Different viewpoints of the same scene
- JPEG or PNG format
- Consistent lighting
- Clear objects

Example:
```
data/input/
  ├── view1.jpg  (front)
  ├── view2.jpg  (right side)
  ├── view3.jpg  (left side)
  └── view4.jpg  (back or top)
```

### 2. Run the Pipeline

```bash
python main.py
```

That's it! The pipeline will:
1. ✨ Generate 6 additional viewpoints
2. 🏗️ Reconstruct the 3D scene
3. 👁️ Detect and label objects
4. 🎵 Generate foley audio for each object
5. 📍 Position audio in 3D space

**Expected time:** 15-30 minutes

### 3. View Results

**Interactive 3D Viewer:**
```
Open in browser: data/output/audio_visualization.html
```

**All Results:**
```
data/output/
  ├── audio_manifest.json         (spatial audio data)
  ├── audio_visualization.html    (interactive viewer)
  ├── unity_scene.json           (Unity export)
  └── pipeline_summary.json      (statistics)

data/audio/
  ├── obj_001.wav  (audio for object 1)
  ├── obj_002.wav  (audio for object 2)
  └── ...
```

---

## 🎯 What You Get

### Visual Outputs
- 📊 3D point cloud of your scene
- 🎨 Object detection visualizations
- 🌐 Interactive HTML viewer

### Audio Outputs
- 🎵 Individual WAV files per object
- 📍 3D spatial positions
- 🎚️ Intensity calculations
- 🎮 Unity-ready export

### Data Outputs
- 📋 Complete JSON manifest
- 📸 Camera poses
- 🏷️ Object labels and categories
- 📐 Depth maps

---

## 🎓 Next Steps

### Learn More
- 📖 Read `QUICKSTART.md` for detailed workflows
- 📚 Check `ARCHITECTURE.md` for technical details
- 🔬 Explore `notebooks/pipeline_demo.ipynb` for step-by-step demo

### Customize
Edit `config.yaml` to:
- Change number of generated views
- Adjust audio duration
- Enable/disable dense reconstruction
- Modify model settings

### Experiment
```bash
# Skip image generation (faster)
python main.py --skip-generation

# CPU-only mode
python main.py --device cpu

# Custom config
python main.py --config my_config.yaml
```

---

## 🆘 Troubleshooting

### "CUDA out of memory"
```bash
# Use CPU mode
python main.py --device cpu

# Or reduce batch size in config.yaml
segmentation:
  batch_size: 1
```

### "COLMAP failed"
Don't worry! The pipeline continues with depth estimation only.
Results will still work, just slightly less accurate.

### "Module not found"
```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

### "API key not found"
Make sure you:
1. Created `.env` file (copy from `.env.example`)
2. Added your actual Gemini API key
3. No extra spaces around the = sign

---

## 💡 Tips for Best Results

### Input Images
- ✅ Use 4-8 images from different angles
- ✅ Ensure good overlap between views
- ✅ Keep lighting consistent
- ✅ Higher resolution is better (but slower)
- ❌ Avoid motion blur
- ❌ Don't use images from different times

### Performance
- 🚀 Use GPU for 5-10x speedup
- 🚀 Skip dense reconstruction for faster results
- 🚀 Limit to 10-15 objects for quick testing
- 💰 Reduce image resolution in config for speed

### Quality
- 🎯 More input views = better reconstruction
- 🎯 Enable dense reconstruction for detailed models
- 🎯 Use higher resolution for better segmentation
- 🎯 Textured objects work better than plain surfaces

---

## 📊 Example Scenes

### Good Scene Examples
- ✅ Living room with furniture
- ✅ Kitchen with appliances
- ✅ Outdoor garden with plants
- ✅ Office space with equipment
- ✅ Street scene with vehicles

### Challenging Scenes
- ⚠️ Empty rooms (few features)
- ⚠️ Reflective surfaces (mirrors, glass)
- ⚠️ Repetitive patterns (tiles, wallpaper)
- ⚠️ Very large or very small scenes

---

## 🎮 Export to Unity

1. Run the pipeline
2. Find `data/output/unity_scene.json`
3. Import into Unity:
   ```csharp
   // Unity script to load audio scene
   string json = File.ReadAllText("unity_scene.json");
   AudioScene scene = JsonUtility.FromJson<AudioScene>(json);
   
   foreach (var source in scene.audioSources) {
       GameObject obj = new GameObject(source.name);
       AudioSource audio = obj.AddComponent<AudioSource>();
       audio.clip = Resources.Load<AudioClip>(source.audioClipPath);
       obj.transform.position = source.position;
       audio.volume = source.volume;
       audio.spatialBlend = 1.0f;
   }
   ```

---

## 📚 Documentation Index

| Document | Purpose |
|----------|---------|
| `GET_STARTED.md` | **You are here** - Quick start |
| `README.md` | Project overview |
| `QUICKSTART.md` | Detailed setup and workflows |
| `ARCHITECTURE.md` | Technical architecture |
| `PROJECT_SUMMARY.md` | Complete feature list |
| `COMPLETION_REPORT.md` | Implementation details |

---

## 🤝 Need Help?

1. Check `QUICKSTART.md` for detailed troubleshooting
2. Review `pipeline.log` for error details
3. Run `python test_setup.py` to verify installation
4. Check GitHub issues (if published)

---

## ✨ Project Highlights

- 🎨 **Multi-Modal AI:** Combines vision, language, and audio
- 🏗️ **Production-Ready:** Robust error handling
- 🚀 **High Performance:** Optimized for consumer GPUs
- 📖 **Well-Documented:** 40KB+ of documentation
- 🔧 **Highly Configurable:** YAML-based settings
- 🎯 **Multiple Outputs:** JSON, HTML, Unity, PLY

---

## 🎉 Ready to Go!

You're all set! Run your first pipeline:

```bash
# 1. Activate environment
venv\Scripts\activate   # Windows
source venv/bin/activate  # Linux/Mac

# 2. Place images in data/input/

# 3. Run!
python main.py

# 4. Open results
start data\output\audio_visualization.html   # Windows
open data/output/audio_visualization.html    # Mac
xdg-open data/output/audio_visualization.html  # Linux
```

**Enjoy creating spatial audio scenes!** 🎵🎨✨

---

*For detailed instructions, see `QUICKSTART.md`*  
*For technical details, see `ARCHITECTURE.md`*  
*For complete feature list, see `PROJECT_SUMMARY.md`*
