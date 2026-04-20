# Multi-Modal Scene Audio Generation Pipeline

A pipeline that processes multi-view images to create 3D scene reconstructions with spatially-positioned foley audio.

## Overview

This project takes 4 input images from different viewpoints and:
1. Generates additional viewpoints using Gemini Nano Banana API
2. Reconstructs the 3D scene using COLMAP and depth estimation
3. Identifies and segments objects using Qwen2.5-VL
4. Generates foley audio for each object using MMAudio
5. Positions audio spatially in the 3D scene

## Setup

### Prerequisites

- Python 3.9+
- NVIDIA GPU with 8GB+ VRAM (recommended)
- CUDA toolkit (for GPU acceleration)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd 848M_Final_Project
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

### API Keys

- **Gemini API Key**: Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
- **HuggingFace Token** (optional): Get from [HuggingFace Settings](https://huggingface.co/settings/tokens)

## Usage

### Basic Usage

```bash
python main.py --input data/input/
```

### With Custom Configuration

```bash
python main.py --input data/input/ --config custom_config.yaml
```

### Configuration

Edit `config.yaml` to customize:
- Number of generated views
- 3D reconstruction settings
- Model parameters
- Audio generation settings

## Project Structure

```
848M_Final_Project/
├── main.py                  # Main orchestration script
├── config.yaml              # Configuration file
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── src/                    # Source code modules
├── data/                   # Data directories
└── notebooks/              # Jupyter notebooks
```

## Hardware Requirements

### Minimum
- GPU: 8GB VRAM
- RAM: 16GB
- Storage: 50GB

### Recommended
- GPU: NVIDIA RTX 3090/4090 (24GB VRAM)
- RAM: 32GB
- Storage: 100GB SSD

## License

MIT License

## Acknowledgments

- Gemini API for image generation
- COLMAP for 3D reconstruction
- Qwen2.5-VL for scene understanding
- MMAudio for audio generation
