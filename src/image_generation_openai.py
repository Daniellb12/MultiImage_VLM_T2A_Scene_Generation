"""Image Generation Module using OpenAI GPT Image 2 API

Prompting strategy mirrors the Nano Banana backend:
  Stage 1 — analyze_scene()
    Reuses gemini-2.5-flash (text-only) for the structured spatial report so
    that a Gemini API key is the only requirement for scene analysis.  If you
    prefer to use an OpenAI text model, pass text_model="gpt-4o".

  Stage 2 — generate_viewpoint()
    Uses client.images.generate() with a rich text prompt built from the
    Stage-1 scene description and the camera spec.  Passing reference images
    to the edit endpoint causes gpt-image-2 to reproduce them instead of
    synthesising a new viewpoint, so generation is text-driven.

    GPT Image 2 notes:
    - quality: "low" / "medium" / "high" / "auto" is passed directly.
    - size: e.g. "1024x1024", "1536x1024" is passed directly.
    - chain_views has no effect since generation is stateless text→image.
    - Returns base64-encoded PNG which is decoded to a numpy array.
"""

import base64
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from src.utils import save_rgb_jpeg_at_size

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Please install openai: pip install openai>=1.0.0")

logger = logging.getLogger(__name__)

_IMAGE_GEN_MODEL = "gpt-image-2"
_TEXT_MODEL_GEMINI = "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# Shared prompt helpers (identical to Nano Banana backend)
# ---------------------------------------------------------------------------

_ANALYSIS_PROMPT = """You are analyzing 4 photographs taken from the 4 corners of a
rectangular room, each facing diagonally inward toward the room's center.
Together they cover the full 360° interior.

Produce a structured spatial report with these exact sections:

1. ROOM ENVELOPE
   - Approximate dimensions and proportions (e.g., "roughly 5m x 4m x 2.7m high")
   - Wall colors/materials for each of the 4 walls (label as North/South/East/West
     based on consistent orientation across the 4 images)
   - Floor material and color
   - Ceiling material, color, and any features (beams, vents, fixtures)

2. FIXED ARCHITECTURAL FEATURES
   - Windows: which wall, size, shape, what's visible through them, light direction
   - Doors: which wall, style, open/closed, what's beyond
   - Built-ins, columns, alcoves, trim, moldings

3. FURNITURE AND OBJECTS (with precise spatial anchoring)
   For each significant object, specify:
   - What it is
   - Which wall it is against OR its position relative to room center
   - Its approximate footprint and height
   - Its color, material, and distinguishing details

4. LIGHTING
   - Primary light source(s) and direction
   - Color temperature (warm/neutral/cool, approximate Kelvin)
   - Shadow direction and softness
   - Any visible light fixtures (on/off state)

5. STYLE AND ATMOSPHERE
   - Design style (e.g., mid-century modern, industrial, traditional)
   - Overall mood, cleanliness, lived-in vs. staged feel
   - Photographic qualities: lens character, depth of field, color grading

Be concrete and specific. This description will be used to regenerate the same
room from new camera angles, so consistency of every detail matters."""

# Cardinal direction names for arc positions (every 45°, clockwise from North).
_ARC_COMPASS = [
    "North", "North-East", "East", "South-East",
    "South", "South-West", "West", "North-West",
]


def _min_views_for_overlap(hfov_deg: float = 65.0, target_overlap: float = 0.60) -> int:
    """Return the minimum number of arc positions for the requested frame overlap.

    For a ~65° HFOV and 60% overlap this evaluates to 12 views (30° spacing).

    Args:
        hfov_deg:       Horizontal field of view in degrees.
        target_overlap: Desired fractional overlap between adjacent frames (0–1).

    Returns:
        Minimum integer number of views for a full 360° arc.
    """
    import math
    half_hfov = math.radians(hfov_deg / 2.0)
    sin_half_step = math.tan(half_hfov) * (1.0 - target_overlap)
    sin_half_step = min(sin_half_step, 1.0)
    half_step = math.asin(sin_half_step)
    step_deg = math.degrees(2.0 * half_step)
    return math.ceil(360.0 / step_deg)


def _make_arc_view_specs(
    n: int,
    radius_fraction: float = 0.4,
) -> List[str]:
    """Generate n camera positions on a horizontal arc around the room centre.

    Adjacent views are spaced 360°/n apart and all point toward the room centre
    at standing eye height (1.6 m).

    For COLMAP and 3DGS to succeed, adjacent views must overlap ≥ 60%.
    With a ~65° HFOV lens this requires ≥ 12 views (30° spacing).
    Use ``_min_views_for_overlap()`` to compute the exact minimum.

    Args:
        n:                Number of views (≥ 1, no upper cap).
        radius_fraction:  Camera arc radius as fraction of shorter room dim.

    Returns:
        List of n camera position strings suitable for _build_view_prompt.
    """
    import math
    n = max(1, n)
    step = 360.0 / n
    specs = []
    for i in range(n):
        angle_deg = i * step
        angle_rad = math.radians(angle_deg)

        label_idx        = round(angle_deg / 45) % 8
        compass          = _ARC_COMPASS[label_idx]
        opposite_idx     = (label_idx + 4) % 8
        opposite_compass = _ARC_COMPASS[opposite_idx]

        x_frac = math.sin(angle_rad) * radius_fraction
        z_frac = -math.cos(angle_rad) * radius_fraction

        prev_angle = (angle_deg - step) % 360
        next_angle = (angle_deg + step) % 360

        spec = (
            f"Camera on a circular arc around the room centre, positioned at "
            f"{angle_deg:.1f}° clockwise from North (approximately {compass} side). "
            f"Approximate offset from centre: {x_frac:+.2f}× room-width East, "
            f"{z_frac:+.2f}× room-depth North (arc radius ≈ {radius_fraction*100:.0f}% "
            f"of the shorter room dimension). "
            f"Camera height: standing eye level (~1.6 m). "
            f"Pointing directly toward the room centre — the {opposite_compass} wall "
            f"fills the far background of the frame. "
            f"Focal length: ~35 mm equivalent (~65° HFOV). "
            f"IMPORTANT: This view must overlap approximately {step:.0f}°-worth of scene "
            f"content with its neighbours at {prev_angle:.1f}° and {next_angle:.1f}°. "
            f"Do NOT rotate or reframe away from the room centre — the angular spacing "
            f"between consecutive views is designed to give COLMAP and 3DGS at least "
            f"60% frame overlap for successful feature matching."
        )
        specs.append(spec)
    return specs


# Legacy fixed specifications kept for backward-compatibility when
# view_strategy = "fixed" is set in config.yaml.
_FIXED_VIEW_SPECIFICATIONS: List[str] = [
    (
        "Camera at the geometric center of the room at standing eye height (~1.6 m), "
        "facing the North wall directly (perpendicular to it). Standard ~35 mm equivalent "
        "focal length. The North wall fills most of the frame; the East and West walls "
        "recede symmetrically at the left and right edges. The floor and ceiling are "
        "visible in the lower and upper thirds."
    ),
    (
        "Camera at the geometric center of the room at standing eye height (~1.6 m), "
        "facing the South wall directly. Same focal length as the reference images. "
        "Mirror composition of the North-wall view: South wall centered, East/West walls "
        "recede to the sides."
    ),
    (
        "Camera at the geometric center of the room at standing eye height (~1.6 m), "
        "facing the East wall directly. The East wall fills the frame; North and South "
        "walls recede symmetrically left and right."
    ),
    (
        "Camera at the geometric center of the room at standing eye height (~1.6 m), "
        "facing the West wall directly. The West wall fills the frame; North and South "
        "walls recede symmetrically left and right."
    ),
    (
        "Camera positioned near the ceiling (approx. 0.3 m below it) at the center of "
        "the room, tilted downward at approximately 60 degrees toward the floor. Wide "
        "~24 mm equivalent focal length. The overhead view reveals the full floor plan, "
        "furniture arrangement, and how objects are positioned relative to each other "
        "and the walls."
    ),
    (
        "Camera at seated height (~1.1 m) in the geometric center of the room, facing "
        "the wall that contains the primary window or brightest light source. The lower "
        "vantage point shows how natural light falls across the floor and lower surfaces "
        "of furniture, with longer shadows than in the standing-height shots."
    ),
    (
        "Camera in the North-East corner of the room at standing eye height (~1.6 m), "
        "pointing diagonally toward the South-West corner (the opposite corner). This "
        "captures the full room depth on a diagonal axis, with two walls visible on the "
        "left and right, and the far corner in the center of the frame."
    ),
    (
        "Camera in the North-West corner of the room at standing eye height (~1.6 m), "
        "pointing diagonally toward the South-East corner. Similar composition to the "
        "previous shot but from the opposite corner, providing coverage of the other "
        "diagonal axis of the room."
    ),
]


def _build_view_prompt(
    scene_description: str,
    view_spec: str,
    prior_specs: Optional[List[str]] = None,
) -> str:
    if prior_specs:
        prior_block = (
            "\nALREADY-COVERED ANGLES (produce a clearly DIFFERENT composition):\n"
            + "\n".join(f"  - {s}" for s in prior_specs)
            + "\n"
        )
    else:
        prior_block = ""

    return f"""You are a photorealistic interior-photography renderer. \
Generate a single high-quality photograph of the room described below, \
taken from the specified camera position. Every detail in the scene \
description MUST appear exactly as written — do not add, remove, or \
change any object, material, color, or lighting condition.

SCENE DESCRIPTION (authoritative — reproduce faithfully):
{scene_description}

CAMERA POSITION FOR THIS IMAGE:
{view_spec}
{prior_block}
RENDERING RULES:
- Photorealistic, as if shot with a professional camera in the same \
session as the reference description.
- Exact same room dimensions, wall colors/materials, floor, ceiling.
- Every piece of furniture in the exact position described, shown from \
this new angle.
- Same lighting direction, color temperature, and shadow softness.
- Same windows, same view through them, same time of day.
- Do NOT copy or reproduce any existing photograph — synthesize a fresh \
image from the description and camera spec above.

OUTPUT: A single photorealistic photograph from the specified camera \
position."""


def _numpy_to_png_bytes(img: np.ndarray) -> bytes:
    """Convert a numpy uint8 image to PNG bytes."""
    img_u8 = img if img.dtype == np.uint8 else (img * 255).astype(np.uint8)
    buf = BytesIO()
    Image.fromarray(img_u8).save(buf, format="PNG")
    return buf.getvalue()


class OpenAIImageGenerator:
    """Generate additional viewpoints using the OpenAI GPT Image 2 API.

    Workflow
    --------
    1. Call ``analyze_scene(images)`` — uses Gemini 2.5 Flash (text-only) to
       produce a structured room report.  Pass ``text_model="gpt-4o"`` to use
       an OpenAI text model instead (requires the OpenAI key to have vision
       access).
    2. Call ``generate_additional_views(images, scene_description=...)`` to
       produce N new views via the GPT Image 2 edits endpoint, which accepts
       multiple reference images alongside the view prompt.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = _IMAGE_GEN_MODEL,
        quality: str = "medium",
        size: str = "1024x1024",
        text_model: str = "gemini",
        gemini_api_key: Optional[str] = None,
        view_strategy: str = "arc",
        arc_radius_fraction: float = 0.4,
    ):
        """
        Args:
            api_key:              OpenAI API key. Reads OPENAI_API_KEY env var if None.
            model:                GPT Image model name (default: gpt-image-2).
            quality:              Output quality — "low", "medium", "high", or "auto".
            size:                 Output dimensions, e.g. "1024x1024", "1536x1024".
            text_model:           "gemini" to use Gemini 2.5 Flash for scene analysis
                                  (requires GEMINI_API_KEY), or "gpt-4o" to use OpenAI.
            gemini_api_key:       Gemini API key when text_model="gemini".
                                  Reads GEMINI_API_KEY env var if None.
            view_strategy:        "arc" (COLMAP-compatible circular arc, default) or
                                  "fixed" (legacy cardinal/corner positions).
            arc_radius_fraction:  For the "arc" strategy, camera arc radius as a
                                  fraction of the shorter room dimension (default 0.4).
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Set it in .env or pass as argument."
            )

        self.model               = model
        self.quality             = quality
        self.size                = size
        self.text_model          = text_model
        self.view_strategy       = view_strategy
        self.arc_radius_fraction = arc_radius_fraction
        self.client = OpenAI(api_key=self.api_key)

        # Scene analysis client — Gemini by default
        if text_model == "gemini":
            gemini_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
            if not gemini_key:
                raise ValueError(
                    "GEMINI_API_KEY not found. Required for scene analysis when "
                    "text_model='gemini'. Set it in .env or pass gemini_api_key=."
                )
            try:
                from google import genai as ggenai
                from google.genai import types as gtypes
                self._genai_client = ggenai.Client(
                    api_key=gemini_key,
                    http_options=gtypes.HttpOptions(api_version="v1beta"),
                )
                self._gtypes = gtypes
            except ImportError:
                raise ImportError(
                    "google-genai is required for scene analysis. "
                    "Install it: pip install google-genai  "
                    "Or pass text_model='gpt-4o' to use OpenAI for analysis."
                )
        else:
            self._genai_client = None
            self._gtypes = None

        logger.info(
            f"Initialized OpenAIImageGenerator — model={model}, "
            f"quality={quality}, size={size}, text_model={text_model}"
        )

    # ------------------------------------------------------------------
    # Stage 1: Scene analysis
    # ------------------------------------------------------------------

    def analyze_scene(self, images: List[np.ndarray]) -> str:
        """
        Send all input images to the text model and return a structured spatial
        report to inject into every generation prompt.

        Args:
            images: The 4 corner images.

        Returns:
            Multi-section scene description string, or a fallback message.
        """
        logger.info(f"Analyzing scene with {len(images)} images (text_model={self.text_model})")

        if self.text_model == "gemini":
            return self._analyze_scene_gemini(images)
        return self._analyze_scene_gpt4o(images)

    def _analyze_scene_gemini(self, images: List[np.ndarray]) -> str:
        gtypes = self._gtypes
        parts = [gtypes.Part.from_text(text=_ANALYSIS_PROMPT)]
        for img in images:
            img_u8 = img if img.dtype == np.uint8 else (img * 255).astype(np.uint8)
            buf = BytesIO()
            Image.fromarray(img_u8).save(buf, format="PNG")
            parts.append(gtypes.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"))

        try:
            response = self._genai_client.models.generate_content(
                model=_TEXT_MODEL_GEMINI,
                contents=parts,
                config=gtypes.GenerateContentConfig(temperature=0.2),
            )
            text = response.text if response.text else ""
            if text:
                logger.info(f"Scene analysis complete ({len(text)} chars)")
                return text
            logger.warning("Scene analysis returned empty text")
            return "Scene analysis unavailable"
        except Exception as e:
            logger.error(f"Error analyzing scene (Gemini): {e}")
            return "Scene analysis unavailable"

    def _analyze_scene_gpt4o(self, images: List[np.ndarray]) -> str:
        content = [{"type": "text", "text": _ANALYSIS_PROMPT}]
        for img in images:
            png_bytes = _numpy_to_png_bytes(img)
            b64 = base64.b64encode(png_bytes).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
            })

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": content}],
                temperature=0.2,
                max_tokens=2048,
            )
            text = response.choices[0].message.content or ""
            if text:
                logger.info(f"Scene analysis complete ({len(text)} chars)")
                return text
            logger.warning("Scene analysis returned empty text")
            return "Scene analysis unavailable"
        except Exception as e:
            logger.error(f"Error analyzing scene (GPT-4o): {e}")
            return "Scene analysis unavailable"

    # ------------------------------------------------------------------
    # Stage 2: Single-view generation
    # ------------------------------------------------------------------

    def generate_viewpoint(
        self,
        reference_images: List[np.ndarray],
        view_spec: str,
        scene_description: str,
        output_size: Tuple[int, int] = (1024, 1024),
        prior_specs: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Generate one new view using the OpenAI images.generate endpoint.

        GPT Image 2 has strong instruction-following from text alone, so we
        anchor generation entirely through the detailed scene description
        produced in Stage 1 rather than passing raw image bytes (which causes
        the model to reproduce the reference images instead of synthesising a
        new viewpoint).

        Args:
            reference_images: Not used for generation; kept for interface
                              compatibility with NanoBananaImageGenerator.
            view_spec:         Camera-explicit position description.
            scene_description: Full Stage-1 structured report.
            output_size:       (width, height) to resize the output to.
            prior_specs:       Already-generated angle descriptions injected
                               into the prompt to prevent duplicates.

        Returns:
            Generated image as uint8 numpy array (H, W, 3).
        """
        logger.info(f"Generating view: {view_spec[:80]}...")

        prompt = _build_view_prompt(scene_description, view_spec, prior_specs=prior_specs)

        try:
            result = self.client.images.generate(
                model=self.model,
                prompt=prompt,
                quality=self.quality,
                size=self.size,
                n=1,
            )

            b64 = result.data[0].b64_json
            image_bytes = base64.b64decode(b64)
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            image = image.resize(output_size, Image.Resampling.LANCZOS)
            arr = np.array(image)
            logger.info(f"Generated image shape: {arr.shape}")
            return arr

        except Exception as e:
            logger.error(f"Error generating viewpoint: {e}")
            raise

    # ------------------------------------------------------------------
    # Batch generation
    # ------------------------------------------------------------------

    def generate_additional_views(
        self,
        input_images: List[np.ndarray],
        scene_description: str,
        num_views: int = 6,
        output_dir: str = "data/Nano_banana_output_images",
        output_size: Tuple[int, int] = (1024, 1024),
        chain_views: bool = True,
        use_cache: bool = True,
        input_paths: Optional[List[str]] = None,
    ) -> List[np.ndarray]:
        """
        Generate ``num_views`` new views and write all images to ``output_dir``
        as JPEG files ready for COLMAP.

        ``chain_views=True`` feeds each generated view back into the reference
        pool for the next call, giving the model progressively more context.

        See NanoBananaImageGenerator.generate_additional_views for full
        parameter documentation.
        """
        if len(input_images) < 4:
            logger.warning(
                f"Expected 4 corner images, got {len(input_images)}. "
                "Spatial anchoring will be weaker."
            )

        # Choose view strategy from config (arc = COLMAP-compatible, fixed = legacy)
        _view_strategy   = getattr(self, 'view_strategy', 'arc')
        _radius_fraction = getattr(self, 'arc_radius_fraction', 0.4)
        if _view_strategy == 'fixed':
            _all_specs = _FIXED_VIEW_SPECIFICATIONS
            num_views  = min(num_views, len(_all_specs))
            specs      = _all_specs[:num_views]
        else:
            specs = _make_arc_view_specs(num_views, radius_fraction=_radius_fraction)

        os.makedirs(output_dir, exist_ok=True)
        out_path = Path(output_dir)

        _IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

        # ── Cache check ───────────────────────────────────────────────────────
        if use_cache:
            existing = sorted(
                p for p in out_path.iterdir()
                if p.suffix.lower() in _IMAGE_EXTS
            )
            if existing:
                logger.info(
                    f"Found {len(existing)} existing image(s) in '{output_dir}' — "
                    "skipping API calls and loading from disk."
                )
                cached: List[np.ndarray] = []
                for p in existing:
                    with Image.open(p) as im:
                        pil = im.convert("RGB")
                        if pil.size != output_size:
                            pil = pil.resize(output_size, Image.Resampling.LANCZOS)
                            pil.save(p, format="JPEG", quality=95)
                            logger.info(
                                f"Resized cached image to {output_size[0]}×{output_size[1]}: {p.name}"
                            )
                        img = np.array(pil)
                    cached.append(img)
                    logger.info(f"Loaded cached view: {p.name}")
                return cached

        # ── Copy originals resized to ``output_size`` (match generated views) ─
        for i, img_arr in enumerate(input_images):
            if input_paths and i < len(input_paths):
                stem = Path(input_paths[i]).stem
            else:
                stem = f"input_view_{i:02d}"
            dest = out_path / f"{stem}.jpg"
            save_rgb_jpeg_at_size(img_arr, dest, output_size, quality=95)
            logger.info(f"Saved input view (resized to {output_size[0]}×{output_size[1]}): {dest.name}")

        # ── Generate and save new views as JPEG ──────────────────────────────
        generated: List[np.ndarray] = []
        reference_pool = list(input_images)
        completed_specs: List[str] = []

        for i, spec in enumerate(specs):
            logger.info(f"Generating view {i + 1}/{num_views}")
            try:
                img = self.generate_viewpoint(
                    reference_images=reference_pool,
                    view_spec=spec,
                    scene_description=scene_description,
                    output_size=output_size,
                    prior_specs=completed_specs if completed_specs else None,
                )

                generated.append(img)
                completed_specs.append(spec)

                dest = out_path / f"generated_view_{i:02d}.jpg"
                save_rgb_jpeg_at_size(img, dest, output_size, quality=95)
                logger.info(f"Saved generated view to: {dest.name}")

                if chain_views:
                    reference_pool.append(img)

            except Exception as e:
                logger.error(f"Failed to generate view {i}: {e}")
                continue

        logger.info(
            f"Generated {len(generated)}/{num_views} views. "
            f"All images saved to '{output_dir}'."
        )
        return generated
