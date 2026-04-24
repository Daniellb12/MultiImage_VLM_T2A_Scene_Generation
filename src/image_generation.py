"""Image Generation Module using Gemini Nano Banana API

Prompting strategy:
  Stage 1 — analyze_scene()
    Sends all 4 corner images to gemini-2.5-flash (text-only) and returns a
    richly-structured spatial report: room envelope, fixed architecture,
    furniture positions, lighting, and photographic style.

  Stage 2 — generate_viewpoint()
    Each generation call receives:
      - the full Stage-1 scene report (injected verbatim into the prompt)
      - all 4 original corner images as visual anchors
      - optionally, any previously-generated views (sequential conditioning)
      - a camera-explicit view spec anchored to room geometry (not floating
        directions like "45 degrees to the right")

  Sequential conditioning (chain_views=True in generate_additional_views)
    View N is generated with views 0..N-1 appended as additional references,
    so each new frame has more spatial anchors and drift is minimised.
"""

import os
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
from io import BytesIO

try:
    from google import genai
    from google.genai import types
except ImportError:
    raise ImportError("Please install google-genai: pip install google-genai")

logger = logging.getLogger(__name__)

# Nano Banana 2 image-generation model (preview — requires v1beta endpoint)
_IMAGE_GEN_MODEL = "gemini-3.1-flash-image-preview"
# Fast text model for the analysis step (no image-gen quota used)
_TEXT_MODEL = "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# Structured analysis prompt (Stage 1)
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


# ---------------------------------------------------------------------------
# View specifications anchored to room geometry (Stage 2)
# ---------------------------------------------------------------------------

# Each spec tells the model exactly where the camera is, where it points, and
# what the expected framing looks like. All positions are relative to the room
# (walls, centre) — not to a floating "main viewpoint".
_VIEW_SPECIFICATIONS: List[str] = [
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
    """
    Construct the full generation prompt for a single new view.

    Injects the structured scene report verbatim so the model has a complete
    textual anchor alongside the visual reference images.

    Args:
        scene_description: Stage-1 room report.
        view_spec:         Camera description for the view to generate.
        prior_specs:       Camera descriptions of views already generated
                           (used to enforce unique angles).
    """
    # Build the "already covered" section only when there are prior views
    if prior_specs:
        prior_block = (
            "\nALREADY-COVERED ANGLES (DO NOT REPRODUCE ANY OF THESE):\n"
            + "\n".join(f"  - {s}" for s in prior_specs)
            + "\n\nThe image you produce MUST be visually distinct from every "
            "already-covered angle listed above. Different framing, different "
            "depth, different wall emphasis. If the new camera position would "
            "produce a composition that looks nearly identical to any prior view, "
            "commit fully to the specified position and exaggerate its unique "
            "perspective rather than defaulting to a generic wide shot.\n"
        )
    else:
        prior_block = ""

    return f"""Generate a photograph of the EXACT SAME ROOM shown in the reference \
images, from a new camera position. This is novel view synthesis — every \
architectural feature, furniture piece, material, color, and lighting condition \
must be perfectly consistent with the reference images. Do not invent new objects, \
do not change existing ones, do not alter the style.

REFERENCE SCENE DESCRIPTION:
{scene_description}

NEW CAMERA POSITION:
{view_spec}
{prior_block}
STRICT CONSISTENCY REQUIREMENTS:
- Same room dimensions, same walls, same floor, same ceiling
- Same furniture in the same positions (shown from this new angle)
- Same wall colors and materials as described above
- Same lighting direction, color temperature, and time of day
- Same windows showing the same view outside
- Same photographic style: lens, exposure, color grading, depth of field
- Objects that were visible in the reference images must appear identical \
when visible from this new angle
- Objects occluded from this new angle should simply not appear; do not \
replace them with different objects

NO-DUPLICATE RULE:
- This image MUST show a clearly different camera position and framing from \
every reference image provided and every already-covered angle listed above.
- Do NOT produce a composition that could be mistaken for an existing view.
- The specified camera position is mandatory — do not substitute a different angle.

OUTPUT: A single photorealistic photograph from the specified camera position, \
as if taken with the same camera as the reference images in the same session."""


def _images_to_parts(images: List[np.ndarray]) -> List[types.Part]:
    """Convert a list of numpy images to google-genai Part objects."""
    parts = []
    for img in images:
        img_u8 = img if img.dtype == np.uint8 else (img * 255).astype(np.uint8)
        buf = BytesIO()
        Image.fromarray(img_u8).save(buf, format="PNG")
        parts.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"))
    return parts


class ImageGenerator:
    """Generate additional viewpoints using Gemini Nano Banana API.

    Workflow
    --------
    1. Call ``analyze_scene(images)`` with the 4 corner images to get a rich
       structured room description (uses the text model, no image-gen quota).
    2. Call ``generate_additional_views(images, scene_description=...)`` to
       produce N new views. Each call injects the room description + all 4
       originals + any previously generated views as visual anchors.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = _IMAGE_GEN_MODEL,
    ):
        """
        Args:
            api_key: Gemini API key. Reads GEMINI_API_KEY env var if None.
            model:   Image-generation model name.
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Set it in .env or pass as argument."
            )

        self.model = model
        # v1beta is required for preview models (gemini-3.1-flash-image-preview).
        # Stable models like gemini-2.5-flash are also accessible via v1beta,
        # so the text analysis step is unaffected.
        self.client = genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(api_version="v1beta"),
        )
        logger.info(f"Initialized ImageGenerator with model: {model}")

    # ------------------------------------------------------------------
    # Stage 1: Scene analysis
    # ------------------------------------------------------------------

    def analyze_scene(self, images: List[np.ndarray]) -> str:
        """
        Send all input images to the text model and return a richly structured
        spatial report that will be injected into every generation prompt.

        Uses ``gemini-2.5-flash`` (text-only output) so the image-gen quota
        is not consumed here.

        Args:
            images: The 4 corner images (all are sent for maximum coverage).

        Returns:
            Multi-section scene description string, or a fallback message.
        """
        logger.info(f"Analyzing scene with {len(images)} images")

        content_parts: List[types.Part] = [types.Part.from_text(text=_ANALYSIS_PROMPT)]
        content_parts.extend(_images_to_parts(images))  # send all 4 corners

        try:
            response = self.client.models.generate_content(
                model=_TEXT_MODEL,
                contents=content_parts,
                config=types.GenerateContentConfig(temperature=0.2),
            )

            text = response.text if response.text else ""
            if text:
                logger.info(f"Scene analysis complete ({len(text)} chars)")
                logger.debug(f"Scene analysis preview: {text[:300]}...")
                return text

            logger.warning("Scene analysis returned empty text")
            return "Scene analysis unavailable"

        except Exception as e:
            logger.error(f"Error analyzing scene: {e}")
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
        Generate one new view.

        Args:
            reference_images: Visual anchors (original 4 + any prior generated
                              views for sequential conditioning).
            view_spec:         Camera-explicit position description anchored to
                               room geometry.
            scene_description: Full Stage-1 structured report to inject into prompt.
            output_size:       (width, height) to resize the generated image to.
            prior_specs:       Camera descriptions already generated, injected
                               into the prompt to prevent duplicate angles.

        Returns:
            Generated image as uint8 numpy array (H, W, 3).
        """
        logger.info(f"Generating view: {view_spec[:80]}...")

        prompt = _build_view_prompt(scene_description, view_spec, prior_specs=prior_specs)

        content_parts: List[types.Part] = [types.Part.from_text(text=prompt)]
        content_parts.extend(_images_to_parts(reference_images))

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=content_parts,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                    temperature=0.4,  # lower = more faithful to references
                ),
            )

            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "inline_data") and part.inline_data:
                        image = Image.open(BytesIO(part.inline_data.data)).convert("RGB")
                        image = image.resize(output_size, Image.Resampling.LANCZOS)
                        arr = np.array(image)
                        logger.info(f"Generated image shape: {arr.shape}")
                        return arr

            raise ValueError("No image found in API response")

        except Exception as e:
            logger.error(f"Error generating viewpoint: {e}")
            raise

    # ------------------------------------------------------------------
    # Batch generation with optional sequential conditioning
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
        Generate ``num_views`` new views of the room and consolidate all images
        (originals + generated) into ``output_dir`` as JPEG files ready for COLMAP.

        Cache behaviour (``use_cache=True``)
        -------------------------------------
        If ``output_dir`` already contains **any** image files from a previous
        run they are loaded from disk and returned immediately — no API calls are
        made.  Pass ``use_cache=False`` to force fresh generation.

        Output layout
        -------------
        ``input_view_00.jpg`` … ``input_view_NN.jpg``   ← original corner images
        ``generated_view_00.jpg`` … ``generated_view_NN.jpg``  ← Gemini output

        JPEG is used throughout because COLMAP is more reliable with JPEG than PNG.

        Args:
            input_images:  The original corner images (numpy uint8 arrays).
            scene_description: Stage-1 structured room report from analyze_scene().
            num_views:     How many additional views to generate (max 8).
            output_dir:    Folder where all JPEGs are written.
                           Defaults to ``data/Nano_banana_output_images``.
            output_size:   (width, height) for each generated image.
            chain_views:   If True, feed each generated view back as a reference
                           for the next call (reduces cross-view drift).
            use_cache:     If True, skip generation when images already exist.
            input_paths:   Original file paths of the corner images (used to
                           derive sensible filenames; optional).

        Returns:
            List of **generated** images as uint8 numpy arrays (inputs not included).
        """
        if len(input_images) < 4:
            logger.warning(
                f"Expected 4 corner images, got {len(input_images)}. "
                "Spatial anchoring will be weaker."
            )

        num_views = min(num_views, len(_VIEW_SPECIFICATIONS))
        specs = _VIEW_SPECIFICATIONS[:num_views]

        os.makedirs(output_dir, exist_ok=True)
        out_path = Path(output_dir)

        _IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

        # ── Cache check ───────────────────────────────────────────────────────
        # Load any existing images regardless of their original filenames.
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
                    img = np.array(Image.open(p).convert("RGB"))
                    cached.append(img)
                    logger.info(f"Loaded cached view: {p.name}")
                # Return only the non-input portion so the caller gets
                # the same type of value as a fresh generation run.
                # (All images are in the folder; Stage 3 reads the folder directly.)
                return cached

        # ── Copy original input images as JPEG ───────────────────────────────
        for i, img_arr in enumerate(input_images):
            if input_paths and i < len(input_paths):
                stem = Path(input_paths[i]).stem
            else:
                stem = f"input_view_{i:02d}"
            dest = out_path / f"{stem}.jpg"
            img_u8 = img_arr if img_arr.dtype == np.uint8 else (img_arr * 255).astype(np.uint8)
            Image.fromarray(img_u8).save(dest, format="JPEG", quality=95)
            logger.info(f"Copied input image to: {dest.name}")

        # ── Generate and save new views as JPEG ──────────────────────────────
        generated: List[np.ndarray] = []
        reference_pool = list(input_images)
        completed_specs: List[str] = []  # tracks angles already generated

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
                completed_specs.append(spec)  # record this angle as done

                dest = out_path / f"generated_view_{i:02d}.jpg"
                Image.fromarray(img).save(dest, format="JPEG", quality=95)
                logger.info(f"Saved generated view to: {dest.name}")

                if chain_views:
                    reference_pool.append(img)

            except Exception as e:
                logger.error(f"Failed to generate view {i}: {e}")
                continue

        logger.info(
            f"Generated {len(generated)}/{num_views} views "
            f"({'with' if chain_views else 'without'} sequential conditioning). "
            f"All images saved to '{output_dir}'."
        )
        return generated


def load_input_images(input_dir: str = "data/input") -> List[np.ndarray]:
    """Load all images from a directory, sorted by filename."""
    from pathlib import Path

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    images: List[np.ndarray] = []

    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    for filepath in sorted(input_path.iterdir()):
        if filepath.suffix.lower() in image_extensions:
            img = Image.open(filepath).convert("RGB")
            images.append(np.array(img))
            logger.info(f"Loaded: {filepath.name}  shape={images[-1].shape}")

    return images


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    images = load_input_images()
    if not images:
        print("No images found in data/input/. Add 4 corner images and re-run.")
        sys.exit(1)

    print(f"Loaded {len(images)} input images")

    generator = ImageGenerator()

    # Stage 1: structured scene analysis
    scene_description = generator.analyze_scene(images)
    print(f"\n{'='*60}\nSCENE ANALYSIS\n{'='*60}\n{scene_description}\n")

    # Stage 2: generate 6 new views with sequential conditioning
    generated = generator.generate_additional_views(
        input_images=images,
        scene_description=scene_description,
        num_views=6,
        chain_views=True,
    )

    print(f"\nGenerated {len(generated)} additional views → data/generated/")
