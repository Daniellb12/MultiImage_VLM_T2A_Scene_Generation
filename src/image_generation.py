"""Image Generation — factory entry point.

Import ``ImageGenerator`` from this module as usual.  The ``backend`` kwarg
(or the ``IMAGE_GEN_BACKEND`` env var) selects the underlying implementation:

    "nano_banana" (default) — Gemini Nano Banana (gemini-3.1-flash-image-preview)
    "openai"                — GPT Image 2       (gpt-image-2)

Examples
--------
# Default: Nano Banana
from src.image_generation import ImageGenerator
gen = ImageGenerator()

# Explicit backend via kwarg
gen = ImageGenerator(backend="openai")

# Nano Banana with all options passed through
gen = ImageGenerator(backend="nano_banana", api_key="...", model="gemini-3.1-flash-image-preview")

# GPT Image 2 with quality/size options
gen = ImageGenerator(backend="openai", quality="high", size="1536x1024")
"""

from __future__ import annotations

import os
from typing import Optional

# Re-export load_input_images so existing imports keep working
from src.image_generation_nano_banana import load_input_images  # noqa: F401

_VALID_BACKENDS = ("nano_banana", "openai")


def ImageGenerator(
    backend: Optional[str] = None,
    **kwargs,
):
    """Factory that returns a backend-specific image generator.

    Args:
        backend: "nano_banana" or "openai".  Falls back to the
                 IMAGE_GEN_BACKEND env var, then "nano_banana".
        **kwargs: Passed verbatim to the backend constructor.

    Returns:
        NanoBananaImageGenerator or OpenAIImageGenerator instance.
    """
    resolved = (
        backend
        or os.getenv("IMAGE_GEN_BACKEND", "nano_banana")
    ).lower().strip()

    if resolved not in _VALID_BACKENDS:
        raise ValueError(
            f"Unknown image generation backend '{resolved}'. "
            f"Choose from: {_VALID_BACKENDS}"
        )

    if resolved == "openai":
        from src.image_generation_openai import OpenAIImageGenerator
        return OpenAIImageGenerator(**kwargs)

    from src.image_generation_nano_banana import NanoBananaImageGenerator
    return NanoBananaImageGenerator(**kwargs)
