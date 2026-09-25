#!/usr/bin/env python3
"""Generation-length limits for the OpenAI-compatible server.

Kept free of FastAPI and of the ctypes binding so it can be unit tested without
building the shared library or touching a GPU.
"""

from __future__ import annotations

#: The pipeline's own default: 2048 audio frames, i.e. ~164s of audio at the
#: tokenizer's 12.5 frames per second. A degenerate decode really does run to it.
PIPELINE_DEFAULT_MAX_AUDIO_TOKENS = 2048

#: Floor for derived caps, so a very short request still gets a usable window.
MIN_DERIVED_MAX_AUDIO_TOKENS = 64

#: This model speaks at roughly 12-17 characters per second and the tokenizer
#: emits 12.5 frames per second, so frames ~= characters. Allow 2x for headroom.
CHARS_TO_FRAMES_FACTOR = 2


def derive_max_audio_tokens(
    text: str,
    fixed_cap: int = 0,
    *,
    pipeline_default: int = PIPELINE_DEFAULT_MAX_AUDIO_TOKENS,
) -> int:
    """Bound generation for one request.

    Args:
        text: The request text whose length the cap is derived from.
        fixed_cap: When positive, use this instead of deriving a cap.
        pipeline_default: Ceiling for a derived cap.

    Returns:
        A positive number of audio frames.

    Raises:
        ValueError: If ``fixed_cap`` or ``pipeline_default`` is not positive.
    """
    if fixed_cap < 0:
        raise ValueError("fixed cap must not be negative")
    if pipeline_default <= 0:
        raise ValueError("pipeline default must be positive")
    if fixed_cap > 0:
        return fixed_cap
    derived = CHARS_TO_FRAMES_FACTOR * len(text.strip())
    return min(pipeline_default, max(MIN_DERIVED_MAX_AUDIO_TOKENS, derived))
