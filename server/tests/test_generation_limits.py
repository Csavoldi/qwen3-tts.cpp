#!/usr/bin/env python3
"""Pure standard-library tests for the generation-length cap."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generation_limits import (
    MIN_DERIVED_MAX_AUDIO_TOKENS,
    PIPELINE_DEFAULT_MAX_AUDIO_TOKENS,
    derive_max_audio_tokens,
)


class DeriveMaxAudioTokensTest(unittest.TestCase):
    def test_short_text_gets_the_floor(self):
        # 1 character would derive 2 frames, which is useless: the floor applies.
        self.assertEqual(derive_max_audio_tokens("a"), MIN_DERIVED_MAX_AUDIO_TOKENS)

    def test_derives_twice_the_character_count(self):
        text = "x" * 100
        self.assertEqual(derive_max_audio_tokens(text), 200)

    def test_never_exceeds_the_pipeline_default(self):
        # A long read-aloud chunk is exactly where the old flat 2048-frame cap
        # produced 163.8s of audio for a 44-character sentence.
        text = "x" * 5000
        self.assertEqual(
            derive_max_audio_tokens(text), PIPELINE_DEFAULT_MAX_AUDIO_TOKENS
        )

    def test_fixed_cap_wins(self):
        self.assertEqual(derive_max_audio_tokens("x" * 5000, fixed_cap=512), 512)
        self.assertEqual(derive_max_audio_tokens("", fixed_cap=1), 1)

    def test_whitespace_does_not_inflate_the_cap(self):
        self.assertEqual(derive_max_audio_tokens("   " * 50), MIN_DERIVED_MAX_AUDIO_TOKENS)

    def test_rejects_invalid_arguments(self):
        with self.assertRaises(ValueError):
            derive_max_audio_tokens("hello", fixed_cap=-1)
        with self.assertRaises(ValueError):
            derive_max_audio_tokens("hello", pipeline_default=0)


if __name__ == "__main__":
    unittest.main()
