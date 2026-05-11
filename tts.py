"""Kokoro-via-MLX text-to-speech wrapper.

Loads `prince-canuma/Kokoro-82M` (default) once and exposes a `synthesize`
method that returns (sample_rate, float32 numpy audio) for a single piece of text.
The model itself yields per-segment results; we concatenate them so callers can
treat one input string as one playable buffer.
"""

from __future__ import annotations

import numpy as np

from mlx_audio.tts.utils import load_model


class KokoroTTS:
    def __init__(
        self,
        model_id: str = "prince-canuma/Kokoro-82M",
        default_voice: str = "af_heart",
        speed: float = 1.0,
        lang_code: str = "a",
    ):
        self.model_id = model_id
        self.default_voice = default_voice
        self.speed = speed
        self.lang_code = lang_code
        self.model = load_model(model_id)

    def synthesize(self, text: str, voice: str | None = None) -> tuple[int, np.ndarray]:
        voice = voice or self.default_voice
        chunks: list[np.ndarray] = []
        sample_rate: int | None = None
        # Kokoro voice ids encode language in their prefix (af_*/am_* = American,
        # bf_*/bm_* = British, etc.). Match the pipeline lang_code to avoid a
        # "language mismatch" mispronunciation warning.
        lang_code = voice[0] if voice and voice[0].isalpha() else self.lang_code
        for result in self.model.generate(
            text=text,
            voice=voice,
            speed=self.speed,
            lang_code=lang_code,
        ):
            audio = np.asarray(result.audio, dtype=np.float32)
            chunks.append(audio)
            sample_rate = int(result.sample_rate)
        if not chunks or sample_rate is None:
            raise RuntimeError("Kokoro produced no audio for input text")
        return sample_rate, np.concatenate(chunks)
