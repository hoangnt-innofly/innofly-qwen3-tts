from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np


def generate_placeholder_wav(
    output_path: str | Path,
    *,
    text: str,
    speaker: str,
    sample_rate: int = 24000,
) -> tuple[Path, int, float]:
    """Short tone WAV so the API contract can be tested without a GPU."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    duration = min(6.0, max(1.2, 0.08 * max(1, len(text.strip()))))
    n = int(sample_rate * duration)
    t = np.arange(n, dtype=np.float32) / sample_rate
    seed = abs(hash((text, speaker))) % (2**31)
    rng = np.random.default_rng(seed)
    freq = 180.0 + (seed % 220)
    tone = 0.18 * np.sin(2 * math.pi * freq * t)
    shimmer = 0.04 * np.sin(2 * math.pi * (freq / 2.5) * t)
    noise = 0.01 * rng.standard_normal(n).astype(np.float32)
    envelope = np.minimum(1.0, t / 0.04) * np.minimum(1.0, (duration - t) / 0.08)
    audio = np.clip((tone + shimmer + noise) * envelope, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16)

    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())

    return output_path, sample_rate, round(duration, 3)
