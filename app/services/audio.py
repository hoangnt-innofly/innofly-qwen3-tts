from __future__ import annotations

import re

import numpy as np

# Qwen3-TTS-12Hz emits ~12 codec frames per second.
# 2048 tokens ≈ 170s — that is why 3-character prompts can "breathe" for minutes
# when the model fails to emit EOS (common with Ono_Anna / short Japanese).
CODEC_HZ = 12
MIN_SECONDS = 3.5
PAD_SECONDS = 2.0
SAFETY = 2.0

_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]")
_LATIN_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ\u0100-\u024f\u1e00-\u1eff']+")
_PUNCT_RE = re.compile(r"[.!?。！？…]+")


def estimate_speech_seconds(text: str) -> float:
    cleaned = (text or "").strip()
    if not cleaned:
        return MIN_SECONDS
    cjk = len(_CJK_RE.findall(cleaned))
    latin_words = len(_LATIN_WORD_RE.findall(cleaned))
    punct = len(_PUNCT_RE.findall(cleaned))
    leftover = max(0, len(re.sub(r"\s+", "", cleaned)) - cjk - sum(len(w) for w in _LATIN_WORD_RE.findall(cleaned)))
    seconds = cjk * 0.32 + latin_words * 0.45 + leftover * 0.12 + punct * 0.25 + PAD_SECONDS
    return max(MIN_SECONDS, seconds)


def budget_max_new_tokens(text: str, ceiling: int = 2048) -> int:
    """Cap decode length so short text cannot fill the full 2048-token window."""
    ceiling = max(64, int(ceiling))
    estimated = int(estimate_speech_seconds(text) * CODEC_HZ * SAFETY)
    floor = int(MIN_SECONDS * CODEC_HZ)
    return min(ceiling, max(floor, estimated))


def trim_low_energy(
    audio: np.ndarray,
    sample_rate: int,
    *,
    rel_threshold: float = 0.18,
    frame_ms: float = 20.0,
    pad_ms: float = 140.0,
    min_keep_seconds: float = 0.25,
) -> np.ndarray:
    """Drop leading/trailing breath and silence; keep a short pad around speech."""
    wave = np.asarray(audio)
    if wave.size == 0 or sample_rate <= 0:
        return wave
    if wave.ndim > 1:
        mono = np.mean(wave, axis=-1)
    else:
        mono = wave

    samples = np.asarray(mono, dtype=np.float32)
    frame = max(1, int(sample_rate * frame_ms / 1000.0))
    usable = (samples.size // frame) * frame
    if usable < frame:
        return audio

    frames = samples[:usable].reshape(-1, frame)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    peak = float(np.percentile(rms, 95)) if rms.size else 0.0
    min_keep = max(1, int(sample_rate * min_keep_seconds))
    if peak < 1e-6:
        return _slice_audio(audio, 0, min(audio.shape[0], min_keep))

    voiced = rms >= (peak * rel_threshold)
    if not np.any(voiced):
        return _slice_audio(audio, 0, min(audio.shape[0], min_keep))

    first = int(np.argmax(voiced))
    last = int(len(voiced) - 1 - np.argmax(voiced[::-1]))
    pad = int(np.ceil(pad_ms / frame_ms))
    start = max(0, first - pad) * frame
    end = min(samples.size, (last + 1 + pad) * frame)
    if end - start < min_keep:
        return audio
    return _slice_audio(audio, start, end)


def _slice_audio(audio: np.ndarray, start: int, end: int) -> np.ndarray:
    if audio.ndim > 1:
        return audio[start:end, ...]
    return audio[start:end]
