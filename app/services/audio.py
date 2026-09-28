from __future__ import annotations

import re

import numpy as np

# Qwen3-TTS-12Hz emits ~12 codec frames per second.
# 2048 tokens ≈ 170s — that is why 3-character prompts can "breathe" for minutes
# when the model fails to emit EOS (common with Ono_Anna / short Japanese).
CODEC_HZ = 12
BUDGET_PAD_SECONDS = 1.5
BUDGET_SAFETY = 1.6
MIN_BUDGET_SECONDS = 4.0
# Above this, estimates are too rough to cut audio or retry on duration.
SHORT_TEXT_SECONDS = 10.0

_HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_KANA_RE = re.compile(r"[\u3040-\u30ff]")
_HANGUL_RE = re.compile(r"[\uac00-\ud7af]")
_LATIN_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ\u0100-\u024f\u1e00-\u1eff']+")
_DIGIT_RE = re.compile(r"[0-9０-９]")
_PUNCT_RE = re.compile(r"[.!?。！？…]+")
_TERMINAL_PUNCT = set(".!?。！？…、，,;；:：~〜」』）)\"'”’")


def estimate_speech_seconds(text: str, language: str = "Auto") -> float:
    """Rough spoken length of `text`, without any padding."""
    cleaned = (text or "").strip()
    if not cleaned:
        return 0.0
    han = len(_HAN_RE.findall(cleaned))
    kana = len(_KANA_RE.findall(cleaned))
    hangul = len(_HANGUL_RE.findall(cleaned))
    words = _LATIN_WORD_RE.findall(cleaned)
    digits = len(_DIGIT_RE.findall(cleaned))
    punct = len(_PUNCT_RE.findall(cleaned))
    counted = han + kana + hangul + digits + sum(len(w) for w in words)
    leftover = max(0, len(re.sub(r"\s+", "", cleaned)) - counted)
    # A kanji is ~2 morae in Japanese, one syllable in Chinese.
    han_seconds = 0.24 if language == "Chinese" else 0.36
    return (
        han * han_seconds
        + kana * 0.13
        + hangul * 0.22
        + len(words) * 0.45
        + digits * 0.35
        + leftover * 0.08
        + punct * 0.25
    )


def budget_max_new_tokens(text: str, ceiling: int = 2048, language: str = "Auto") -> int:
    """Cap decode length so short text cannot fill the full 2048-token window."""
    ceiling = max(64, int(ceiling))
    seconds = estimate_speech_seconds(text, language) + BUDGET_PAD_SECONDS
    estimated = int(seconds * CODEC_HZ * BUDGET_SAFETY)
    floor = int(MIN_BUDGET_SECONDS * CODEC_HZ)
    return min(ceiling, max(floor, estimated))


def max_plausible_seconds(text: str, language: str = "Auto") -> float | None:
    """Longest believable clip for short text; None when text is too long to judge."""
    expected = estimate_speech_seconds(text, language)
    if expected <= 0 or expected > SHORT_TEXT_SECONDS:
        return None
    return expected * 1.8 + 0.8


def prepare_text(text: str) -> str:
    """End the text with punctuation so the model has a clear place to emit EOS."""
    cleaned = (text or "").strip()
    if not cleaned or cleaned[-1] in _TERMINAL_PUNCT:
        return cleaned
    last = cleaned[-1]
    if _HAN_RE.match(last) or _KANA_RE.match(last) or _HANGUL_RE.match(last) or last == "ー":
        return cleaned + "。"
    return cleaned + "."


def trim_speech(
    audio: np.ndarray,
    sample_rate: int,
    *,
    expected_seconds: float | None = None,
    rel_threshold: float = 0.12,
    frame_ms: float = 20.0,
    merge_gap_ms: float = 220.0,
    pad_ms: float = 120.0,
    fade_ms: float = 30.0,
) -> np.ndarray:
    """Keep the spoken part: drop silence, breaths, and babble after the text.

    Breaths at the edges are cut off; breaths between words become silence so
    the pause length is kept.
    """
    wave = np.asarray(audio)
    if wave.size == 0 or sample_rate <= 0:
        return wave
    mono = np.mean(wave, axis=-1) if wave.ndim > 1 else wave
    samples = np.asarray(mono, dtype=np.float32)

    frame = max(1, int(sample_rate * frame_ms / 1000.0))
    n_frames = samples.size // frame
    if n_frames < 3:
        return wave

    frames = samples[: n_frames * frame].reshape(n_frames, frame)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    peak = float(np.percentile(rms, 95))
    if peak < 1e-6:
        return wave

    voiced = rms >= peak * rel_threshold
    segments = _segments(voiced, max_gap=int(merge_gap_ms / frame_ms), min_len=3)
    if not segments:
        return wave

    if expected_seconds is not None and expected_seconds > 0:
        frames_per_sec = 1000.0 / frame_ms
        cutoff = segments[0][0] + int((expected_seconds * 1.5 + 0.4) * frames_per_sec)
        kept = [seg for seg in segments if seg[0] <= cutoff]
        segments = kept or segments[:1]

    speech_rms = float(np.median(np.concatenate([rms[a:b] for a, b in segments])))
    breaths = [_is_breath(frames, rms, seg, speech_rms) for seg in segments]
    if not all(breaths):
        first = breaths.index(False)
        last = len(breaths) - 1 - breaths[::-1].index(False)
        muted = [seg for seg, b in zip(segments[first : last + 1], breaths[first : last + 1]) if b]
        segments = segments[first : last + 1]
    else:
        muted = []

    pad = int(np.ceil(pad_ms / frame_ms))
    start = max(0, segments[0][0] - pad) * frame
    end = min(wave.shape[0], (segments[-1][1] + pad) * frame)
    out = np.array(wave[start:end], copy=True)

    ramp_len = max(2, int(sample_rate * 0.005))
    for s, e in muted:
        _mute(out, s * frame - start, e * frame - start, ramp_len)

    fade = min(out.shape[0] // 2, int(sample_rate * fade_ms / 1000.0))
    if fade > 1:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        if out.ndim > 1:
            ramp = ramp[:, None]
        out[:fade] = out[:fade] * ramp
        out[-fade:] = out[-fade:] * ramp[::-1]
    return out


def _mute(out: np.ndarray, start: int, end: int, ramp_len: int) -> None:
    """Silence out[start:end] in place with short ramps so no click is left."""
    start, end = max(0, start), min(out.shape[0], end)
    if end - start <= 2 * ramp_len:
        return
    down = np.linspace(1.0, 0.0, ramp_len, dtype=np.float32)
    up = down[::-1]
    if out.ndim > 1:
        down, up = down[:, None], up[:, None]
    out[start : start + ramp_len] *= down
    out[start + ramp_len : end - ramp_len] = 0
    out[end - ramp_len : end] *= up


def _segments(mask: np.ndarray, *, max_gap: int, min_len: int) -> list[tuple[int, int]]:
    """Runs of True as (start, end) frame indices, merging gaps up to `max_gap`."""
    runs: list[list[int]] = []
    i, n = 0, len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        if runs and i - runs[-1][1] <= max_gap:
            runs[-1][1] = j
        else:
            runs.append([i, j])
        i = j
    return [(s, e) for s, e in runs if e - s >= min_len]


def _is_breath(
    frames: np.ndarray,
    rms: np.ndarray,
    seg: tuple[int, int],
    speech_rms: float,
) -> bool:
    """Segment that is quieter than speech and noise-like (flat spectrum)."""
    s, e = seg
    if float(np.mean(rms[s:e])) > speech_rms * 0.6:
        return False
    window = np.hanning(frames.shape[1]).astype(np.float32)
    power = np.abs(np.fft.rfft(frames[s:e] * window, axis=1)) ** 2 + 1e-12
    flatness = np.exp(np.mean(np.log(power), axis=1)) / np.mean(power, axis=1)
    return float(np.mean(flatness)) > 0.3
