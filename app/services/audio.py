from __future__ import annotations

import re

import numpy as np

# Qwen3-TTS-12Hz emits ~12 codec frames per second.
# 2048 tokens ≈ 170s — that is why 3-character prompts can "breathe" for minutes
# when the model fails to emit EOS (common with Ono_Anna / short Japanese).
CODEC_HZ = 12
BUDGET_PAD_SECONDS = 1.5
BUDGET_SAFETY = 1.6
# Room for a breath before a one-word text; trim_speech removes what is left.
MIN_BUDGET_SECONDS = 6.0
# Above this, estimates are too rough to cut audio or retry on duration.
SHORT_TEXT_SECONDS = 10.0

_HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_KANA_RE = re.compile(r"[\u3041-\u3096\u30a1-\u30fa\u30fc]")
_HANGUL_RE = re.compile(r"[\uac00-\ud7af]")
_LATIN_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ\u0100-\u024f\u1e00-\u1eff']+")
_DIGIT_RE = re.compile(r"[0-9０-９]")
_PUNCT_RE = re.compile(r"[.!?。！？…]+")
_COMMA_RE = re.compile(r"[、，,・･;；]+")
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
    commas = len(_COMMA_RE.findall(cleaned))
    counted = han + kana + hangul + digits + sum(len(w) for w in words)
    counted += sum(len(m) for m in _PUNCT_RE.findall(cleaned) + _COMMA_RE.findall(cleaned))
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
        + punct * 0.45
        + commas * 0.25
    )


def budget_for_seconds(expected: float, ceiling: int = 2048) -> int:
    """Decode length for speech expected to last `expected` seconds."""
    ceiling = max(64, int(ceiling))
    estimated = int((expected + BUDGET_PAD_SECONDS) * CODEC_HZ * BUDGET_SAFETY)
    floor = int(MIN_BUDGET_SECONDS * CODEC_HZ)
    return min(ceiling, max(floor, estimated))


def budget_max_new_tokens(text: str, ceiling: int = 2048, language: str = "Auto") -> int:
    """Cap decode length so short text cannot fill the full 2048-token window."""
    return budget_for_seconds(estimate_speech_seconds(text, language), ceiling)


def max_plausible_seconds(text: str, language: str = "Auto") -> float | None:
    """Longest believable clip for short text; None when text is too long to judge."""
    expected = estimate_speech_seconds(text, language)
    if expected <= 0 or expected > SHORT_TEXT_SECONDS:
        return None
    return expected * 1.8 + 0.8


MAX_LIST_ITEMS = 60
# Decode ceiling per drill mora. It only stops runaway output; drills are never
# cut by time, the model reads until it stops by itself.
DRILL_MAX_ITEM_SECONDS = 2.5
_SILENT_MORAE = {"っ", "ッ"}
# One mora per item: あ, ア, きゃ, ファ, っ, かー. Words like はい or 東京 never
# qualify, so normal sentences keep the model's own rhythm.
_KANA_ITEM_RE = re.compile(
    r"^[\u3041-\u3096\u30a1-\u30fa][ぁぃぅぇぉゃゅょゎァィゥェォャュョヮ]?ー?$"
)
_LIST_SEP_RE = re.compile(r"[・･、,，\s]+")


def split_list_items(text: str) -> list[str] | None:
    """Split kana drill text like 'あ・い・う・え・お' into items, or None."""
    cleaned = (text or "").strip().strip("。．.!！?？")
    if not _LIST_SEP_RE.search(cleaned):
        return None
    parts = [p for p in _LIST_SEP_RE.split(cleaned) if p]
    if not (2 <= len(parts) <= MAX_LIST_ITEMS):
        return None
    if not all(_KANA_ITEM_RE.match(p) for p in parts):
        return None
    return parts


def drill_text(items: list[str]) -> str:
    """['あ', 'い', 'う'] → 'あ。い。う。' so every mora gets its own stop.

    The drill stays one generation: a lone 'あ。' gives the model too little
    text and it fills the gap with breaths and laughs.
    """
    return "。".join(items) + "。"


def audible_items(items: list[str]) -> int:
    """Morae that should show up as a separate sound (a lone っ is silent)."""
    return sum(1 for item in items if item not in _SILENT_MORAE)


def clean_drill(
    audio: np.ndarray,
    sample_rate: int,
    expected_units: int,
    *,
    rel_threshold: float = 0.12,
    frame_ms: float = 20.0,
    merge_gap_ms: float = 120.0,
    pad_ms: float = 100.0,
    fade_ms: float = 30.0,
) -> tuple[np.ndarray, int]:
    """Count the separate sounds in a kana drill and drop the extra ones.

    Returns (audio, sounds). Breath-like sounds are removed only when exactly
    `expected_units` sounds remain, so a hissy す or し is never taken for a
    breath and lost; otherwise the audio is returned untouched.
    """
    wave = np.asarray(audio)
    analysed = _frame_rms(wave, sample_rate, frame_ms)
    if analysed is None:
        return wave, 0
    frames, rms = analysed
    peak = float(np.percentile(rms, 95))
    if peak < 1e-6:
        return wave, 0
    voiced = rms >= peak * rel_threshold
    segments = _segments(voiced, max_gap=int(merge_gap_ms / frame_ms), min_len=3)
    if len(segments) <= expected_units:
        return wave, len(segments)

    speech_rms = float(np.median(np.concatenate([rms[a:b] for a, b in segments])))
    breaths = [_is_breath(frames, rms, seg, speech_rms) for seg in segments]
    kept = [seg for seg, b in zip(segments, breaths) if not b]
    if not kept or len(kept) != expected_units:
        return wave, len(segments)

    frame = frames.shape[1]
    pad = int(np.ceil(pad_ms / frame_ms))
    start = max(0, kept[0][0] - pad) * frame
    end = min(wave.shape[0], (kept[-1][1] + pad) * frame)
    out = np.array(wave[start:end], copy=True)
    ramp_len = max(2, int(sample_rate * 0.005))
    for (s, e), b in zip(segments, breaths):
        if b and kept[0][0] < s < kept[-1][1]:
            _mute(out, s * frame - start, e * frame - start, ramp_len)
    _fade_edges(out, sample_rate, fade_ms)
    return out, expected_units


def _frame_rms(
    audio: np.ndarray, sample_rate: int, frame_ms: float
) -> tuple[np.ndarray, np.ndarray] | None:
    """Mono audio as (frames, per-frame RMS), or None when too short."""
    wave = np.asarray(audio)
    if wave.size == 0 or sample_rate <= 0:
        return None
    mono = np.mean(wave, axis=-1) if wave.ndim > 1 else wave
    samples = np.asarray(mono, dtype=np.float32)
    frame = max(1, int(sample_rate * frame_ms / 1000.0))
    n_frames = samples.size // frame
    if n_frames < 3:
        return None
    frames = samples[: n_frames * frame].reshape(n_frames, frame)
    return frames, np.sqrt(np.mean(frames * frames, axis=1))


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
    rel_threshold: float = 0.14,
    frame_ms: float = 20.0,
    merge_gap_ms: float = 100.0,
    pad_ms: float = 40.0,
    fade_ms: float = 15.0,
    strip_breath: bool = True,
) -> tuple[np.ndarray, bool]:
    """Keep the spoken part: drop silence, breaths, and babble after the text.

    Returns (audio, has_speech); has_speech is False when only breath or
    silence came out. Breaths at the edges are cut off; breaths between words
    become silence so the pause length is kept. Pass strip_breath=False for
    lone morae such as す or し, whose hiss looks like a breath.
    """
    wave = np.asarray(audio)
    analysed = _frame_rms(wave, sample_rate, frame_ms)
    if analysed is None:
        return wave, False
    frames, rms = analysed
    frame = frames.shape[1]
    peak = float(np.percentile(rms, 95))
    if peak < 1e-6:
        return wave, False

    voiced = rms >= peak * rel_threshold
    segments = _segments(voiced, max_gap=int(merge_gap_ms / frame_ms), min_len=3)
    if not segments:
        return wave, False

    speech_rms = float(np.median(np.concatenate([rms[a:b] for a, b in segments])))
    breaths = [strip_breath and _is_breath(frames, rms, seg, speech_rms) for seg in segments]
    if all(breaths):
        return wave, False
    first = breaths.index(False)

    if expected_seconds is not None and expected_seconds > 0:
        # Measured from the first real sound, so a breath before it cannot
        # push the words past the cutoff.
        frames_per_sec = 1000.0 / frame_ms
        cutoff = segments[first][0] + int((expected_seconds * 1.5 + 0.4) * frames_per_sec)
        keep = [i for i, seg in enumerate(segments) if seg[0] <= cutoff]
        segments = [segments[i] for i in keep]
        breaths = [breaths[i] for i in keep]

    last = len(breaths) - 1 - breaths[::-1].index(False)
    muted = [seg for seg, b in zip(segments[first : last + 1], breaths[first : last + 1]) if b]
    segments = segments[first : last + 1]

    pad = int(np.ceil(pad_ms / frame_ms))
    start = max(0, segments[0][0] - pad) * frame
    end = min(wave.shape[0], (segments[-1][1] + pad) * frame)
    out = np.array(wave[start:end], copy=True)

    ramp_len = max(2, int(sample_rate * 0.005))
    for s, e in muted:
        _mute(out, s * frame - start, e * frame - start, ramp_len)

    _fade_edges(out, sample_rate, fade_ms)
    return out, True


def _fade_edges(out: np.ndarray, sample_rate: int, fade_ms: float) -> None:
    fade = min(out.shape[0] // 2, int(sample_rate * fade_ms / 1000.0))
    if fade > 1:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        if out.ndim > 1:
            ramp = ramp[:, None]
        out[:fade] = out[:fade] * ramp
        out[-fade:] = out[-fade:] * ramp[::-1]


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
    """Segment that is quieter than speech and noise-like (flat spectrum of breath/sigh)."""
    s, e = seg
    seg_rms = float(np.mean(rms[s:e]))
    if seg_rms > speech_rms * 0.70:
        return False
    window = np.hanning(frames.shape[1]).astype(np.float32)
    power = np.abs(np.fft.rfft(frames[s:e] * window, axis=1)) ** 2 + 1e-12
    flatness = np.exp(np.mean(np.log(power), axis=1)) / np.mean(power, axis=1)
    # A true breath/noise segment has high spectral flatness (noise-like, no harmonic formants)
    return float(np.mean(flatness)) > 0.22
