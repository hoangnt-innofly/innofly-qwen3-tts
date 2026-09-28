from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

import numpy as np

from app.services.audio import (
    budget_max_new_tokens,
    estimate_speech_seconds,
    max_plausible_seconds,
    prepare_text,
    trim_speech,
)

logger = logging.getLogger("qwen3-tts-api")

# Items per generate_custom_voice call when a list is spoken item by item.
BATCH_SIZE = 8


class TTSEngine:
    """Loads Qwen3-TTS 1.7B CustomVoice once and reuses it for text-to-speech jobs."""

    def __init__(
        self,
        model_id: str,
        *,
        device_map: str = "auto",
        dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        free_vram: bool = True,
    ) -> None:
        self.model_id = model_id
        self.device_map = device_map
        self.dtype_name = dtype
        self.attn_implementation = attn_implementation
        self.free_vram = free_vram
        self.model = None
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    def load(self) -> None:
        if self._ready:
            return

        import torch
        from qwen_tts import Qwen3TTSModel

        self.free_vram = self.free_vram and torch.cuda.is_available()
        device_map = self._resolve_device_map(torch)
        dtype = self._resolve_dtype(torch, device_map)
        attn_candidates = self._attn_candidates()

        last_error: Exception | None = None
        for attn in attn_candidates:
            try:
                logger.info(
                    "Loading %s (device_map=%s, dtype=%s, attn=%s, free_vram=%s)",
                    self.model_id,
                    device_map,
                    dtype,
                    attn,
                    self.free_vram,
                )
                self.model = Qwen3TTSModel.from_pretrained(
                    self.model_id,
                    device_map=device_map,
                    dtype=dtype,
                    attn_implementation=attn,
                )
                self._ready = True
                logger.info("Qwen3-TTS CustomVoice ready (%s)", self._vram_log())
                return
            except Exception as exc:
                last_error = exc
                logger.warning("Load failed with attn=%s: %s", attn, exc, exc_info=True)
                self.model = None
                self._free_cuda()
                if not self._is_attn_error(exc):
                    break

        detail = f"{type(last_error).__name__}: {last_error}" if last_error else "unknown error"
        raise RuntimeError(
            f"Failed to load Qwen3-TTS model {self.model_id}: {detail}"
        ) from last_error

    @staticmethod
    def _is_attn_error(exc: BaseException) -> bool:
        text = str(exc).lower()
        return any(
            token in text
            for token in (
                "attn",
                "attention",
                "flash_attn",
                "sdpa",
                "eager",
            )
        )

    def generate(
        self,
        *,
        text: str | list[str],
        language: str,
        speaker: str,
        output_path: str | Path,
        instruct: str = "",
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.05,
        max_new_tokens: int = 2048,
        do_sample: bool = True,
        seed: int = 42,
        subtalker_temperature: float = 0.9,
        subtalker_top_k: int = 50,
        subtalker_top_p: float = 1.0,
        max_attempts: int = 1,
        pause_ms: int = 0,
    ) -> tuple[Path, int, float]:
        """Synthesize `text`; a list is spoken item by item with `pause_ms` between."""
        if not self._ready:
            self.load()

        import torch
        import soundfile as sf

        items = [text] if isinstance(text, str) else list(text)
        spoken = [prepare_text(t) for t in items]
        expected = [estimate_speech_seconds(t, language) for t in spoken]
        limits = [max_plausible_seconds(t, language) for t in spoken]
        token_budget = max(
            budget_max_new_tokens(t, ceiling=max_new_tokens, language=language) for t in spoken
        )
        if token_budget < max_new_tokens:
            logger.info(
                "Capped max_new_tokens %s → %s for %s item(s)",
                max_new_tokens,
                token_budget,
                len(spoken),
            )

        n = len(spoken)
        attempts = max(1, int(max_attempts)) if any(lim is not None for lim in limits) else 1
        best: list[np.ndarray | None] = [None] * n
        done = [False] * n
        sample_rate = 24000
        try:
            for attempt in range(attempts):
                todo = [i for i in range(n) if not done[i]]
                if not todo:
                    break
                attempt_seed = seed + attempt
                torch.manual_seed(attempt_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(attempt_seed)
                cool = 0.8**attempt

                for start in range(0, len(todo), BATCH_SIZE):
                    chunk = todo[start : start + BATCH_SIZE]
                    kwargs: dict[str, Any] = {
                        "text": [spoken[i] for i in chunk],
                        "language": [language] * len(chunk),
                        "speaker": [speaker] * len(chunk),
                        "non_streaming_mode": True,
                        "do_sample": do_sample,
                        "temperature": temperature * cool,
                        "top_k": top_k,
                        "top_p": top_p,
                        "repetition_penalty": repetition_penalty,
                        "subtalker_dosample": do_sample,
                        "subtalker_temperature": subtalker_temperature * cool,
                        "subtalker_top_k": subtalker_top_k,
                        "subtalker_top_p": subtalker_top_p,
                        "max_new_tokens": token_budget,
                    }
                    if instruct.strip():
                        kwargs["instruct"] = instruct.strip()

                    wavs, sr = self.model.generate_custom_voice(**kwargs)
                    sample_rate = int(sr) if sr else 24000
                    for i, wav in zip(chunk, wavs):
                        raw = np.asarray(wav)
                        audio = trim_speech(
                            raw,
                            sample_rate,
                            expected_seconds=expected[i] if limits[i] is not None else None,
                        )
                        duration = audio.shape[0] / sample_rate
                        logger.info(
                            "Attempt %s/%s seed=%s item %s/%s: raw %.2fs → trimmed %.2fs (limit %s)",
                            attempt + 1,
                            attempts,
                            attempt_seed,
                            i + 1,
                            n,
                            raw.shape[0] / sample_rate,
                            duration,
                            f"{limits[i]:.2f}s" if limits[i] is not None else "none",
                        )
                        if limits[i] is None or duration <= limits[i]:
                            best[i] = audio
                            done[i] = True
                            continue
                        # Over the limit: keep the shortest, skipping near-empty clips.
                        plausible = duration >= expected[i] * 0.35
                        if best[i] is None or (plausible and audio.shape[0] < best[i].shape[0]):
                            best[i] = audio
                    del wavs

            clips = [clip for clip in best if clip is not None]
            gap = np.zeros(int(sample_rate * max(0, pause_ms) / 1000.0), dtype=clips[0].dtype)
            parts: list[np.ndarray] = []
            for k, clip in enumerate(clips):
                if k:
                    parts.append(gap)
                parts.append(clip)
            audio = np.concatenate(parts) if len(parts) > 1 else parts[0]

            duration = float(audio.shape[0] / sample_rate)
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_path), audio, sample_rate)
            return output_path, sample_rate, round(duration, 3)
        finally:
            if self.free_vram:
                self.release_vram()
            else:
                self._free_cuda()

    def release_vram(self) -> None:
        """Drop the GPU model so Comfy/LTX can use the 12GB card (reload on next job)."""
        self.model = None
        self._ready = False
        self._free_cuda()
        logger.info("Unloaded TTS from GPU (%s)", self._vram_log())

    def _resolve_device_map(self, torch) -> str:
        requested = (self.device_map or "auto").strip()
        if requested in {"auto", "cuda", "cuda:0"} and torch.cuda.is_available():
            return "cuda:0"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; falling back to CPU")
            return "cpu"
        if requested == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        return requested

    def _resolve_dtype(self, torch, device_map: str):
        name = (self.dtype_name or "bfloat16").lower()
        mapping = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        dtype = mapping.get(name, torch.bfloat16)
        if str(device_map) == "cpu" and dtype in {torch.bfloat16, torch.float16}:
            return torch.float32
        return dtype

    def _attn_candidates(self) -> list[str]:
        preferred = (self.attn_implementation or "sdpa").strip() or "sdpa"
        ordered = [preferred]
        for name in ("sdpa", "flash_attention_2", "eager"):
            if name not in ordered:
                ordered.append(name)
        return ordered

    @staticmethod
    def _vram_log() -> str:
        try:
            import torch

            if not torch.cuda.is_available():
                return "cpu"
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            return f"cuda allocated={allocated:.2f}GiB reserved={reserved:.2f}GiB"
        except Exception:
            return "unknown"

    @staticmethod
    def _free_cuda() -> None:
        try:
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
