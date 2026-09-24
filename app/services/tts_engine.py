from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

import numpy as np

from app.services.audio import budget_max_new_tokens, trim_low_energy

logger = logging.getLogger("qwen3-tts-api")


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
        text: str,
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
    ) -> tuple[Path, int, float]:
        if not self._ready:
            self.load()

        import torch
        import soundfile as sf

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        token_budget = budget_max_new_tokens(text, ceiling=max_new_tokens)
        kwargs: dict[str, Any] = {
            "text": text,
            "language": language,
            "speaker": speaker,
            "non_streaming_mode": True,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": token_budget,
        }
        if instruct.strip():
            kwargs["instruct"] = instruct.strip()

        if token_budget < max_new_tokens:
            logger.info(
                "Capped max_new_tokens %s → %s for %s chars",
                max_new_tokens,
                token_budget,
                len(text),
            )

        try:
            wavs, sample_rate = self.model.generate_custom_voice(**kwargs)
            audio = np.asarray(wavs[0])
            del wavs
            raw_duration = float(audio.shape[0] / sample_rate) if sample_rate else 0.0
            audio = trim_low_energy(audio, int(sample_rate) if sample_rate else 24000)
            duration = float(audio.shape[0] / sample_rate) if sample_rate else 0.0
            if raw_duration and duration < raw_duration * 0.9:
                logger.info("Trimmed trailing breath/silence %.2fs → %.2fs", raw_duration, duration)
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_path), audio, sample_rate)
            return output_path, int(sample_rate), round(duration, 3)
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
