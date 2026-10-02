from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

import numpy as np

from app.services.audio import prepare_text, trim_speech

logger = logging.getLogger("omnivoice-tts-engine")


class OmniVoiceEngine:
    """Loads k2-fsa/OmniVoice (VoiceStudio model) for zero-shot and multilingual TTS."""

    def __init__(
        self,
        model_id: str = "k2-fsa/OmniVoice",
        *,
        device_map: str = "auto",
        dtype: str = "float16",
        free_vram: bool = True,
    ) -> None:
        self.model_id = model_id
        self.device_map = device_map
        self.dtype_name = dtype
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

        self.free_vram = self.free_vram and torch.cuda.is_available()
        device_map = self._resolve_device_map(torch)
        dtype = self._resolve_dtype(torch, device_map)

        try:
            from omnivoice import OmniVoice
        except ImportError as exc:
            raise RuntimeError(
                "Package 'omnivoice' is not installed. Install it with: pip install omnivoice"
            ) from exc

        try:
            logger.info(
                "Loading OmniVoice %s (device_map=%s, dtype=%s, free_vram=%s)",
                self.model_id,
                device_map,
                dtype,
                self.free_vram,
            )
            self.model = OmniVoice.from_pretrained(
                self.model_id,
                device_map=device_map,
                dtype=dtype,
            )
            self._ready = True
            logger.info("OmniVoice engine ready (%s)", self._vram_log())
        except Exception as exc:
            self.model = None
            self._free_cuda()
            logger.error("Failed to load OmniVoice model %s: %s", self.model_id, exc, exc_info=True)
            raise RuntimeError(f"Failed to load OmniVoice model {self.model_id}: {exc}") from exc

    def generate(
        self,
        *,
        text: str,
        output_path: str | Path,
        ref_audio: str | Path | None = None,
        ref_text: str | None = None,
        num_step: int = 16,
        speed: float = 1.0,
        guidance_scale: float = 2.0,
        seed: int = 42,
        sample_rate: int = 24000,
    ) -> tuple[Path, int, float]:
        """Synthesize `text` cleanly with OmniVoice (no extra breaths, laughs, or noise)."""
        if not self._ready:
            self.load()

        import soundfile as sf
        import torch

        try:
            spoken = prepare_text(text)

            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

            kwargs: dict[str, Any] = {
                "text": spoken,
                "num_step": num_step,
                "speed": speed,
            }
            if guidance_scale is not None:
                kwargs["guidance_scale"] = guidance_scale

            if ref_audio:
                ref_path = Path(ref_audio)
                if ref_path.is_file():
                    kwargs["ref_audio"] = str(ref_path)
                    if ref_text:
                        kwargs["ref_text"] = ref_text.strip()

            logger.info("OmniVoice synthesizing: %r (speed=%.2f, num_step=%d)", spoken[:60], speed, num_step)
            audio_out = self.model.generate(**kwargs)

            # Extract numpy array from list / tensor
            if isinstance(audio_out, (list, tuple)):
                audio_arr = audio_out[0]
            else:
                audio_arr = audio_out

            if hasattr(audio_arr, "cpu"):
                audio_arr = audio_arr.detach().cpu().numpy()

            audio_arr = np.asarray(audio_arr, dtype=np.float32)
            if audio_arr.ndim > 1:
                audio_arr = audio_arr.squeeze()

            # Trim silent edges and unwanted heavy breath artifacts
            if audio_arr.size > 0:
                audio_arr, has_speech = trim_speech(audio_arr, sample_rate, strip_breath=True)

            # Normalize peak
            max_val = np.max(np.abs(audio_arr)) if audio_arr.size > 0 else 0
            if max_val > 1.0:
                audio_arr = audio_arr / max_val

            duration = float(len(audio_arr) / sample_rate) if sample_rate > 0 else 0.0

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_path), audio_arr, sample_rate)

            logger.info("OmniVoice generated %.2fs clean audio at %s", duration, output_path)
            return output_path, sample_rate, round(duration, 3)
        finally:
            if self.free_vram:
                self.release_vram()
            else:
                self._free_cuda()

    def release_vram(self) -> None:
        """Drop the GPU model so VRAM is freed for other tasks."""
        self.model = None
        self._ready = False
        self._free_cuda()
        logger.info("Unloaded OmniVoice from GPU (%s)", self._vram_log())

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
        name = (self.dtype_name or "float16").lower()
        mapping = {
            "fp16": torch.float16,
            "float16": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        dtype = mapping.get(name, torch.float16)
        if str(device_map) == "cpu" and dtype in {torch.bfloat16, torch.float16}:
            return torch.float32
        return dtype

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
