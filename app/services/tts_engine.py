from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

import numpy as np

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
        self._compute_device = "cpu"
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
        self._compute_device = device_map
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
                load_kwargs: dict[str, Any] = {
                    "dtype": dtype,
                    "attn_implementation": attn,
                }
                # device_map pins modules via accelerate; later .to(cpu)/.to(cuda)
                # then leaves embeddings on CPU while input_ids go to CUDA.
                if not self.free_vram:
                    load_kwargs["device_map"] = device_map
                self.model = Qwen3TTSModel.from_pretrained(self.model_id, **load_kwargs)
                if self.free_vram:
                    self.release_vram()
                self._ready = True
                logger.info("Qwen3-TTS CustomVoice ready (%s)", self._vram_log())
                return
            except Exception as exc:
                last_error = exc
                logger.warning("Load failed with attn=%s: %s", attn, exc)
                self.model = None
                self._free_cuda()

        raise RuntimeError(f"Failed to load Qwen3-TTS model {self.model_id}") from last_error

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
            "max_new_tokens": max_new_tokens,
        }
        if instruct.strip():
            kwargs["instruct"] = instruct.strip()

        try:
            if self.free_vram:
                self._place_on_gpu()
            wavs, sample_rate = self.model.generate_custom_voice(**kwargs)
            audio = np.asarray(wavs[0])
            del wavs
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_path), audio, sample_rate)
            duration = float(audio.shape[0] / sample_rate) if sample_rate else 0.0
            return output_path, int(sample_rate), round(duration, 3)
        finally:
            if self.free_vram:
                self.release_vram()
            else:
                self._free_cuda()

    def release_vram(self) -> None:
        """Move weights off GPU and empty the CUDA cache. Generate still uses CUDA."""
        self._move_weights("cpu")
        self._free_cuda()
        logger.info("Released TTS VRAM (%s)", self._vram_log())

    def _place_on_gpu(self) -> None:
        self._move_weights(self._compute_device)

    def _inner_model(self):
        wrapper = self.model
        if wrapper is None:
            return None
        return getattr(wrapper, "model", wrapper)

    def _move_weights(self, device: str) -> None:
        import torch
        import torch.nn as nn

        wrapper = self.model
        inner = self._inner_model()
        if inner is None:
            return

        target = torch.device(device)

        def move_module(mod: nn.Module) -> None:
            if hasattr(mod, "hf_device_map"):
                try:
                    delattr(mod, "hf_device_map")
                except Exception:
                    mod.hf_device_map = {}
            mod.to(target)

        seen: set[int] = set()
        stack: list[Any] = [inner]
        for name in ("speech_tokenizer", "speaker_encoder", "talker"):
            obj = getattr(inner, name, None)
            if obj is not None:
                stack.append(obj)

        for obj in stack:
            if obj is None or id(obj) in seen:
                continue
            seen.add(id(obj))
            if isinstance(obj, nn.Module):
                try:
                    move_module(obj)
                except Exception as exc:
                    logger.warning("Could not move %s to %s: %s", type(obj).__name__, device, exc)
            elif hasattr(obj, "to"):
                try:
                    obj.to(target)
                except Exception as exc:
                    logger.warning("Could not move %s to %s: %s", type(obj).__name__, device, exc)

        for name, value in list(vars(inner).items()):
            if torch.is_tensor(value) and value.device != target:
                setattr(inner, name, value.to(target))

        if hasattr(wrapper, "device"):
            try:
                wrapper.device = next(inner.parameters()).device
            except (StopIteration, TypeError, AttributeError):
                wrapper.device = target

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
