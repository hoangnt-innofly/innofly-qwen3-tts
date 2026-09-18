# Qwen3-TTS 1.7B CustomVoice API

FastAPI backend for multilingual **text-to-speech** with [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) **`Qwen3-TTS-12Hz-1.7B-CustomVoice`**.

Submit text + speaker + language. Response includes `audio_url` (WAV).

## Model (t2s)

| Setting | Value |
| --- | --- |
| Model | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` |
| Mode | CustomVoice text-to-speech (`generate_custom_voice`) |
| Languages | Auto, Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian |
| Speakers | Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee |
| Instruct | natural-language style / emotion (1.7B only) |
| VRAM | ~4 GB BF16 |

Each speaker can speak any supported language. Native pairing is usually better (Ryan/Aiden → English, Vivian/Serena → Chinese, Ono_Anna → Japanese, Sohee → Korean).

## API

| Method | Path | What it does |
| --- | --- | --- |
| `POST` | `/api/v1/generate` | Form TTS, **wait**, return `audio_url` |
| `POST` | `/api/v1/jobs` | Same form, return immediately (`202`) |
| `GET` | `/api/v1/jobs/{job_id}` | Poll until `status=succeeded` and `audio_url` is set |
| `GET` | `/api/v1/voices` | Speakers + languages for the form |
| `GET` | `/media/audio/{file}` | Stream the wav |
| `GET` | `/health` | CUDA / mock / defaults |
| `GET` | `/docs` | Swagger |

Multipart fields matching `generate_custom_voice`:

| Field | Required | Notes |
| --- | --- | --- |
| `text` | yes | Text to speak |
| `language` | no | `Auto` (default) or a supported language |
| `speaker` | no | One of the 9 CustomVoice speakers (`Vivian` default) |
| `instruct` | no | Style/emotion instruction, e.g. `Nói chậm và vui` |
| `temperature` | no | Sampling temperature, default `0.9` |
| `top_p` | no | Nucleus sampling, default `1.0` |
| `top_k` | no | Default `50` |
| `repetition_penalty` | no | Default `1.05` |
| `max_new_tokens` | no | Default `2048` |
| `do_sample` | no | Default `true` |
| `seed` | no | Default `42` |

Example (sync — response includes `audio_url`):

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/generate" `
  -F "text=Xin chào, đây là Qwen3-TTS. Hello from multilingual text-to-speech." `
  -F "language=Auto" `
  -F "speaker=Vivian" `
  -F "instruct=Nói ấm áp, rõ ràng, hơi vui"
```

```json
{
  "job_id": "...",
  "status": "succeeded",
  "text": "...",
  "language": "Auto",
  "speaker": "Vivian",
  "instruct": "Nói ấm áp, rõ ràng, hơi vui",
  "mode": "t2s",
  "sample_rate": 24000,
  "duration_seconds": 4.12,
  "audio_url": "http://127.0.0.1:8000/media/audio/<job_id>.wav",
  "error": null
}
```

Jobs are serialized on one GPU worker so concurrent requests do not OOM.

## Setup (Windows + Python 3.12)

Qwen3-TTS recommends Python 3.12.

```powershell
cd qwen3-tts-api
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
copy .env.example .env
```

### A. API contract only (no GPU)

In `.env` set `TTS_MOCK=1`, then:

```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open http://127.0.0.1:8000 — submit text, get a placeholder wav URL.

### B. Real Qwen3-TTS 1.7B CustomVoice

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
python -m pip install -U qwen-tts
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python scripts/download_models.py
```

If you download weights locally, set `TTS_MODEL_ID` to that folder, for example `models/Qwen3-TTS-12Hz-1.7B-CustomVoice`. Otherwise the Hugging Face id is used and weights download on first load.

Optional FlashAttention 2 (lower VRAM, needs compatible GPU + `flash-attn`):

```powershell
pip install -U flash-attn --no-build-isolation
```

Then set `TTS_ATTN_IMPLEMENTATION=flash_attention_2` in `.env`. The engine falls back to `sdpa` / `eager` if that fails.

## Project layout

```
app/            FastAPI app, job queue, Qwen3-TTS engine
static/         upload demo page (Vietnamese form)
scripts/        weight download helper
storage/        generated wavs
```
