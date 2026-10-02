# Multilingual TTS API (OmniVoice & Qwen3-TTS)

FastAPI backend cho dịch vụ **chuyển văn bản thành giọng nói (Text-to-Speech)** chất lượng cao, tích hợp song song 2 model:
1. 🌟 **OmniVoice** (Model cốt lõi của [VoiceStudio](https://github.com/debpalash/VoiceStudio) / [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice)): Hỗ trợ **646 ngôn ngữ**, ngữ điệu câu hỏi thoại tiếng Việt/quốc tế cực kỳ tự nhiên, hỗ trợ **Zero-shot Voice Cloning** (nhân bản giọng từ file mẫu 3–10s).
2. 🎙️ **Qwen3-TTS** (`Qwen3-TTS-12Hz-1.7B-CustomVoice`): Model preset voices đa ngôn ngữ.

---

## 🌟 Bảng So Sánh Model & Tính Năng

| Đặc điểm | OmniVoice (VoiceStudio - Mặc định) | Qwen3-TTS 1.7B |
| :--- | :--- | :--- |
| **Model ID** | `k2-fsa/OmniVoice` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` |
| **Ngôn ngữ** | **646 ngôn ngữ** (Auto, Tiếng Việt, Nhật, Anh, Trung, Hàn, Pháp, v.v.) | 10 ngôn ngữ chính |
| **Độ tự nhiên câu hỏi (?)** | ⭐⭐⭐⭐⭐ Cực mượt, ngữ điệu luyến láy tự nhiên | Thường bị flat / bẹt âm ở câu hỏi tiếng Việt |
| **Voice Cloning** | **Có (Zero-shot)**: Chỉ cần file âm thanh mẫu 3–10s | Không (chỉ dùng preset có sẵn) |
| **Preset Tiếng Nhật** | Anime Kawaii, Onee-san, NHK News, Ikemen, Shonen, Narrator,... | Ono_Anna, Hana, Yuki, Ken, Ryo,... |
| **VRAM GPU** | ~3.5 - 4 GB (Tự động thu hồi VRAM sau mỗi job: `TTS_FREE_VRAM=1`) | ~4 GB |

---

## 🚀 Cài Đặt & Khởi Động Nhanh (Setup)

### 1. Chuẩn bị môi trường (Python 3.10 – 3.12)

```powershell
cd qwen3-tts-api
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2. Cài đặt Dependencies (PyTorch CUDA 12 & Toàn bộ thư viện)

```powershell
# Cài đặt PyTorch với CUDA 12.1
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Cài đặt toàn bộ dependencies (OmniVoice, Qwen3-TTS, FastAPI,...)
python -m pip install -r requirements.txt

# Tạo file cấu hình từ mẫu
copy .env.example .env
```

### 3. Tải Model Weights (Tùy chọn)

> [!NOTE]
> **Không bắt buộc:** Model sẽ tự động tải từ Hugging Face trong lần gọi đầu tiên nếu chưa có sẵn.
> Nếu muốn tải sẵn về máy để chạy offline:
```powershell
python scripts/download_models.py --model omnivoice
# Hoặc tải cả 2 model: python scripts/download_models.py --model all
```

### 4. Khởi chạy Server

```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- 🌐 **Giao diện Web Form**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- 📖 **Swagger API Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🎌 Danh Sách Preset Giọng Nhật Bản Nổi Bật (OmniVoice)

| Preset Speaker | Phong cách giọng |
| :--- | :--- |
| `Japanese_Anime_Female` | Nữ ngọt ngào, tươi vui, phong cách Anime Kawaii / Idol |
| `Japanese_Warm_Female` | Nữ ấm áp, dịu dàng, phong cách chị gái Onee-san / ASMR |
| `Japanese_News_Female` | Nữ phát thanh viên đài NHK, trang trọng, chuẩn âm điệu |
| `Japanese_Story_Female` | Nữ kể chuyện / Radio Podcast diễn cảm, sâu lắng |
| `Japanese_Young_Male` | Nam trẻ trung, năng động, phong cách Shonen Anime |
| `Japanese_Ikemen_Male` | Nam trầm ấm, điềm tĩnh, phong cách lãng tử (Ikemen) |
| `Japanese_News_Male` | Nam phát thanh viên NHK đĩnh đạc, phát âm chuẩn xác |
| `Japanese_Narrator_Male` | Nam trung niên trầm sâu, dẫn chuyện phim tài liệu / lịch sử |

---

## 📡 API Endpoints & Gọi Mẫu

| Method | Path | Chức năng |
| :--- | :--- | :--- |
| `POST` | `/api/v1/generate` | Sinh file âm thanh WAV (chờ và trả về `audio_url` trong cùng 1 request) |
| `POST` | `/api/v1/jobs` | Tạo job và chờ kết quả |
| `GET` | `/api/v1/voices` | Lấy danh sách giọng đọc & ngôn ngữ (`?engine=omnivoice` hoặc `qwen3`) |
| `POST` | `/api/v1/free-memory` | Giải phóng VRAM GPU ngay lập tức |
| `DELETE` | `/api/v1/media?path=` | Xóa file âm thanh đã sinh |
| `GET` | `/media/audio/{file}` | Stream / Tải file audio WAV |
| `GET` | `/health` | Kiểm tra trạng thái GPU / Engine |

*Lưu ý: Mọi route `/api/v1/*` yêu cầu gửi header `api-key` trùng với `SECRET_API_KEY` trong file `.env`.*

### Ví dụ gọi API `curl`:

**1. Tạo giọng tiếng Việt (OmniVoice):**
```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/generate" `
  -H "api-key: change-me" `
  -F "text=Hôm nay bạn có những câu hỏi gì cần mình giải đáp không?" `
  -F "language=Vietnamese" `
  -F "engine=omnivoice" `
  -F "speed=1.0"
```

**2. Tạo giọng tiếng Nhật Anime Kawaii:**
```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/generate" `
  -H "api-key: change-me" `
  -F "text=こんにちは！今日の調子はいかがですか？何かお手伝いできることはありますか？" `
  -F "language=Japanese" `
  -F "speaker=Japanese_Anime_Female" `
  -F "engine=omnivoice"
```

**3. Voice Cloning (Nhân bản giọng từ file âm thanh mẫu 3–10s):**
```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/generate" `
  -H "api-key: change-me" `
  -F "text=Đoạn này được đọc bằng chính âm sắc từ file mẫu của bạn!" `
  -F "ref_audio_file=@sample_voice.wav" `
  -F "engine=omnivoice"
```

---

## 💡 Cấu hình Engine Mặc Định trong `.env`

```ini
# Chạy OmniVoice (VoiceStudio 646 languages - Khuyến nghị):
TTS_ENGINE=omnivoice
TTS_MODEL_ID=k2-fsa/OmniVoice

# Hoặc chuyển sang Qwen3-TTS:
# TTS_ENGINE=qwen3
# TTS_MODEL_ID=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice

# Tự động giải phóng VRAM sau mỗi lượt sinh audio (cho GPU 12GB dùng chung):
TTS_FREE_VRAM=1
```
