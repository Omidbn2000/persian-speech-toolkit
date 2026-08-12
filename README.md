```markdown
# Real-time Persian Speech Processing Toolkit

A complete pipeline for Voice Activity Detection (VAD), Automatic Speech Recognition (ASR), and Text-to-Speech (TTS) — all optimized for Persian language, running entirely on CPU.

---

## ✨ Features

- 🎤 **Real-time VAD** – Silero VAD with pre-roll buffer, hangover, and automatic speech segment saving  
- 📝 **Real-time ASR** – Shenava‑Koochik v1.0 ONNX model, Persian-optimized, with timestamped transcript saving  
- 🔊 **TTS** – Mana‑Persian‑Piper for natural Persian speech synthesis  
- 🧵 **Clean architecture** – Separate engines for VAD, ASR, TTS; ASR inference runs outside the audio callback  
- ⚙️ **Fully configurable** – All parameters in `config.json` with debug toggles  
- 💾 **Session-based output** – Each run creates a timestamped folder for transcripts and saved audio segments  
- 📋 **Combined transcripts** – TXT, plain text, continuous text, and JSON exports  

---

## 📁 Project Structure

```
.
├── assets/
│   ├── audios/vad_segments/   # Saved speech segments (per session)
│   └── transcripts/           # ASR transcripts (per session)
├── config/
│   └── config.json            # Main configuration file
├── models/
│   ├── Mana-Persian-Piper/    # Persian TTS model (Piper)
│   ├── Shenava-Koochik-v1.0/  # Original NeMo checkpoint (not used at runtime)
│   └── Shenava-Koochik-v1.0-ONNX-fp16/  # ONNX ASR model (used)
├── src/
│   ├── main.py                # Entry point
│   ├── config.py              # Configuration loader
│   ├── tts/engine.py          # TTS engine (Piper)
│   ├── asr/engine.py          # ASR engine (ONNX)
│   └── vad/engine.py          # VAD engine (Silero)
├── requirements.txt
└── README.md
```

---

## 🧠 Models

### Automatic Speech Recognition (ASR)
**Shenava‑Koochik v1.0 ONNX fp16**  
- Persian CTC ASR, 114M parameters, ONNX runtime  
- Download from: [Reza2kn/Shenava-Koochik-v1.0-ONNX-fp16](https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0-ONNX-fp16)  
- Place the entire repository folder under `models/Shenava-Koochik-v1.0-ONNX-fp16/`  
- Required files:  
  - `shenava_koochik_1_0_ctc_fixed2005_len_att70_13_fp16_full_io_embedded.onnx`  
  - `tokens.json`  
  - `mel_filters_slaney_80x257.json`

### Text-to-Speech (TTS)
**Mana‑Persian‑Piper**  
- Persian Piper voice, medium quality  
- Download from: [Mana-Persian-Piper on Hugging Face](https://huggingface.co/Reza2kn/Mana-Persian-Piper)  
- Place the model files inside `models/Mana-Persian-Piper/`  
- Required files:  
  - `fa_IR-mana-medium.onnx`  
  - `fa_IR-mana-medium.onnx.json`  
  - `ckpt/config.json` (optional, for reference)

### Voice Activity Detection (VAD)
**Silero VAD**  
- Downloaded automatically on first run (via `silero-vad` package). No manual setup needed.

---

## 🔧 Installation

### Prerequisites
- Python **3.12** (other versions may cause compatibility issues with some dependencies)  
- `pip` and a working microphone

### Steps
1. Clone or download this repository.  
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the required models (see [Models section](#models)) and place them in the correct folders.  
4. (Optional) Adjust `config/config.json` to your preferences.

---

## ⚙️ Configuration

All settings are in `config/config.json`. Key sections:

| Section | Description |
|---------|-------------|
| `general` | `debug_asr`, `debug_vad`, `debug_main` (verbose logs), `tts_enabled` |
| `model_paths` | Paths to ONNX models and tokenizer files |
| `vad` | Sample rate, thresholds, saving options |
| `asr` | Transcripts directory, frame length |
| `segmentation` | Pre‑roll, speech start/silence end blocks, minimum segment duration |

---

## 🚀 Usage

**Run from the project root directory:**

```bash
py -m src.main
```

The program will:
1. Load all engines (TTS, VAD, ASR).  
2. Open the microphone and start continuous speech recognition.  
3. Display a real‑time status bar with VAD confidence, speech state, and segment count.  
4. Automatically transcribe detected speech segments and save transcripts.  
5. If `tts_enabled` is `true` in config, speak the transcribed text aloud.  
6. Press `Ctrl+C` to stop; combined transcript files will be generated.

### Output Structure

Each run creates timestamped directories:

- **Transcripts**: `assets/transcripts/YYYYMMDD_HHMMSS/`
  - `segment_XXXX.txt` – individual transcriptions  
  - `all_transcripts_...` – combined, plain, continuous, JSON  
- **Saved audio (VAD)**: `assets/audios/vad_segments/YYYYMMDD_HHMMSS/` (if `save_audio` is enabled)  
  - `segment_XXXX.wav` – raw speech segments

---

## 👤 Contact

**Omid Babaeinejad**  
📧 omidbn2000@gmail.com  
📱 +98 936 402 0724  

For questions, suggestions, or contributions, feel free to reach out.

---

## 📄 License

This project itself is unlicensed. However, the included models have their own licenses:

- Shenava-Koochik: Apache 2.0  
- Mana-Persian-Piper: Apache 2.0  
- Silero VAD: MIT  

Please respect the respective licenses when using or distributing the models.

---

<div dir="rtl">

# ابزار پردازش گفتار فارسی در زمان واقعی

یک خط‌لوله کامل برای تشخیص فعالیت صوتی (VAD)، بازشناسی خودکار گفتار (ASR) و تبدیل متن به گفتار (TTS) — بهینه‌سازی شده برای زبان فارسی و اجرای کامل روی CPU.

---

## ✨ ویژگی‌ها

- 🎤 **تشخیص فعالیت صوتی در زمان واقعی** – مدل Silero VAD با بافر پیش‌آماده‌سازی، تأخیر قطع و ذخیره خودکار قطعات گفتاری  
- 📝 **بازشناسی گفتار فارسی** – مدل ONNX شنوا‑کوچیک نسخه ۱، بهینه برای فارسی، با ذخیره رونوشت‌ها با برچسب زمان  
- 🔊 **تبدیل متن به گفتار** – مانا‑پرشین‑پایپر برای تولید گفتار طبیعی فارسی  
- 🧵 **معماری تمیز** – موتورهای مجزا برای VAD، ASR و TTS؛ استنتاج ASR خارج از حلقه صوتی انجام می‌شود  
- ⚙️ **قابل تنظیم کامل** – تمام پارامترها در فایل `config.json` با گزینه‌های اشکال‌زدایی  
- 💾 **خروجی مبتنی بر جلسه** – هر اجرا یک پوشه با زمان اجرا برای رونوشت‌ها و قطعات صوتی ذخیره شده ایجاد می‌کند  
- 📋 **خروجی‌های ترکیبی** – فرمت‌های TXT، متن ساده، متن پیوسته و JSON  

---

## 📁 ساختار پروژه

```
.
├── assets/
│   ├── audios/vad_segments/   # قطعات گفتاری ذخیره شده (در هر جلسه)
│   └── transcripts/           # رونوشت‌های ASR (در هر جلسه)
├── config/
│   └── config.json            # فایل پیکربندی اصلی
├── models/
│   ├── Mana-Persian-Piper/    # مدل TTS فارسی (Piper)
│   ├── Shenava-Koochik-v1.0/  # چک‌پوینت اصلی NeMo (در زمان اجرا استفاده نمی‌شود)
│   └── Shenava-Koochik-v1.0-ONNX-fp16/  # مدل ONNX ASR (مورد استفاده)
├── src/
│   ├── main.py                # نقطه ورود
│   ├── config.py              # بارگذار تنظیمات
│   ├── tts/engine.py          # موتور TTS
│   ├── asr/engine.py          # موتور ASR
│   └── vad/engine.py          # موتور VAD
├── requirements.txt
└── README.md
```

---

## 🧠 مدل‌ها

### بازشناسی خودکار گفتار (ASR)
**شنوا‑کوچیک نسخه ۱ ONNX fp16**  
- مدل CTC فارسی، ۱۱۴ میلیون پارامتر، اجرا با ONNX Runtime  
- دانلود از: [Reza2kn/Shenava-Koochik-v1.0-ONNX-fp16](https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0-ONNX-fp16)  
- کل پوشه مخزن را در مسیر `models/Shenava-Koochik-v1.0-ONNX-fp16/` قرار دهید.  
- فایل‌های ضروری:  
  - `shenava_koochik_1_0_ctc_fixed2005_len_att70_13_fp16_full_io_embedded.onnx`  
  - `tokens.json`  
  - `mel_filters_slaney_80x257.json`

### تبدیل متن به گفتار (TTS)
**مانا‑پرشین‑پایپر**  
- صدای فارسی Piper با کیفیت متوسط  
- دانلود از: [Mana-Persian-Piper در Hugging Face](https://huggingface.co/Reza2kn/Mana-Persian-Piper)  
- فایل‌های مدل را در `models/Mana-Persian-Piper/` قرار دهید.  
- فایل‌های ضروری:  
  - `fa_IR-mana-medium.onnx`  
  - `fa_IR-mana-medium.onnx.json`  
  - `ckpt/config.json` (اختیاری)

### تشخیص فعالیت صوتی (VAD)
**Silero VAD**  
- به‌طور خودکار در اولین اجرا دانلود می‌شود. نیازی به تنظیم دستی نیست.

---

## 🔧 نصب

### پیش‌نیازها
- پایتون **۳.۱۲** (نسخه‌های دیگر ممکن است مشکل سازگاری ایجاد کنند)  
- `pip` و یک میکروفون سالم

### مراحل
۱. مخزن را کلون یا دانلود کنید.  
۲. وابستگی‌ها را نصب کنید:
   ```bash
   pip install -r requirements.txt
   ```
۳. مدل‌های مورد نیاز را دانلود کرده و در پوشه‌های مناسب قرار دهید.  
۴. (اختیاری) فایل `config/config.json` را مطابق میل خود تنظیم کنید.

---

## ⚙️ پیکربندی

تمام تنظیمات در `config/config.json` قرار دارند. بخش‌های کلیدی:

| بخش | توضیحات |
|-----|---------|
| `general` | `debug_asr`، `debug_vad`، `debug_main` (گزارش‌های دقیق)، `tts_enabled` |
| `model_paths` | مسیر فایل‌های ONNX و توکنایزر |
| `vad` | نرخ نمونه‌برداری، آستانه‌ها، گزینه‌های ذخیره‌سازی |
| `asr` | پوشه رونوشت‌ها، طول فریم |
| `segmentation` | پیش‌آماده‌سازی، بلوک‌های شروع گفتار/پایان سکوت، حداقل طول قطعه |

---

## 🚀 نحوه استفاده

**از پوشه ریشه پروژه اجرا کنید:**

```bash
py -m src.main
```

برنامه:
۱. تمام موتورها (TTS، VAD، ASR) را بارگذاری می‌کند.  
۲. میکروفون را باز کرده و بازشناسی پیوسته گفتار را آغاز می‌کند.  
۳. یک نوار وضعیت زنده با میزان اطمینان VAD، وضعیت گفتار و تعداد قطعات نمایش می‌دهد.  
۴. قطعات گفتاری تشخیص داده شده را به‌طور خودکار رونویسی و ذخیره می‌کند.  
۵. اگر `tts_enabled` در تنظیمات `true` باشد، متن رونویسی شده را با صدای بلند می‌خواند.  
۶. با فشردن `Ctrl+C` متوقف می‌شود؛ فایل‌های ترکیبی رونوشت تولید خواهند شد.

### ساختار خروجی

هر اجرا پوشه‌هایی با برچسب زمان ایجاد می‌کند:

- **رونوشت‌ها**: `assets/transcripts/YYYYMMDD_HHMMSS/`
  - `segment_XXXX.txt` – رونوشت‌های جداگانه  
  - `all_transcripts_...` – ترکیبی، ساده، پیوسته، JSON  
- **صدای ذخیره شده (VAD)**: `assets/audios/vad_segments/YYYYMMDD_HHMMSS/` (اگر `save_audio` فعال باشد)  
  - `segment_XXXX.wav` – قطعات گفتاری خام

---

## 👤 تماس

**امید بابایی‌نژاد**  
📧 omidbn2000@gmail.com  
📱 ۰۹۳۶۴۰۲۰۷۲۴  

برای سوالات، پیشنهادات یا مشارکت، در تماس باشید.

---

## 📄 مجوز

این پروژه خود فاقد مجوز است. با این حال، مدل‌های استفاده شده مجوزهای خاص خود را دارند:

- Shenava-Koochik: Apache 2.0  
- Mana-Persian-Piper: Apache 2.0  
- Silero VAD: MIT  

لطفاً هنگام استفاده یا توزیع مدل‌ها، مجوزهای مربوطه را رعایت کنید.

</div>
```