# 🌾 CropGenie — HarvestAI

**AI Agriculture Assistant** — a full-stack application that helps farmers get expert advice on crops, diseases, weather, fertilizer, and more through an intuitive chat interface with voice support.

## Tech Stack

| Layer     | Technology                              |
|-----------|-----------------------------------------|
| Backend   | **FastAPI** (Python)                    |
| Frontend  | **Streamlit**                           |
| AI Engine | **Google Gemini 1.5 Flash**             |
| Speech→Text | **SpeechRecognition** (Google Web Speech API — free) |
| Text→Speech | **gTTS** (Google Text-to-Speech — free) |

### Why this combination?

- **FastAPI** — async, type-safe, auto-generates API docs, production-ready
- **Streamlit** — fastest way to build a professional multi-page data/AI app with minimal code
- **Google Web Speech API** — free, no API key needed, supports Urdu, Pashto, English
- **gTTS** — free, high-quality voice synthesis, supports the same languages

## Project Structure

```
CropGenie-HarvestAI/
├── backend/
│   ├── app.py              # FastAPI entry point
│   ├── config.py           # Environment variable config (pydantic-settings)
│   ├── routes/
│   │   ├── chat.py         # POST /api/chat — Gemini AI conversations
│   │   └── speech.py       # POST /api/speech-to-text, /api/text-to-speech
│   ├── services/
│   │   ├── gemini_service.py   # Gemini API wrapper + agriculture prompt
│   │   └── speech_service.py   # STT (SpeechRecognition) + TTS (gTTS)
│   ├── models/
│   │   └── schemas.py      # Pydantic request/response models
│   └── utils/              # Shared helpers (expand as needed)
├── frontend/
│   └── app.py              # Streamlit chat UI with voice I/O
├── data/                   # Datasets (added in future features)
├── .env.example            # Environment variable template
├── .gitignore
├── requirements.txt
└── README.md
```

## Prerequisites

- **Python 3.10+**
- **pip** (Python package manager)
- A **Google Gemini API key** — [get one free at Google AI Studio](https://aistudio.google.com/apikey)

## Quick Start

### 1. Clone & navigate

```bash
cd CropGenie-HarvestAI
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows (PowerShell):
venv\Scripts\Activate.ps1

# macOS / Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
# Copy the template
cp .env.example .env
```

Edit `.env` and add your Gemini API key:

```
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

> ⚠️ **Never commit your `.env` file** — it's in `.gitignore` for safety.

### 5. Start the backend

Open **Terminal 1**:

```bash
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

Verify it's running: [http://localhost:8000/health](http://localhost:8000/health)
View API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### 6. Start the frontend

Open **Terminal 2** (with venv activated):

```bash
streamlit run frontend/app.py
```

The app opens at [http://localhost:8501](http://localhost:8501).

## Features

### ✅ Implemented

- **Text Chat** — Ask farming questions, get AI-powered answers with conversation context
- **Voice Input** — Record your question via microphone; supports English, Urdu, Pashto
- **Voice Output** — AI responses are spoken aloud; auto-play with mute toggle
- **Language Selector** — Switch between English (US/UK/India), Urdu, and Pashto
- **Farmer-Friendly UI** — Large buttons, chat bubbles, status indicators

### 🗺️ Roadmap

- Crop recommendation engine (soil + climate data)
- Disease detection via image upload (CNN model)
- Weather integration (OpenWeatherMap API)
- Market price tracking
- Multi-page navigation with feature sidebar

## API Reference

| Method | Endpoint                  | Description                     |
|--------|---------------------------|---------------------------------|
| GET    | `/health`                 | Health check                    |
| POST   | `/api/chat`               | Send message, get AI reply      |
| POST   | `/api/speech-to-text`     | Upload audio → transcribed text |
| POST   | `/api/text-to-speech`     | Text → MP3 audio bytes          |

Full interactive docs available at `/docs` (Swagger UI) when backend is running.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| "Cannot reach the backend" | Make sure `uvicorn` is running on port 8000 |
| "GEMINI_API_KEY is not set" | Check your `.env` file exists and has the key |
| Mic not working | Grant browser microphone permissions |
| Speech not recognised | Speak clearly; try a different language setting |

## License

Proprietary — prepared for client delivery.
