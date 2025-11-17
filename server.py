import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import whisper
import io
import json
from typing import Optional
import logging
import google.generativeai as genai
from elevenlabs import ElevenLabs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")

load_dotenv("gemini.env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
TTS_VOICE = os.getenv("TTS_VOICE", "21m00Tcm4TlvDq8ikWAM") 

if not GEMINI_API_KEY:
    raise ValueError("❌ Missing GEMINI_API_KEY in gemini.env.")
if not ELEVENLABS_API_KEY:
    logger.warning("⚠️ ELEVENLABS_API_KEY missing. TTS will not work.")

genai.configure(api_key=GEMINI_API_KEY)
tts_client = ElevenLabs(api_key=ELEVENLABS_API_KEY) if ELEVENLABS_API_KEY else None
gemini_model = genai.GenerativeModel("models/gemini-flash-latest")

app = FastAPI(title="EVA Therapy Bot (Gemini Version)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=".", html=True), name="static")

logger.info("Loading Whisper model...")
whisper_model = whisper.load_model("small")
logger.info("✅ Whisper model loaded successfully.")

conversation_history = {}

def get_therapy_prompt(emotion: str, history: list = None) -> str:
    """Generate a soft, empathetic therapy context for EVA."""
    base_prompt = f"""
You are EVA, a compassionate and emotionally intelligent AI therapist.
Your tone should be warm, caring, and understanding.

Current user emotion: {emotion}

Guidelines:
- Speak with empathy, not clinical detachment.
- Keep responses natural (2-3 sentences).
- Offer gentle reflections, comfort, or encouragement.
- If emotion is 'sad', be comforting and kind.
- If emotion is 'happy', celebrate with warmth.
- If 'neutral', ask gentle questions to explore feelings.
- Avoid robotic and overly formal responses.
"""
    if history and len(history) > 0:
        base_prompt += "\n\nRecent conversation:\n"
        for msg in history[-3:]:
            base_prompt += f"User: {msg['user']}\nEVA: {msg['eva']}\n"

    base_prompt += "\nRespond naturally as EVA:"
    return base_prompt.strip()

async def get_gemini_response(text: str, emotion: str, session_id: str) -> str:
    """Generate empathetic response using Gemini 1.5 Flash."""
    try:
        history = conversation_history.get(session_id, [])
        prompt = get_therapy_prompt(emotion, history)
        full_prompt = f"{prompt}\n\nUser: {text}\nEVA:"

        response = gemini_model.generate_content(full_prompt)
        reply = response.text.strip() if hasattr(response, "text") else "I'm here for you."

        conversation_history.setdefault(session_id, []).append({
            "user": text,
            "eva": reply,
            "emotion": emotion
        })
        conversation_history[session_id] = conversation_history[session_id][-10:]

        return reply or "I'm here for you. Could you tell me a bit more about what’s going on?"

    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return "I'm really sorry, I'm having trouble responding right now. Could you tell me more about how you're feeling?"

@app.post("/process-audio/")
async def process_audio(
    file: UploadFile,
    emotion: Optional[str] = Form("neutral"),
    session_id: Optional[str] = Form(None)
):
    if not session_id:
        session_id = str(uuid.uuid4())

    temp_path = f"temp_{uuid.uuid4()}.wav"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = whisper_model.transcribe(temp_path)
        transcription = result["text"].strip()

        if not transcription:
            transcription = "I couldn’t hear you clearly."
            reply = "I'm having trouble hearing you. Could you repeat that?"
        else:
            reply = await get_gemini_response(transcription, emotion, session_id)

        return JSONResponse({
            "transcription": transcription,
            "reply": reply,
            "session_id": session_id,
            "emotion_detected": emotion
        })

    except Exception as e:
        logger.error(f"Audio processing error: {e}")
        return JSONResponse({
            "transcription": "Error processing audio.",
            "reply": "I couldn’t process that. Could you try again?",
            "session_id": session_id,
            "error": True
        })
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/process-text/")
async def process_text(
    text: str = Form(...),
    emotion: Optional[str] = Form("neutral"),
    session_id: Optional[str] = Form(None)
):
    if not session_id:
        session_id = str(uuid.uuid4())

    try:
        reply = await get_gemini_response(text, emotion, session_id)
        return JSONResponse({
            "reply": reply,
            "session_id": session_id,
            "emotion_detected": emotion
        })
    except Exception as e:
        logger.error(f"Text processing error: {e}")
        return JSONResponse({
            "reply": "I'm here for you. Let's take a moment to breathe together.",
            "session_id": session_id,
            "error": True
        })

@app.post("/generate-speech/")
async def generate_speech(text: str = Form(...)):
    """Convert text into speech using ElevenLabs."""
    try:
        if not tts_client:
            raise ValueError("Missing ELEVENLABS_API_KEY.")

        response = tts_client.text_to_speech.convert(
            voice_id=TTS_VOICE,
            model_id="eleven_turbo_v2_5",
            text=text,
            output_format="mp3_44100_128"
        )

        audio_data = b"".join(response)
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=speech.mp3"}
        )

    except Exception as e:
        logger.error(f"TTS generation error: {e}")
        return JSONResponse({"error": "Speech generation failed"}, status_code=500)


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            emotion_data = json.loads(data)
            await websocket.send_text(json.dumps({
                "status": "emotion_received",
                "emotion": emotion_data.get("emotion", "neutral")
            }))
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session: {session_id}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "services": {
            "whisper": "loaded",
            "gemini": "configured",
            "tts": "ready" if ELEVENLABS_API_KEY else "missing"
        }
    }

@app.delete("/session/{session_id}")
async def cleanup_session(session_id: str):
    if session_id in conversation_history:
        del conversation_history[session_id]
        return {"status": "session_cleared"}
    return {"status": "session_not_found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
