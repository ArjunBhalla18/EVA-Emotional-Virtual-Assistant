import os
import uuid
import shutil
import asyncio
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import io
import json
import base64
from typing import Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv("gemini.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not OPENAI_API_KEY or not GEMINI_API_KEY:
    raise ValueError("Missing API keys. Please check your gemini.env file.")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="EVA Therapy Bot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/static", StaticFiles(directory=".", html=True), name="static")

# Whisper
logger.info("Loading Whisper model...")
whisper_model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8",
    num_workers=4
)
logger.info("Whisper model loaded successfully.")

# Gemini model with safety settings
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

gemini_model = genai.GenerativeModel(
    'gemini-1.5-flash',
    safety_settings=safety_settings,
    generation_config=genai.types.GenerationConfig(
        temperature=0.7,
        max_output_tokens=200, 
    )
)

# Conversation context
conversation_history = {}

def get_therapy_prompt(emotion: str, history: list = None) -> str:
    base_prompt = f"""You are EVA, a compassionate and professional AI therapy assistant. 

Current user emotion: {emotion}

Guidelines:
- Be warm, empathetic, and understanding
- Keep responses conversational and natural (2-3 sentences max)
- If emotion is 'sad' or 'distressed', offer comfort and validation
- If emotion is 'happy', share in their joy while being supportive
- If emotion is 'neutral', talk to them more to know how they feel
- Use therapeutic techniques like active listening and reflection
- Ask open-ended questions to encourage sharing
- Never provide medical diagnoses or replace professional therapy

"""
    
    if history and len(history) > 0:
        base_prompt += f"\nRecent conversation context:\n"
        for msg in history[-3:]:  # Last 3 exchanges for context
            base_prompt += f"User: {msg['user']}\nEVA: {msg['eva']}\n"
    
    base_prompt += "\nRespond naturally and therapeutically:"
    return base_prompt

async def get_gemini_response_with_emotion(text: str, emotion: str, session_id: str) -> str:
    try:
        history = conversation_history.get(session_id, [])
        prompt = get_therapy_prompt(emotion, history)
        full_prompt = f"{prompt}\n\nUser: {text}\nEVA:"
        
        response = await gemini_model.generate_content_async(full_prompt)
        reply = response.text.strip()
        
        # Update conversation history
        if session_id not in conversation_history:
            conversation_history[session_id] = []
        
        conversation_history[session_id].append({
            "user": text,
            "eva": reply,
            "emotion": emotion
        })
        
        # Keeping only last 10 exchanges to manage memory
        if len(conversation_history[session_id]) > 10:
            conversation_history[session_id] = conversation_history[session_id][-10:]
            
        return reply
        
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return "I'm here for you. Sometimes I need a moment to process. Could you tell me more about how you're feeling?"

#Audio processing
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

        segments, info = whisper_model.transcribe(
            temp_path, 
            beam_size=5,
            vad_filter=True,  # Voice activity detection
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        transcription = " ".join([segment.text.strip() for segment in segments])
        
        if not transcription.strip():
            transcription = "I couldn't hear you clearly. Could you please repeat that?"
            reply = "I'm having trouble hearing you. Could you please speak a bit louder or check your microphone?"
        else:
            reply = await get_gemini_response_with_emotion(transcription, emotion, session_id)

        return JSONResponse({
            "transcription": transcription,
            "reply": reply,
            "session_id": session_id,
            "emotion_detected": emotion
        })

    except Exception as e:
        logger.error(f"Audio processing error: {e}")
        return JSONResponse({
            "transcription": "Sorry, I had trouble processing your audio.",
            "reply": "I'm having some technical difficulties. Could you try typing your message instead?",
            "session_id": session_id,
            "error": True
        })
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Text processing
@app.post("/process-text/")
async def process_text(
    text: str = Form(...), 
    emotion: Optional[str] = Form("neutral"),
    session_id: Optional[str] = Form(None)
):
    if not session_id:
        session_id = str(uuid.uuid4())
    
    try:
        reply = await get_gemini_response_with_emotion(text, emotion, session_id)
        
        return JSONResponse({
            "reply": reply,
            "session_id": session_id,
            "emotion_detected": emotion
        })
        
    except Exception as e:
        logger.error(f"Text processing error: {e}")
        return JSONResponse({
            "reply": "I'm here to listen. Sometimes I need a moment to gather my thoughts. How are you feeling right now?",
            "session_id": session_id,
            "error": True
        })

# OpenAI TTS endpoint
@app.post("/generate-speech/")
async def generate_speech(text: str = Form(...)):
    """Generate high-quality speech using OpenAI TTS."""
    try:
        response = openai_client.audio.speech.create(
            model="tts-1", 
            voice="nova",  
            input=text,
            speed=0.9     
        )
        def generate_audio():
            for chunk in response.iter_bytes():
                yield chunk
        
        return StreamingResponse(
            generate_audio(),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=speech.mp3"}
        )
        
    except Exception as e:
        logger.error(f"TTS generation error: {e}")
        return JSONResponse({"error": "Speech generation failed"}, status_code=500)

# WebSocket real-time emotion updates
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_text()
            emotion_data = json.loads(data)
            
            if session_id in conversation_history:
                pass
                
            await websocket.send_text(json.dumps({
                "status": "emotion_received",
                "emotion": emotion_data.get("emotion", "neutral")
            }))
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session: {session_id}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "services": {
        "whisper": "loaded",
        "gemini": "configured",
        "openai": "configured"
    }}

# Cleanup endpoint
@app.delete("/session/{session_id}")
async def cleanup_session(session_id: str):
    """Clean up session data."""
    if session_id in conversation_history:
        del conversation_history[session_id]
        return {"status": "session_cleared"}
    return {"status": "session_not_found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
