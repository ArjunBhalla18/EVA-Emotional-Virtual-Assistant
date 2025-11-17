# EVA: Emotional Virtual Assistant

EVA or Emotional Virtual Assistant is a sophisticated, multimodal AI companion designed to offer empathetic and therapeutic conversations in real-time. By integrating advanced speech recognition, natural language processing, text-to-speech, and real-time facial emotion analysis, EVA provides a supportive and interactive experience for users seeking a compassionate listener and an easily available online therapist.

EVA version- 1.0.0 (first prototype)

---

## Core Features

* **Multimodal Interaction**: Seamlessly switch between **voice** and **text** input for natural conversation.
* **Real-time Emotion Detection**: Utilizes the user's webcam and TensorFlow.js to analyze facial expressions, allowing EVA to adapt its responses based on the user's emotional state (e.g., happy, sad, neutral).
* **Empathetic Conversational AI**: Powered by **Google Gemini API**, EVA generates context-aware, warm, and therapeutic responses tailored to the conversation history and detected emotions.
* **High-Fidelity Voice Output**: Leverages **ElevenLab's Text-to-Speech (TTS)** API to produce realistic and natural-sounding vocal responses, enhancing the conversational experience.
* **Accurate Speech-to-Text**: Integrates the `faster-whisper` model for efficient and precise transcription of spoken words.
* **Intelligent Session Management**: Maintains a coherent conversation flow by tracking history for each session, ensuring context is not lost.

---

## Tech Stack & Architecture

EVA is built with a Python backend and a vanilla JavaScript frontend, orchestrated by a simple-to-use launch script.

* **Backend**: **FastAPI** running on **Uvicorn** ASGI server.
* **Frontend**: **HTML**, **Tailwind CSS**, **JavaScript**.
* **Other**:
    * **Response Generation**: Google Gemini (`gemini-1.5-flash`)
    * **Speech-to-Text (STT)**: `faster-whisper`
    * **Text-to-Speech (TTS)**: ElevenLab TTS
    * **Facial Emotion Recognition**: **TensorFlow.js** with MediaPipe Face Mesh

### Conversation Flow
1.  **User Input**: The user can either type a message or record their voice.
2.  **Emotion Analysis**: Simultaneously, the frontend uses TensorFlow.js to detect the user's facial emotion via their webcam.
3.  **Backend Processing**:
    * **Voice**: The audio is sent to the backend, transcribed by `faster-whisper`.
    * **Text**: The typed text is sent directly.
4.  **AI Response Generation**: The transcribed text, along with the detected emotion and recent conversation history, is sent to the **Gemini API** to generate an empathetic response.
5.  **Voice Synthesis**: The generated text response is converted into speech using the **ElevenLab TTS API**.
6.  **Frontend Output**: The text response is displayed in the chat, and the synthesized audio is played back to the user.

---

## Requirements & Dependencies

#### System
* **Python**: Version 3.8 or newer

#### API Keys
* **Google Gemini API Key**: For AI response generation.
* **ELevenLabs API Key**: For text-to-speech functionality.
  
#### Core Libraries
* **Backend**: `FastAPI`, `Uvicorn`
* **AI/ML**: `google-generativeai`, `ElevenLabs`, `faster-whisper`
* **Frontend**: `TensorFlow.js` (for emotion detection)

---

## Acknowledgements

* Google Gemini for the incredible generative AI capabilities.
* ElevenLabs for the high-quality TTS.
* OpenAI for the Whisper models.
* TensorFlow.js and MediaPipe for making client-side ML accessible.
* FastAPI for enabling the creation of fast, modern Python web services.
