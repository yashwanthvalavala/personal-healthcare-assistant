import logging
import os
import pickle
import time
import base64
import io

from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer, util
import torch
import speech_recognition as sr
from deep_translator import GoogleTranslator
from langdetect import detect
from gtts import gTTS
from groq import Groq, GroqError
from PIL import Image
from functools import lru_cache
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configuration ---
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
EMBEDDINGS_PATH = "embeddings.pkl"
TEXTS_PATH = "texts.pkl"

# API Key
GROQ_API_KEY = "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"

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set")

# --- Globals ---
MODEL = None
TEXTS, EMBEDDINGS = None, None

# --- Utility Functions ---
def get_embeddings_model():
    global MODEL
    if MODEL is None:
        start_time = time.time()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        MODEL = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        logger.info(f"Model loaded on {device} in {time.time() - start_time:.2f}s")
    return MODEL

def load_embeddings():
    global TEXTS, EMBEDDINGS
    if TEXTS is None or EMBEDDINGS is None:
        try:
            if os.path.exists(EMBEDDINGS_PATH) and os.path.exists(TEXTS_PATH):
                with open(EMBEDDINGS_PATH, "rb") as f:
                    EMBEDDINGS = pickle.load(f)
                with open(TEXTS_PATH, "rb") as f:
                    TEXTS = pickle.load(f)
                logger.info("Embeddings and texts loaded.")
            else:
                logger.error("Embeddings or texts file not found.")
                TEXTS, EMBEDDINGS = [], None
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            TEXTS, EMBEDDINGS = [], None
    return TEXTS, EMBEDDINGS

@lru_cache(maxsize=100)
def cached_detect(text):
    try:
        return detect(text)
    except Exception as e:
        logger.error(f"Language detection error: {e}")
        return "en"

def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        logger.debug("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        try:
            logger.debug("Listening for audio input...")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            for lang in ["hi-IN", "en-US"]:
                try:
                    query = recognizer.recognize_google(audio, language=lang)
                    logger.info(f"Recognized query: {query} (lang: {lang})")
                    return query, "hi" if "hi" in lang else "en"
                except sr.UnknownValueError:
                    logger.debug(f"Speech not recognized in {lang}")
                    continue
        except sr.WaitTimeoutError:
            logger.warning("No speech detected within timeout.")
        except Exception as e:
            logger.error(f"Speech recognition error: {e}")
    return None, None

def text_to_speech(text, language, is_image=False):
    try:
        if not text or not isinstance(text, str):
            logger.error("Invalid text for TTS: Empty or not a string")
            return None
        logger.debug(f"Generating TTS for text: {text}, language: {language}")
        if language == "hi":
            detected = cached_detect(text)
            logger.debug(f"Detected language: {detected}")
            if detected != "hi" and not is_image:
                text = GoogleTranslator(source="en", target="hi").translate(text)
                logger.debug(f"Translated text to Hindi: {text}")
            tts = gTTS(text=text, lang="hi")
        else:
            tts = gTTS(text=text, lang="en")
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        audio_bytes = audio_buffer.read()
        logger.debug(f"Audio buffer size: {len(audio_bytes)} bytes")
        if len(audio_bytes) == 0:
            logger.error("Generated audio is empty")
            return None
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        return audio_base64
    except Exception as e:
        logger.error(f"TTS Error: {e}", exc_info=True)
        return None

def query_top_k(query, model, texts, embeddings, k=1):
    try:
        if not texts or embeddings is None:
            logger.error("No texts or embeddings available for query")
            return []
        query_embedding = model.encode(query, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
        top_indices = torch.topk(scores, k=min(k, len(texts))).indices
        return [texts[i] for i in top_indices]
    except Exception as e:
        logger.error(f"Query error: {e}")
        return []

def process_text_query(query, model, texts, embeddings):
    try:
        if not query.strip():
            return "Please enter a valid query."
        lang = cached_detect(query)
        logger.debug(f"Processing text query: {query}, detected lang: {lang}")
        query_en = GoogleTranslator(source="hi", target="en").translate(query) if lang == "hi" else query
        results = query_top_k(query_en, model, texts, embeddings)
        result_en = results[0] if results else "No relevant answer found."
        return GoogleTranslator(source="en", target="hi").translate(result_en) if lang == "hi" else result_en
    except Exception as e:
        logger.error(f"Text processing error: {e}")
        return "Error processing query."

def process_image(image_file, language):
    try:
        if not image_file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            logger.error(f"Invalid image format: {image_file.filename}")
            return {"error": "Invalid image format", "description": None, "audio_base64": None}
        image_file.seek(0, os.SEEK_END)
        file_size = image_file.tell()
        if file_size > 5 * 1024 * 1024:
            logger.error(f"Image too large: {file_size} bytes")
            return {"error": "Image size exceeds 5MB", "description": None, "audio_base64": None}
        image_file.seek(0)
        
        image_data = image_file.read()
        encoded = base64.b64encode(image_data).decode("utf-8")
        mime_type = "image/png" if image_file.filename.endswith(".png") else "image/jpeg"

        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the image"},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}}
                ]
            }]
        )
        description = response.choices[0].message.content
        logger.debug(f"Image description: {description}")
        audio_base64 = text_to_speech(description, language, is_image=True)
        return {"description": description, "audio_base64": audio_base64}
    except GroqError as e:
        logger.error(f"Groq error: {e}")
        return {"error": "Groq API failed", "description": None, "audio_base64": None}
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        return {"error": "Error processing image", "description": None, "audio_base64": None}

# --- Flask Routes ---
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    audio_base64 = None
    query = None
    diagnosis_type = None

    if request.method == "POST":
        diagnosis_type = request.form.get("diagnosis_type")
        model = get_embeddings_model()
        texts, embeddings = load_embeddings()

        if not texts or embeddings is None:
            error = "Embeddings not loaded. Please run precompute_embeddings.py."
            logger.error(error)
            return render_template("index.html", error=error, result=result, audio_base64=audio_base64, query=query, diagnosis_type=diagnosis_type)

        try:
            if diagnosis_type == "voice":
                query, lang = speech_to_text()
                if not query:
                    error = "Could not recognize speech."
                    logger.warning(error)
                else:
                    logger.debug(f"Voice query: {query}, language: {lang}")
                    query_en = GoogleTranslator(source="hi", target="en").translate(query) if lang == "hi" else query
                    logger.debug(f"Translated query (if Hindi): {query_en}")
                    results = query_top_k(query_en, model, texts, embeddings)
                    result = results[0] if results else "No relevant answer found."
                    logger.debug(f"Query result: {result}")
                    audio_base64 = text_to_speech(result, lang)
                    if not audio_base64:
                        error = "Failed to generate audio output."
                        logger.error(error)
                    else:
                        logger.info("Voice diagnosis processed successfully.")

            elif diagnosis_type == "image":
                lang = request.form.get("language", "en")
                image_file = request.files.get("image")
                if not image_file:
                    error = "No image uploaded."
                    logger.warning(error)
                else:
                    image_result = process_image(image_file, lang)
                    if image_result.get("error"):
                        error = image_result["error"]
                    else:
                        result = image_result["description"]
                        audio_base64 = image_result["audio_base64"]
                        logger.info("Image diagnosis processed successfully.")

            elif diagnosis_type == "text":
                query = request.form.get("text_query")
                lang = request.form.get("language", "en")
                if not query:
                    error = "Empty text query."
                    logger.warning(error)
                else:
                    result = process_text_query(query, model, texts, embeddings)
                    audio_base64 = text_to_speech(result, lang)
                    if not audio_base64:
                        error = "Failed to generate audio output."
                        logger.error(error)
                    else:
                        logger.info("Text diagnosis processed successfully.")

        except Exception as e:
            logger.error(f"Unhandled error: {e}", exc_info=True)
            error = "An unexpected error occurred."

    return render_template("index.html", error=error, result=result, audio_base64=audio_base64, query=query, diagnosis_type=diagnosis_type)

# --- Run App ---
if __name__ == "__main__":
    logger.info("Testing text_to_speech...")
    test_audio = text_to_speech("This is a test.", "en")
    if test_audio:
        logger.info(f"Test audio generated, base64 length: {len(test_audio)}")
    else:
        logger.error("Test audio generation failed.")
    logger.info("Starting Flask server...")
    print("Flask server starting on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)