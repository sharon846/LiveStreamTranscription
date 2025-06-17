import os
from flask import Flask, request, abort
import numpy as np
from faster_whisper import WhisperModel
import sounddevice as sd
from flask import Flask, request
import numpy as np
import threading
import queue

# Initialize the Flask app
# Flask is a lightweight web framework that will allow our Chrome extension
# to send data to this Python script.
app = Flask(__name__)
pcm_queue = queue.Queue()

# ─── Configuration ───────────────────────────────────────────────────────────────
MODEL_NAME    = "ivrit-ai/whisper-large-v3-turbo-ct2"
DEVICE        = "cuda" if sd.query_hostapis()[sd.default.hostapi]['name'].lower() == "asio" else "cpu"
COMPUTE_TYPE  = "int8" if DEVICE == "cpu" else "float16"

# ─── Load Faster-Whisper Model ───────────────────────────────────────────────
print(f"📦 Loading {MODEL_NAME} on {DEVICE} (compute_type={COMPUTE_TYPE})…")
model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)

# ─── Transcription Function ───────────────────────────────────────────────────
def transcribe(pcm):
    # Whisper expects NumPy float32 at 16kHz mono
    segments, _ = model.transcribe(pcm, language="he")
    return " ".join(segment.text for segment in segments).strip()

# This is the main endpoint that the Chrome extension will send audio data to.
# It supports POST requests to '/upload_audio'.
@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    tab_title = request.args.get('tabTitle')
    if not tab_title:
        abort(400, description="Missing 'tabTitle' query parameter.")

    raw = request.data
    if not request.data:
        return "OK", 200
    
    pcm_queue.put((tab_title, raw))
    return "QUEUED", 200

def transcription_worker():
    while True:
        tab_title, raw = pcm_queue.get()
        try:
            # Decode PCM
            pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            # Transcribe
            result = transcribe(pcm)
            print(f"🗣️ [{tab_title}] {result}")
        except Exception as e:
            print(f"❌ Error processing {tab_title}: {e}")
        finally:
            pcm_queue.task_done()

# This function starts the Flask server.
def run_server():
    """
    Starts the Flask development server.
    - host='127.0.0.1' makes it only accessible from your own computer.
    - port=5000 is the standard port for local development.
    """
    print("Python Audio Capture Server is starting...")
    print("Ready to receive audio from the Chrome Extension.")
    print("Press CTRL+C to stop the server.")
    # The 'app.run' command starts the server and blocks until you stop it.
    app.run(host='127.0.0.1', port=5000)

threading.Thread(target=transcription_worker, daemon=True).start()

if __name__ == '__main__':
    run_server()
