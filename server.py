import os
from flask import Flask, request, abort
import io
import numpy as np
import av
from faster_whisper import WhisperModel
import sounddevice as sd
import ffmpeg

# Initialize the Flask app
# Flask is a lightweight web framework that will allow our Chrome extension
# to send data to this Python script.
app = Flask(__name__)

# ─── Configuration ───────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16000
CHUNK_SECONDS = 8
SHIFT_SECONDS = 4
MODEL_NAME    = "ivrit-ai/whisper-large-v3-turbo-ct2"
DEVICE        = "cuda" if sd.query_hostapis()[sd.default.hostapi]['name'].lower() == "asio" else "cpu"
COMPUTE_TYPE  = "int8" if DEVICE == "cpu" else "float16"


# ─── Load Faster-Whisper Model ───────────────────────────────────────────────
print(f"📦 Loading {MODEL_NAME} on {DEVICE} (compute_type={COMPUTE_TYPE})…")
model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)


# --- Configuration ---
# Directory where the raw audio files will be saved.
# Make sure this directory exists.
OUTPUT_DIR = "captured_audio"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory: {OUTPUT_DIR}")

# ─── Transcription Function ───────────────────────────────────────────────────
def transcribe(audio_np: np.ndarray) -> str:
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    segments, _ = model.transcribe(audio_np, language="he")
    return " ".join(segment.text for segment in segments).strip()


def decode_webm_chunk_ffmpeg(data: bytes) -> np.ndarray:
    try:
        out, _ = (
            ffmpeg
            .input('pipe:0')
            .output('pipe:1', format='f32le', acodec='pcm_f32le', ac=1, ar=16000)
            .run(input=data, capture_stdout=True, capture_stderr=True)
        )
        audio_np = np.frombuffer(out, dtype=np.float32)
        return audio_np
    except ffmpeg.Error as e:
        print("FFmpeg error:", e.stderr.decode())
        return np.array([])

# This is the main endpoint that the Chrome extension will send audio data to.
# It supports POST requests to '/upload_audio'.
@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    tab_title = request.args.get('tabTitle')
    if not tab_title:
        abort(400, description="Missing 'tabTitle' query parameter.")

    if not request.data:
        return "OK", 200

    # Save to disk
    safe_title = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in tab_title).strip()
    file_path = os.path.join(OUTPUT_DIR, f"{safe_title}_audio.webm")

    try:
        with open(file_path, 'ab') as f:
            f.write(request.data)
    except IOError as e:
        abort(500, description=f"Write failed: {e}")

    #audio_np = decode_webm_chunk_ffmpeg(request.data)

    #if audio_np.size > 0:
    #    text = transcribe(audio_np)
    #    print(f"[{tab_title}] {text}")

    return "OK", 200

# This function starts the Flask server.
def run_server():
    """
    Starts the Flask development server.
    - host='127.0.0.1' makes it only accessible from your own computer.
    - port=5000 is the standard port for local development.
    """
    print("Python Audio Capture Server is starting...")
    print(f"Audio will be saved in the '{OUTPUT_DIR}' directory.")
    print("Ready to receive audio from the Chrome Extension.")
    print("Press CTRL+C to stop the server.")
    # The 'app.run' command starts the server and blocks until you stop it.
    app.run(host='127.0.0.1', port=5000)


if __name__ == '__main__':
    run_server()
