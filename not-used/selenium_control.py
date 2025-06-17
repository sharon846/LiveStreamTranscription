import numpy as np
import sounddevice as sd
import threading
import queue
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from faster_whisper import WhisperModel
import subprocess
import os
import time


# === CONFIG ===
SAMPLE_RATE   = 16000
CHUNK_SECONDS = 8
SHIFT_SECONDS = 4
CHUNK_SIZE = SAMPLE_RATE * CHUNK_SECONDS
SHIFT_SIZE = SAMPLE_RATE * SHIFT_SECONDS

MODEL_NAME    = "ivrit-ai/whisper-large-v3-turbo-ct2"
DEVICE        = "cuda" if sd.query_hostapis()[sd.default.hostapi]['name'].lower() == "asio" else "cpu"
COMPUTE_TYPE  = "int8" if DEVICE == "cpu" else "float16"

# Live pages you want to listen to
PAGES = [
    {"title": "kan", "url": "https://www.kan.org.il/live/", "device_name": "CABLE-C Output"},
    {"title": "n12", "url": "https://www.mako.co.il/mako-vod-live-tv/VOD-6540b8dcb64fd31006.htm", "device_name": "CABLE-D Output"},
]

import subprocess
import os

def launch_chrome_tab(url: str, profile_name: str):
    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    user_data_dir = os.path.abspath(f"./chrome_profiles/{profile_name}")
    os.makedirs(user_data_dir, exist_ok=True)

    return subprocess.Popen([
        chrome_path,
        f'--user-data-dir={user_data_dir}',
        '--new-window',
        url
    ])

from pycaw.pycaw import AudioUtilities
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import IAudioSessionManager2, IAudioSessionControl2

def get_chrome_sessions():
    sessions = AudioUtilities.GetAllSessions()
    chrome_sessions = []
    for session in sessions:
        ctl = session._ctl.QueryInterface(IAudioSessionControl2)
        process = ctl.GetProcessId()
        name = session.Process and session.Process.name() or None
        if name and "chrome" in name.lower():
            chrome_sessions.append((ctl.GetProcessId(), session))
    return chrome_sessions

'''

model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)

# ─── Transcription Function ───────────────────────────────────────────────────
def transcribe(audio_np, link_id):
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    segments, _ = model.transcribe(audio_np, language="he")
    txt = " ".join(segment.text for segment in segments).strip()
    sentence = [word[::-1] for word in txt.split()]
    sentence = " ".join(sentence)

    ts  = time.strftime("%H:%M:%S", time.localtime())
    print(f"[{ts}] [{link_id}] {sentence}")

chunk_queues = {}

def launch_chrome_with_device(url, device_name, profile_id):
    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    profile_dir = f"C:\\Temp\\ChromeProfile{profile_id}"

    os.makedirs(profile_dir, exist_ok=True)

    # Launch Chrome process
    proc = subprocess.Popen([
        chrome_path,
        "--new-window",
        "--user-data-dir=" + profile_dir,
        "--autoplay-policy=no-user-gesture-required",
        url
    ])

    pid = proc.pid

    # Wait a bit for Chrome to fully initialize
    time.sleep(4)

    # Use SoundVolumeView to redirect Chrome's audio output to specific device
    subprocess.run([
        "C:\soundvolumeview-x64\SoundVolumeView.exe",
        "/SetAppDefault",
        device_name,
        "Render",
        f"{pid}"
    ], check=True)

    return pid

def capture_stream(device_name, link_id, sample_rate, chunk_seconds, shift_seconds):
    """Capture from a given device and transcribe overlapping chunks tagged with link_id."""
    # Step 1: Find device index
    index = None
    for i, dev in enumerate(sd.query_devices()):
        if device_name.lower() in dev["name"].lower() and dev["max_input_channels"] > 0:
            index = i
            break
    if index is None:
        raise RuntimeError(f"Device not found: {device_name}")

    print(f"🎧 Capturing from {device_name} as '{link_id}' (index={index})")

    # Step 2: Setup queue
    chunk_queue = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(f"[{link_id}] Stream warning:", status)
        chunk_queue.put(indata[:, 0].copy())

    # Step 3: Start input stream
    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype='float32',
        blocksize=int(sample_rate * shift_seconds),
        callback=callback,
        device=index
    )
    stream.start()

    # Step 4: Start producer loop (your style)
    def producer():
        buf = np.empty((0,), dtype=np.float32)
        chunk_sz = int(sample_rate * chunk_seconds)
        shift_sz = int(sample_rate * shift_seconds)

        while True:
            data = chunk_queue.get()
            buf = np.concatenate([buf, data])
            while len(buf) >= chunk_sz:
                yield buf[:chunk_sz]
                buf = buf[shift_sz:]

    # Step 5: Background thread to transcribe
    def transcription_worker():
        for chunk in producer():
            transcribe(chunk, link_id)

    threading.Thread(target=transcription_worker, daemon=True).start()

SAMPLE_RATE   = 16000
CHUNK_SECONDS = 8
SHIFT_SECONDS = 4

# === MAIN ===
if __name__ == "__main__":
    for i, page in enumerate(PAGES):
        launch_chrome_with_device(
            url=page["url"],
            device_name=page["device_name"],
            profile_id=i
        )
        time.sleep(1)
        capture_stream(page["device_name"], link_id=page["title"], sample_rate=SAMPLE_RATE, chunk_seconds=CHUNK_SECONDS, shift_seconds=SHIFT_SECONDS)

    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print("🛑 Exiting…")
'''