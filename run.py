import numpy as np
import sounddevice as sd
import threading
import queue
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# === CONFIG ===
SAMPLE_RATE = 44100
CHUNK_SECONDS = 4
SHIFT_SECONDS = 2
CHUNK_SIZE = SAMPLE_RATE * CHUNK_SECONDS
SHIFT_SIZE = SAMPLE_RATE * SHIFT_SECONDS

# Live pages you want to listen to
PAGES = [
    {"url": "https://www.kan.org.il/live/", "device_name": "CABLE-C Output"},
    {"url": "https://www.mako.co.il/mako-vod-live-tv/VOD-6540b8dcb64fd31006.htm", "device_name": "CABLE-D Output"},
]

chunk_queue = queue.Queue()

# Your transcribe function
def transcribe(chunk):
    print("🔊 Transcribing chunk of shape:", chunk.shape)

# Capture raw audio stream from a specific device
def capture_stream(device_name):
    index = None
    for i, dev in enumerate(sd.query_devices()):
        if device_name.lower() in dev["name"].lower() and dev["max_input_channels"] > 0:
            index = i
            break

    if index is None:
        raise RuntimeError(f"Audio input device '{device_name}' not found!")

    print(f"🎧 Capturing from device: {device_name} (index={index})")

    def callback(indata, frames, time, status):
        if status:
            print("⚠️  Audio warning:", status)
        chunk_queue.put(indata[:, 0].copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=SHIFT_SIZE,
        callback=callback,
        device=index,
    )
    stream.start()
    return stream

# Launch Chrome tab with the AudioPick extension enabled
def launch_chrome_tab(url):
    chrome_options = Options()
    chrome_options.add_argument("--use-fake-ui-for-media-stream")
    chrome_options.add_argument("--autoplay-policy=no-user-gesture-required")
    chrome_options.add_argument("--disable-notifications")
    chrome_options.add_argument("--new-window")
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)
    print(f"🌐 Launched tab: {url}")
    return driver  # keep this alive

# Chunk generator for transcribe()
def chunk_generator():
    buf = np.empty((0,), dtype=np.float32)
    while True:
        data = chunk_queue.get()
        buf = np.concatenate([buf, data])
        while len(buf) >= CHUNK_SIZE:
            yield buf[:CHUNK_SIZE]
            buf = buf[SHIFT_SIZE:]

# Worker to consume chunks
def transcribe_worker():
    for chunk in chunk_generator():
        transcribe(chunk)

# === MAIN ===
if __name__ == "__main__":
    # Start audio capture per virtual device
    streams = []
    for page in PAGES:
        streams.append(capture_stream(page["device_name"]))
        launch_chrome_tab(page["url"])
        time.sleep(2)  # Let page load & start audio

    # Start transcription worker
    threading.Thread(target=transcribe_worker, daemon=True).start()

    # Keep alive
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print("🛑 Exiting…")
