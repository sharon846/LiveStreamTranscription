import subprocess
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd
from playwright.sync_api import sync_playwright

# === CONFIGURATION ===
PAGE_URLS = [
    "https://www.kan.org.il/live/",
    "https://www.mako.co.il/mako-vod-live-tv/VOD-6540b8dcb64fd31006.htm",
    # …add more pages here
]
SAMPLE_RATE       = 44100
FRAMES_PER_BUFFER = 1024
CHANNELS          = len(PAGE_URLS)

buffers = [deque() for _ in range(CHANNELS)]

def find_audio_src(page_url, timeout=10000):
    """
    Launch a headless browser, navigate to page_url, listen for any network
    request whose URL ends with .m3u8/.mp3/.aac, return the first one.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        audio_url = None

        def on_request(request):
            nonlocal audio_url
            url = request.url
            if any(url.lower().endswith(ext) for ext in (".m3u8", ".mp3", ".aac")):
                audio_url = url
                # once found, stop listening
                page.off("request", on_request)

        page.on("request", on_request)
        page.goto(page_url, wait_until="networkidle")
        # give it a bit of extra time for the player to spin up
        page.wait_for_timeout(timeout)
        browser.close()

        if not audio_url:
            raise RuntimeError(f"No HLS/mp3/aac URL found on {page_url}")
        return audio_url

def reader_thread(idx, page_url):
    try:
        direct_url = find_audio_src(page_url)
        print(f"[Ch{idx}] → {direct_url}")
    except Exception as e:
        print(f"[Ch{idx}] Error extracting audio URL: {e}")
        return

    cmd = [
        "ffmpeg",
        "-i", direct_url,
        "-f", "f32le",
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",
        "-loglevel", "quiet",
        "pipe:1"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    while True:
        raw = proc.stdout.read(FRAMES_PER_BUFFER * 4)
        if not raw:
            break
        pcm = np.frombuffer(raw, dtype=np.float32)
        buffers[idx].append(pcm)
    proc.stdout.close()
    print(f"[Ch{idx}] Stream ended.")

# launch threads
threads = []
for i, url in enumerate(PAGE_URLS):
    t = threading.Thread(target=reader_thread, args=(i, url), daemon=True)
    t.start()
    threads.append(t)

def audio_callback(outdata, frames, time, status):
    block = np.zeros((frames, CHANNELS), dtype=np.float32)
    for ch in range(CHANNELS):
        if buffers[ch]:
            data = buffers[ch].popleft()
            if len(data) < frames:
                data = np.pad(data, (0, frames - len(data)), 'constant')
            block[:, ch] = data[:frames]
            if len(data) > frames:
                buffers[ch].appendleft(data[frames:])
    outdata[:] = block

with sd.OutputStream(
    samplerate=SAMPLE_RATE,
    blocksize=FRAMES_PER_BUFFER,
    dtype="float32",
    channels=CHANNELS,
    callback=audio_callback
):
    print(f"Playing {CHANNELS} streams → {CHANNELS} channels. Ctrl-C to quit.")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print("Stopped.")
