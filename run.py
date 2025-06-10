import sounddevice as sd
import numpy as np
import time
from threading import Thread
from queue import Queue
from faster_whisper import WhisperModel
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import queue

class HebrewTranscriptionGUI:
    def __init__(self, title="Hebrew Transcription"):
        self.root = tk.Tk()
        self.root.title(title)

        self.text_area = ScrolledText(self.root, wrap=tk.WORD, font=("Arial", 16), width=70, height=25, bg="white", fg="black")
        self.text_area.pack(expand=True, fill="both")

        # Right-align text with tag
        self.text_area.tag_configure('rtl', justify='right')

        self.queue = queue.Queue()
        self.root.after(100, self.update_gui)

    def update_gui(self):
        while not self.queue.empty():
            sentence = self.queue.get()
            self.text_area.insert(tk.END, sentence + "\n", 'rtl')
            self.text_area.yview(tk.END)
        self.root.after(100, self.update_gui)

    def add_sentence(self, sentence):
        self.queue.put(sentence)

    def start(self):
        self.root.mainloop()

# ─── Configuration ───────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16000
CHUNK_SECONDS = 8
SHIFT_SECONDS = 4
MODEL_NAME    = "ivrit-ai/whisper-large-v3-turbo-ct2"
DEVICE        = "cuda" if sd.query_hostapis()[sd.default.hostapi]['name'].lower() == "asio" else "cpu"
COMPUTE_TYPE  = "int8" if DEVICE == "cpu" else "float16"

# ─── LIST DEVICES ─────────────────────────────────────────────────────────────
print("🔎 Available audio devices:")
devices = sd.query_devices()
for i, d in enumerate(devices):
    print(f"{i}: {d['name']} ({d['hostapi']})")

# Find the index of "CABLE Output"
target_device_name = "CABLE Output"
device_index = next(
    (i for i, d in enumerate(devices) if target_device_name in d["name"]),
    None
)

if device_index is None:
    raise RuntimeError(f"Device '{target_device_name}' not found. Make sure VB-Cable is installed.")

print(f"\n✅ Using device #{device_index}: {devices[device_index]['name']}\n")

# ─── Load Faster-Whisper Model ───────────────────────────────────────────────
print(f"📦 Loading {MODEL_NAME} on {DEVICE} (compute_type={COMPUTE_TYPE})…")
model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)

# ─── Transcription Function ───────────────────────────────────────────────────
def transcribe(audio_np: np.ndarray) -> str:
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    segments, _ = model.transcribe(audio_np, language="he")
    return " ".join(segment.text for segment in segments).strip()

# ─── Streaming Audio Setup ────────────────────────────────────────────────────
chunk_queue = Queue()

def audio_callback(indata, frames, time_info, status):
    mono = indata.mean(axis=1).astype(np.float32)
    chunk_queue.put(mono)

def producer():
    buf      = np.empty((0,), dtype=np.float32)
    chunk_sz = SAMPLE_RATE * CHUNK_SECONDS
    shift_sz = SAMPLE_RATE * SHIFT_SECONDS

    while True:
        data = chunk_queue.get()
        buf  = np.concatenate([buf, data])
        while len(buf) >= chunk_sz:
            yield buf[:chunk_sz]
            buf = buf[shift_sz:]

def clear_prv_text(txt):
    max_len = min(len(prv_text), len(txt))
    for i in range(max_len, 0, -1):
        if prv_text[-i:] == txt[:i]:
            return txt[i:]
    return txt

prv_text = ""
def consumer():
    global prv_text
    for chunk in producer():
        txt = transcribe(chunk)
        #txt = clear_prv_text(txt)

        # Optional: reverse each Hebrew word (visual fix)
        sentence = [word[::-1] for word in txt.split()]
        sentence = " ".join(sentence)

        ts  = time.strftime("%H:%M:%S", time.localtime())
        print(f"[{ts}] {sentence}")
        gui.add_sentence(f"[{ts}] {txt}")
        prv_text = txt

# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    gui = HebrewTranscriptionGUI()  # create GUI

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=2,
        callback=audio_callback,
        device=device_index
    )
    stream.start()
    print("🔴 Capturing from CABLE Output (Ctrl+C to stop)…")

    t = Thread(target=consumer, daemon=True)
    t.start()

    gui.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopped.")
        stream.stop()
        stream.close()
