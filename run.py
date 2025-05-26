import sounddevice as sd
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import time
from threading import Thread
from queue import Queue
import logging

# ─── Configuration ───────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16000
CHUNK_SECONDS = 8
SHIFT_SECONDS = 8   # <=8
MODEL_NAME    = "ivrit-ai/whisper-large-v3"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

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

# ─── Load Model ───────────────────────────────────────────────────────────────
logging.getLogger("transformers").setLevel(logging.ERROR)
print(f"📦 Loading {MODEL_NAME} on {DEVICE}…")
processor = WhisperProcessor.from_pretrained(MODEL_NAME)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)

# ─── Transcription Function ───────────────────────────────────────────────────
def transcribe(audio_np: np.ndarray) -> str:
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    inputs = processor(
        audio_np,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        return_attention_mask=True
    )
    input_features = inputs.input_features.to(DEVICE)
    attention_mask = torch.ones(
        input_features.shape[:2], dtype=torch.long, device=DEVICE
    )
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="he", task="transcribe")
    generated_ids = model.generate(
        input_features,
        attention_mask=attention_mask,
        forced_decoder_ids=forced_decoder_ids
    )
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

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

def consumer():
    for chunk in producer():
        txt = transcribe(chunk)

        sentence = [word[::-1] for word in txt.split()]
        sentence = " ".join(sentence)

        ts  = time.strftime("%H:%M:%S", time.localtime())
        print(f"[{ts}] {sentence}")

# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=2,                   # CABLE Output is stereo
        callback=audio_callback,
        device=device_index
    )
    stream.start()
    print("🔴 Capturing from CABLE Output (Ctrl+C to stop)…")

    t = Thread(target=consumer, daemon=True)
    t.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopped.")
        stream.stop()
        stream.close()