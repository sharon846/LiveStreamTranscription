import sounddevice as sd
import numpy as np

SAMPLE_RATE = 16000
DURATION = 5
VIRTUAL_DEVICE_NAME = "Voicemeeter Out A3"

def get_device_index(name):
    for i, d in enumerate(sd.query_devices()):
        if name.lower() in d["name"].lower() and d["max_input_channels"] > 0:
            return i
    raise RuntimeError(f"Device '{name}' not found.")

def main():
    idx = get_device_index(VIRTUAL_DEVICE_NAME)
    print(f"\n🎙 Listening to: {sd.query_devices()[idx]['name']}")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=2, device=idx, dtype='float32')
    sd.wait()
    print(f"📊 Max Amplitude: {np.max(np.abs(audio)):.5f}")

if __name__ == '__main__':
    main()