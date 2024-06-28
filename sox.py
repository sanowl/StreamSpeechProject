import numpy as np
import wave
import os

os.makedirs('dummy_input', exist_ok=True)

samplerate = 16000
duration = 1  # seconds
frequency = 440  # Hz

t = np.linspace(0., duration, int(samplerate * duration))
y = np.sin(2. * np.pi * frequency * t)

with wave.open('dummy_input/dummy.wav', 'w') as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(samplerate)
    f.writeframes((y * 32767).astype(np.int16).tobytes())
