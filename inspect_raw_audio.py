import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

filepath = os.path.join("raw_audio", os.listdir("raw_audio")[0])
samples, sample_rate = sf.read(filepath)

print(f"File: {filepath}")
print(f"Sample Rate: {sample_rate}")
print(f"Shape: {samples.shape}")
print(f"Type: {samples.dtype}")

if samples.ndim == 2:
    print("Stereo detected → averaging to mono")
    samples = np.mean(samples, axis=1)

peak = np.max(np.abs(samples))
print(f"Peak Amplitude: {peak}")
print(f"Min: {np.min(samples)}  Max: {np.max(samples)}")

# Plot raw waveform
plt.plot(samples[:5000])
plt.title("Raw Audio: First 5000 Samples")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
