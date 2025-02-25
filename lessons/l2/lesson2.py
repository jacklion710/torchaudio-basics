import torch
import matplotlib.pyplot as plt
import torchaudio

# Load the audio file
file_path = '../../Kick.wav'  # Update path if needed
waveform, sample_rate = torchaudio.load(file_path)

print(f"Sample rate: {sample_rate} Hz")
print(f"Audio shape: {waveform.shape}")

# Convert to mono if stereo by averaging channels
if waveform.shape[0] > 1:
    waveform = torch.mean(waveform, dim=0, keepdim=True)

# Normalize audio to range [-1, 1]
max_val = torch.max(torch.abs(waveform))
if max_val > 0:
    waveform = waveform / max_val
print(f"Max amplitude after normalization: {torch.max(torch.abs(waveform)).item()}")

# Create time axis in seconds
time = torch.arange(0, waveform.shape[1]) / sample_rate

# Plot the waveform
plt.figure(figsize=(10, 4))
plt.plot(time, waveform[0].numpy())  # Convert to numpy for plotting
plt.title('Normalized Kick Drum Waveform')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()