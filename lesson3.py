import torchaudio
import matplotlib.pyplot as plt

filename = 'audio/obsc.wav'  # Audio file path
waveform, sample_rate = torchaudio.load(filename)

# Compute the spectrogram
spectrogram = torchaudio.transforms.Spectrogram()(waveform)

# Plot the spectrogram
plt.figure(figsize=(10, 4))
plt.imshow(spectrogram.log2()[0,:,:].numpy(), cmap='viridis', aspect='auto')
plt.title('Spectrogram')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.colorbar(format='%+2.0f dB')
plt.show()

# Compute the MFCCs
mfcc = torchaudio.transforms.MFCC(sample_rate=sample_rate)(waveform)

# Plot the MFCCs
plt.figure(figsize=(10,4))
plt.imshow(mfcc[0].detach().numpy(), cmap='viridis', aspect='auto')
plt.title('MFCC')
plt.xlabel('Time')
plt.ylabel('MFCC Coefficients')
plt.colorbar()
plt.show()

# Compute mel-spectrogram
mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate)(waveform)

# Plot the mel-spectrogram
plt.figure(figsize=(10, 4))
plt.imshow(mel_spectrogram.log2()[0,:,:].numpy(), cmap='viridis', aspect='auto')
plt.title('Mel-Spectrogram')
plt.xlabel('Time')
plt.ylabel('Mel Frequency')
plt.colorbar(format='%+2.0f dB')
plt.show()