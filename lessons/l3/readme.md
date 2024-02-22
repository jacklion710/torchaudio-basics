# Lesson 3: Audio Feature Extraction with PyTorch

In our journey through audio processing with PyTorch, we now arrive at a crucial milestone: feature extraction. This lesson covers how to extract meaningful features from audio signals, which is essential for audio analysis and machine learning tasks.

## Understanding Audio Features

Feature extraction transforms raw audio data into a structured format that's more informative and less redundant for machine learning models. Common audio features include Spectrograms, Mel-Frequency Cepstral Coefficients (MFCCs), and Mel-Spectrograms.

### Spectrograms

A spectrogram is a visual representation of the spectrum of frequencies in a sound or other signal as they vary with time. It's a powerful tool for analyzing the frequency content of audio signals.

#### Computing and Plotting a Spectrogram

```python
import torchaudio
import matplotlib.pyplot as plt

filename = 'audio/obsc.wav'  # Replace with your audio file path
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
```

### Mel-Frequency Cepstral Coefficients (MFCCs)

MFCCs are coefficients that collectively make up an MFC. They are derived from a type of cepstral representation of the audio clip (a nonlinear "spectrum-of-a-spectrum").

#### Computing and Plotting MFCCs

```python
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
```

### Mel-Spectrogram

A Mel-Spectrogram is a Spectrogram where the frequencies are converted to the Mel scale, more closely approximating human auditory system's response.

#### Computing and Plotting a Mel-Spectrogram

```python
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
```

Feature extraction is a pivotal process in audio analysis, providing a bridge between raw audio data and machine learning models. By understanding and utilizing Spectrograms, MFCCs, and Mel-Spectrograms, we can effectively capture the essence of audio signals for further analysis or model training.

In the next lesson, we will take these concepts further and start building an audio classification model using these features.
