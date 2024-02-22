# Lesson 2: Working with Audio Data in PyTorch

In this lesson, we dive into how to work with audio data using `torchaudio`, an extension for PyTorch designed to handle audio processing with the same flexibility and efficiency as PyTorch does for computer vision and natural language processing.

## Getting Started with torchaudio

`torchaudio` complements PyTorch by providing efficient, GPU-accelerated audio processing functionalities. It supports loading, transforming, and saving audio files. Before we start, make sure `torchaudio` is installed in your environment.

### Loading an Audio File

The first step in audio processing is to load an audio file. `torchaudio` makes this easy with its `load` function.

```python
import torchaudio

filename = '../../audio/obsc.wav'  # Replace with your audio file path
waveform, sample_rate = torchaudio.load(filename)
```

This code loads an audio file, returning the waveform as a tensor and its sample rate. The waveform tensor shape is typically `[channels, frames]`, where `channels` represent the audio channels (mono or stereo), and `frames` represent the discrete audio samples over time.

### Visualizing Audio Data

Visualizing audio can provide insights into its characteristics. Let's plot the waveform of our loaded audio.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style for plots
sns.set(style="whitegrid")

plt.figure(figsize=(10, 4))
plt.plot(waveform.t().numpy())
plt.title('Audio Waveform')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.savefig('../../plots/audio_waveform_seaborn.png')
plt.show()
```

In this snippet:
- We transpose the waveform tensor with `.t()` to convert it to `[frames, channels]` for plotting.
- We use `matplotlib` to plot the waveform and `seaborn` to enhance the visual style.
- The X-axis represents time, and the Y-axis represents amplitude.

In this lesson, you've learned how to load and visualize audio data using `torchaudio` and `matplotlib`. These skills form the foundation for more advanced audio processing tasks, such as feature extraction and audio classification, which we'll explore in upcoming lessons.

In Lesson 3, where we will dive into audio feature extraction techniques that are crucial for building machine learning models.
