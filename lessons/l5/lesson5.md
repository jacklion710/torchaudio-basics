# Lesson 5: Advanced Audio Processing Techniques

In this lesson, we delve into advanced techniques for enhancing audio model performance, including data augmentation and feature engineering. These strategies are vital for improving the robustness and accuracy of audio classification models.

## Audio Data Augmentation

Data augmentation is a technique used to increase the diversity of your training set by applying random transformations. It helps prevent overfitting and makes your model more robust to variations in audio data.

### Adding Noise

One simple yet effective augmentation technique is adding random noise to your audio waveform.

```python
def add_noise(waveform, noise_level=0.005):
    noise = torch.randn(waveform.size()) * noise_level
    augmented_waveform = waveform + noise
    return augmented_waveform
```

This function adds Gaussian noise to the waveform, controlled by the `noise_level` parameter.

### Pitch Shifting

Pitch shifting alters the pitch of the audio signal without changing its duration.

```python
def change_pitch(waveform, sample_rate, pitch_shift=5):
    augmented_waveform = torchaudio.functional.pitch_shift(waveform, sample_rate, pitch_shift)
    return augmented_waveform
```

Here, `pitch_shift` specifies the number of semitones by which the pitch is shifted.

### Time Shifting

Time shifting moves the audio signal forward or backward in time.

```python
def time_shift(waveform, shift_max=0.1):
    shift_amount = int(random.random() * shift_max * waveform.size(1))
    return torch.roll(waveform, shifts=shift_amount, dims=1)
```

`shift_max` controls the maximum fraction of the waveform that can be shifted.

## Feature Extraction

Beyond raw waveforms, models often benefit from using spectral features like mel-spectrograms, MFCCs, and their derivatives (delta features).

### Extracting Mel-Spectrogram Features

```python
def extract_features(waveform, sample_rate):
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        n_mels=64,
        hop_length=512 
    )(waveform)
    
    delta = torchaudio.functional.compute_deltas(mel_spectrogram)
    delta_delta = torchaudio.functional.compute_deltas(delta)
    
    features = torch.cat([mel_spectrogram, delta, delta_delta], dim=0)
    return features
```

This function computes a mel-spectrogram and its first and second derivatives, providing a rich representation of the audio signal.

## Demonstration

Let's apply these techniques to an example audio file and visualize the results.

```python
filename = '../../audio/obsc.wav'
waveform, sample_rate = torchaudio.load(filename)

waveform_noise = add_noise(waveform)
waveform_pitch = change_pitch(waveform, sample_rate)
waveform_shifted = time_shift(waveform)

features = extract_features(waveform, sample_rate)

plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.title("Original Waveform")
plt.plot(waveform.t().numpy())

plt.subplot(3, 1, 2)
plt.title("Waveform with Noise")
plt.plot(waveform_noise.t().numpy())

plt.subplot(3, 1, 3)
plt.title("Features")
plt.imshow(features.log2()[0,:,:].numpy(), cmap='viridis', aspect='auto')
plt.show()
```

## Conclusion

In this lesson, we explored advanced techniques for processing audio data, including data augmentation and feature extraction. These strategies are crucial for developing high-performing audio classification models, especially in scenarios with limited training data.

In the next lesson, we will learn how to leverage transfer learning for audio tasks, using pre-trained models to achieve even better performance with less data.
