# lesson5.py
import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import random

def add_noise(waveform, noise_level=0.005):
    noise = torch.randn(waveform.size()) * noise_level
    augmented_waveform = waveform + noise
    return augmented_waveform

def change_pitch(waveform, sample_rate, pitch_shift=5):
    augmented_waveform = torchaudio.functional.pitch_shift(waveform, sample_rate, pitch_shift)
    return augmented_waveform

def time_shift(waveform, shift_max=0.1):
    shift_amount = int(random.random() * shift_max * waveform.size(1))
    return torch.roll(waveform, shifts=shift_amount, dims=1)

def extract_features(waveform, sample_rate):
    # Adjust n_fft and/or n_mels based on the warning suggestion
    n_fft = 1024  # This is an example; adjust based on your specific needs
    n_mels = 64  # Adjusted based on the warning; find a balance that suits your data
    
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        hop_length=512 
    )(waveform)
    
    # Compute delta and delta-delta
    delta = torchaudio.functional.compute_deltas(mel_spectrogram)
    delta_delta = torchaudio.functional.compute_deltas(delta)
    
    features = torch.cat([mel_spectrogram, delta, delta_delta], dim=0)
    return features

if __name__ == "__main__":
    filename = 'audio/obsc.wav'
    waveform, sample_rate = torchaudio.load(filename)
    
    # Apply augmentations
    waveform_noise = add_noise(waveform)
    waveform_pitch = change_pitch(waveform, sample_rate)
    waveform_shifted = time_shift(waveform)
    
    # Extract features
    features = extract_features(waveform, sample_rate)
    
    # Plotting
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