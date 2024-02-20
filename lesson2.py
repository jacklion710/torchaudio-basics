import torchaudio
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style for plots
sns.set(style="whitegrid")

# Display available audio backend
# print(torchaudio.list_audio_backends())

# Assuming you've loaded an audio file
filename = 'audio/obsc.wav'  # Audio file path
waveform, sample_rate = torchaudio.load(filename)

# Plotting with Matplotlib functions using Seaborn's styling
plt.figure(figsize=(10, 4))
plt.plot(waveform.t().numpy())
plt.title('Audio Waveform')
plt.xlabel('Time')
plt.ylabel('Amplitude')
# Save the plot using Matplotlib's savefig, as seaborn enhances the style but does not replace this functionality
plt.savefig('plots/audio_waveform_seaborn.png')
plt.show()
