import torch
from torch.utils.data import Dataset, DataLoader
import os
import torchaudio

class AudioDataset(Dataset):
    def __init__(self, directory, transform=None):
        """
        Args:
            directory (string): Directory with all the audio files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.directory = directory
        self.transform = transform
        self.files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.wav')]
        # Assuming labels are determined by filename or directory structure
        self.labels = [int(file.split('_')[0]) for file in os.listdir(directory) if file.endswith('.wav')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_path = self.files[idx]
        waveform, sample_rate = torchaudio.load(audio_path)
        label = self.labels[idx]

        if self.transform:
            waveform = self.transform(waveform)

        sample = {'waveform': waveform, 'label': label}
        return sample
