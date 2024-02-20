import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os

# Assuming you have a CSV with columns 'filename' and 'label'
class AudioClassificationDataset(Dataset):
    def __init__(self, annotations_file, audio_dir, transformation=None):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir  # This should be the base directory containing all the fold directories
        self.transformation = transformation

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_filename = self.annotations.iloc[index]['slice_file_name']
        fold = self.annotations.iloc[index]['fold']  # Get the fold number
        label = self.annotations.iloc[index]['classID']
        
        # Construct the path with the fold information
        audio_path = os.path.join(self.audio_dir, f'fold{fold}', audio_filename)
        
        waveform, sample_rate = torchaudio.load(audio_path)
        
        if self.transformation:
            waveform = self.transformation(waveform)
        
        return waveform, label
    
def pad_collate(batch):
    max_length = max([waveform.size(1) for waveform, _ in batch])
    batch_padded = []
    for waveform, label in batch:
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        pad_amount = max_length - waveform.size(1)
        padded_waveform = torch.nn.functional.pad(waveform, (0, pad_amount), 'constant', 0)
        batch_padded.append((padded_waveform, label))
    
    waveforms_padded = torch.stack([x[0] for x in batch_padded])
    waveforms_padded = torch.squeeze(waveforms_padded, 1)  # Remove the channel dimension
    labels = torch.tensor([x[1] for x in batch_padded])
    return waveforms_padded, labels

# Load a pre-trained Wav2Vec2 model
wav2vec2_bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = wav2vec2_bundle.get_model()

# Define the new model for audio classification that includes Wav2Vec2 and a classifier on top
class Wav2Vec2ForAudioClassification(torch.nn.Module):
    def __init__(self, pretrained_wav2vec2, num_classes):
        super(Wav2Vec2ForAudioClassification, self).__init__()
        self.wav2vec2 = pretrained_wav2vec2
        # Assuming the feature size based on the base model. Adjust if using a different model
        self.classifier = torch.nn.Linear(768, num_classes)  # 768 for base models

    def forward(self, audio_input):
        # Note: Adjust .forward() usage based on how your Wav2Vec2 model outputs features
        features = self.wav2vec2(audio_input).get('last_hidden_state')
        features = torch.mean(features, dim=1)
        output = self.classifier(features)
        return output

# Specify the number of classes you are classifying into
num_classes = 10
model = Wav2Vec2ForAudioClassification(model, num_classes)

# Transformation: Ensure the audio is resampled to 16kHz, which Wav2Vec2 expects
transform = torchaudio.transforms.Resample(orig_freq=44100, new_freq=16000)

# Create the dataset and dataloader
train_dataset = AudioClassificationDataset('data/UrbanSound8K.csv', 'data/audio', transformation=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True, collate_fn=pad_collate)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.train()

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}")
