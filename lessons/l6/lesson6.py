# lesson6.py
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
from tqdm import tqdm

class AudioClassificationDataset(Dataset):
    def __init__(self, annotations_file, audio_dir, transformation=None):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.transformation = transformation

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_filename = self.annotations.iloc[index]['slice_file_name']
        fold = self.annotations.iloc[index]['fold']
        label = self.annotations.iloc[index]['classID']
        audio_path = os.path.join(self.audio_dir, f'fold{fold}', audio_filename)
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono by averaging channels if not already mono
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        if self.transformation:
            waveform = self.transformation(waveform)

        # Ensure waveform is squeezed if it's mono to remove channel dimension
        waveform = torch.squeeze(waveform)
        
        return waveform, label
    
def pad_collate(batch):
    # print("Batch size:", len(batch))
    
    # Adjusting for 1D waveforms
    max_length = max([waveform.shape[0] for waveform, _ in batch])
    # print(f"Max length in batch: {max_length}")
    
    batch_padded = []
    for i, (waveform, label) in enumerate(batch):
        # print(f"Waveform {i} initial shape: {waveform.shape}")
        pad_amount = max_length - waveform.shape[0]
        padded_waveform = torch.nn.functional.pad(waveform.unsqueeze(0), (0, pad_amount), 'constant', 0)  # Add a channel dimension before padding
        batch_padded.append((padded_waveform, label))
        # print(f"Waveform {i} padded shape: {padded_waveform.shape}")
    
    # Removing unnecessary print statement for clarity
    waveforms_padded = torch.stack([x[0] for x in batch_padded]).squeeze(1)  # Remove the temporary channel dimension after stacking
    labels = torch.tensor([x[1] for x in batch_padded])
    return waveforms_padded, labels

# Load a pre-trained Wav2Vec2 model
wav2vec2_bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
temp_model = wav2vec2_bundle.get_model()

# Determine the correct feature size using a temporary DataLoader
train_dataset_temp = AudioClassificationDataset('../../data/UrbanSound8K.csv', '../../data/audio', transformation=None)
temp_loader = DataLoader(dataset=train_dataset_temp, batch_size=1, shuffle=False)
temp_inputs, _ = next(iter(temp_loader))
temp_features, _ = temp_model(temp_inputs)
feature_size = temp_features.shape[-1]

# Define the model class with the determined feature size
class Wav2Vec2ForAudioClassification(torch.nn.Module):
    def __init__(self, pretrained_wav2vec2, num_classes, feature_size):
        super(Wav2Vec2ForAudioClassification, self).__init__()
        self.wav2vec2 = pretrained_wav2vec2
        self.classifier = torch.nn.Linear(feature_size, num_classes)

    def forward(self, audio_input):
        features, _ = self.wav2vec2(audio_input)
        features = torch.mean(features, dim=1)
        output = self.classifier(features)
        return output

num_classes = 10
model = Wav2Vec2ForAudioClassification(wav2vec2_bundle.get_model(), num_classes, feature_size)

# Continue with your script setup for DataLoader, device, training loop, etc.
transform = torchaudio.transforms.Resample(orig_freq=44100, new_freq=16000)
train_dataset = AudioClassificationDataset('../../data/UrbanSound8K.csv', '../../data/audio', transformation=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True, collate_fn=pad_collate)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.train()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
for epoch in range(num_epochs):
    total_loss = 0
    # Wrap train_loader with tqdm for a progress bar
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}")


model_save_path = "../../models/pretrained_audio_classification_model.pth"
model_info = {
    "model_state": model.state_dict(),
    "feature_size": feature_size
}
torch.save(model_info, model_save_path)
print(f"Model saved to {model_save_path}")