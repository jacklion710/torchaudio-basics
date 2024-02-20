import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pandas as pd

# Define the UrbanSoundDataset class
class UrbanSoundDataset(Dataset):
    def __init__(self, csv_file, root_dir, fold, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
            root_dir (string): Directory with all the audio files.
            fold (int): The specific fold to use.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.fold = fold
        self.transform = transform
        self.fold_data = self.annotations[self.annotations['fold'] == fold]

    def __len__(self):
        return len(self.fold_data)

    def __getitem__(self, idx):
        audio_filename = self.fold_data.iloc[idx]['slice_file_name']
        audio_path = os.path.join(self.root_dir, f'fold{self.fold}', audio_filename)
        waveform, sample_rate = torchaudio.load(audio_path)  # This line loads the waveform
        
        # Convert waveform to mono if it's not already
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        if self.transform:
            waveform = self.transform(waveform)
        
        # Pad the waveform tensor to ensure consistent size
        target_width = 751  # Example target width, adjust based on your data
        current_width = waveform.size(-1)
        padding_width = target_width - current_width
        if padding_width > 0:
            waveform = F.pad(waveform, (0, padding_width), "constant", 0)
        
        label = self.fold_data.iloc[idx]['classID']
        return waveform, label


# Define the CNN model
class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 5))
        # Initialize the fully connected layer without specifying the number of input features
        self.fc = nn.Linear(32 * 5 * 5, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adaptive_pool(x)  # Apply adaptive pooling to standardize output size
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = self.fc(x)
        return x
    
# Function to train the model for one epoch
def train(model, train_loader, optimizer, loss_function, device):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)  # Assuming inputs is already [batch_size, channels, height, width]
        # print(f"Output shape: {outputs.shape}, Labels shape: {labels.shape}")  
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Function to evaluate the model
def validate(model, validate_loader, device):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in validate_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs) 
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Ensures all audio waveforms in a batch have the same length by padding shorter ones
def pad_collate(batch):
    # Find the maximum length of a waveform in the batch
    max_len = max([x[0].size(-1) for x in batch])
    
    # Pad each waveform in the batch to the maximum length
    batch_padded = []
    for waveform, label in batch:
        # Calculate the amount of padding needed for this waveform
        pad_amount = max_len - waveform.size(-1)
        # Pad the waveform at the end along the time dimension
        padded_waveform = F.pad(waveform, (0, pad_amount), "constant", 0)
        batch_padded.append((padded_waveform, label))
    
    # Stack all the waveforms and labels together
    waveforms_padded = torch.stack([x[0] for x in batch_padded])
    labels = torch.tensor([x[1] for x in batch_padded])
    return waveforms_padded, labels

# Prepare dataset
csv_file = 'data/UrbanSound8K.csv'
root_dir = 'data/audio/'
transform = T.MelSpectrogram(sample_rate=16000, n_mels=64, n_fft=1024, hop_length=512)
annotations = pd.read_csv(csv_file)

# Cross-validation setup
num_epochs = 100
learning_rate = 0.001
num_folds = 10
results = []

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Perform 10-fold cross-validation
for fold in range(1, num_folds + 1):
    print(f"Processing fold {fold}")
    
    # Splitting dataset
    train_data = UrbanSoundDataset(csv_file=csv_file, root_dir=root_dir, fold=fold, transform=transform)
    validate_data = UrbanSoundDataset(csv_file=csv_file, root_dir=root_dir, fold=fold, transform=transform)
    
    # DataLoader setup
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=pad_collate)
    validate_loader = DataLoader(validate_data, batch_size=32, shuffle=False, collate_fn=pad_collate)
    
    # Model setup
    model = AudioCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()

    inputs, labels = next(iter(train_loader))
    # print(f"Batch input shape from DataLoader: {inputs.shape}")
    # print(f"Batch label shape from DataLoader: {labels.shape}")
    
    # Training loop
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, loss_function, device)
        print(f'Fold {fold}, Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}')
    
    # Validation
    accuracy = validate(model, validate_loader, device)
    results.append(accuracy)
    print(f'Validation Accuracy for fold {fold}: {accuracy:.2f}%')

# Calculate average accuracy across all folds
average_accuracy = np.mean(results)
print(f'Average Accuracy across all folds: {average_accuracy:.2f}%')