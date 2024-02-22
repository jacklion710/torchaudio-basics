# Lesson 4: Building a Simple Audio Classification Model

In this lesson, we're focusing on constructing a basic CNN model to classify different types of urban sounds using the UrbanSound8K dataset. This dataset consists of various urban sounds from 10 classes, making it an excellent choice for our classification project.

## Preparing the Dataset

First, we define a custom dataset class, `UrbanSoundDataset`, to handle loading audio files, transforming them into mel-spectrograms, and padding them to ensure consistent sizes.

## The UrbanSoundDataset Class
```python
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import pandas as pd
import os

class UrbanSoundDataset(Dataset):
    def __init__(self, csv_file, root_dir, fold, transform=None):
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
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if self.transform:
            waveform = self.transform(waveform)
        label = self.fold_data.iloc[idx]['classID']
        return waveform, label
```

* `__init__`: Initializes the dataset object, loading the dataset annotations from a CSV file and setting up the path to the audio files.
* `__len__`: Returns the total number of samples in the dataset.
* `__getitem__`: Loads and returns a single sample from the dataset, including the audio waveform and its corresponding label. It also applies any transformations specified (e.g., converting to a mel-spectrogram).

## Obtain the data

The dataset used in this project is UrbanSound8K, which can be found [here](https://urbansounddataset.weebly.com/urbansound8k.html). It contains 8732 labeled sound excerpts of urban sounds from 10 classes, pre-sorted into ten folds for easier cross-validation and experimentation. You must download the dataset and insert fold0-fold9 directories into the data/audio/ directory. This lessons code revolves around this dataset howevr with some refactoring you can apply these concepts to any dataset!

## Data Paths

We must specify the paths to the metadata and the data itself

```py
csv_file = '../../data/UrbanSound8K.csv'
root_dir = '../../data/audio/'
```

## Data Transformation

Transforming the raw audio data into a more manageable form is crucial for processing. We use the mel-spectrogram representation, which captures the energy of sound in different frequency bands over time.

```py
transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=64, n_fft=1024, hop_length=512)
```

This line of code creates a transformation pipeline that converts audio waveforms into mel-spectrograms with specified parameters.

## Building the CNN Model

Our model, `AudioCNN`, consists of two convolutional layers for feature extraction followed by max pooling layers, an adaptive average pooling layer to standardize the output size, and a fully connected layer for classification.

```python
class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 5))
        self.fc = nn.Linear(32 * 5 * 5, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
```

* Convolutional Layers: conv1 and conv2 are used to extract audio features.
* Pooling Layers: pool reduces the dimensionality of the data to make the processing more efficient.
* Adaptive Pooling: adaptive_pool ensures the output size is consistent, which is crucial for connecting to fully connected layers.
* Fully Connected Layer: fc makes the final classification based on the extracted features.

## Training and Evaluation

### Training Function

Training involves passing our data through the model, calculating the loss (difference between the predicted and actual labels), and updating the model to reduce this loss.

```python
def train(model, train_loader, optimizer, loss_function, device):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)
```
* model.train(): Prepares the model for training.
* optimizer.zero_grad(): Clears old gradients; otherwise, they'll accumulate.
* loss.backward(): Computes the gradient of the loss.
* optimizer.step(): Updates the model parameters.

### Validation Function

Validation helps us assess the model's performance on unseen data, ensuring it's learning general patterns rather than memorizing the training data.

```py
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
```

* model.eval(): Prepares the model for evaluation, affecting certain layers like dropout.
* torch.no_grad(): Indicates that we don't need to compute gradients, which reduces memory consumption and speeds up computation.

## Understanding the pad_collate Function

When training neural networks with audio data, one challenge we often face is the variable length of audio files. Since most neural network architectures expect inputs of consistent size, we need to standardize the size of our audio inputs. This is where the pad_collate function comes into play.

## Purpose of pad_collate

The pad_collate function is designed to ensure that all audio waveforms in a batch have the same length. It does this by padding shorter audio files with zeros until they match the length of the longest file in the batch. This uniformity is crucial for batching and processing through the neural network.

## How pad_collate Works

Let's break down the pad_collate function step by step:

```py
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
```

3. **Find the Maximum Length:** First, we determine the length of the longest audio waveform in the current batch.
2. **Padding the Waveforms:** For each waveform in the batch, we calculate how much padding is needed to match the maximum length. We then apply this padding to the end of the waveform.
3. **Stacking for the Batch:** Once all waveforms are padded to the same length, we stack them together along with their labels. This results in a tensor of waveforms that can be processed as a batch by the neural network.

## The Importance of pad_collate

Using the pad_collate function allows us to efficiently train neural networks on audio data with varying lengths. By ensuring consistent input sizes, we can leverage the power of batching, which significantly speeds up the training process and makes it more manageable.

## Cross-validation Setup

To ensure our model's robustness, we perform 10-fold cross-validation and average the accuracy across all folds.

```python
num_epochs = 100
learning_rate = 0.001
num_folds = 10
results = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
```

* We iterate through each fold, training the model on the training set and evaluating it on the validation set.
* After each fold, we record the model's performance.

Building a simple CNN model for audio classification with PyTorch demonstrates the power of neural networks in processing and understanding complex data like audio signals. This foundational knowledge sets the stage for exploring more advanced models and techniques in the field of audio analysis.

In the next lesson, we will delve into advanced audio data augmentation techniques to further improve our model's performance.
