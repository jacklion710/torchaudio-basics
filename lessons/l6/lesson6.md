# Lesson 6: Transfer Learning in Audio with PyTorch

In this lesson, we focus on leveraging transfer learning to improve the performance of audio classification tasks. Transfer learning allows us to use a pre-trained model and fine-tune it on a specific task, reducing the need for a large dataset and computational resources.

## The Concept of Transfer Learning

Transfer learning involves taking a model trained on a large dataset and adapting it to a similar task. This is particularly useful in audio processing, where training models from scratch can be resource-intensive.

## Using Wav2Vec 2.0 for Audio Classification

Wav2Vec 2.0 is a model pre-trained on vast amounts of unlabeled audio data. It can be fine-tuned for various audio tasks, including audio classification.

### Preparing the Dataset

We start by using our UrbanSound8K dataset class, `AudioClassificationDataset`, which loads audio files, applies transformations, and prepares them for the model.

```python
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
```

This class handles loading audio data, converting it to mono if necessary, and applying any specified transformations.

## Padding Audio
Our friend the pad_collate function from lesson4 becomes useful to us again

```py
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
```

## Loading a Pre-trained Wav2Vec2 Model

```py
# Load a pre-trained Wav2Vec2 model
wav2vec2_bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
temp_model = wav2vec2_bundle.get_model()
```

Here, we load a pre-trained Wav2Vec 2.0 model from torchaudio's pipeline. Wav2Vec 2.0 has been trained on a large corpus of unlabeled audio data and can extract rich, meaningful features from raw audio waveforms. This capability makes it an excellent starting point for audio classification tasks.

Determining the Correct Feature Size
```py
# Determine the correct feature size using a temporary DataLoader
train_dataset_temp = AudioClassificationDataset('data/UrbanSound8K.csv', 'data/audio', transformation=None)
temp_loader = DataLoader(dataset=train_dataset_temp, batch_size=1, shuffle=False)
temp_inputs, _ = next(iter(temp_loader))
temp_features, _ = temp_model(temp_inputs)
feature_size = temp_features.shape[-1]
```

Before defining our classification model, we need to determine the size of the features extracted by Wav2Vec 2.0. We do this by passing a sample batch through the model and observing the output feature size. This size is critical for defining the subsequent linear classifier layer that will output the predictions for our specific audio classification task.

## Define the model
```py
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
```
This class integrates the pre-trained Wav2Vec 2.0 model with a custom linear layer (`self.classifier`) for classification. The linear layer's input features match the size determined previously, ensuring compatibility with the extracted features.

### Data Augmentation and Preprocessing

We apply resampling as a preprocessing step to ensure the audio input matches the sample rate expected by Wav2Vec 2.0.

```py
transform = torchaudio.transforms.Resample(orig_freq=44100, new_freq=16000)
```

### The `mode_info` Variable

```py
model_info = {
    "model_state": model.state_dict(),
    "feature_size": feature_size
}
```

The `model_info` variable is a dictionary that stores not only the trained model's state but also the feature size used by the classifier. This is particularly important for transfer learning because it encapsulates both the learned parameters and the configuration necessary to reproduce the model's architecture. When we save this dictionary:

```py
torch.save(model_info, model_save_path)
```

We're ensuring that anyone who loads the model later has all the information needed to correctly initialize the model architecture and load the learned weights. This approach facilitates model sharing and deployment, as the model's architecture and its state are bundled together.

## Training the Model

Training involves fine-tuning the pre-trained Wav2Vec 2.0 model on our specific audio classification task.

### Setup

Before training, we initialize our model, set the loss function, and choose an optimizer.

```python
model = Wav2Vec2ForAudioClassification(wav2vec2_bundle.get_model(), num_classes, feature_size)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

### The Training Loop

During training, we iterate over our dataset, compute the loss, and update the model's weights accordingly.

```python
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
```

We use `tqdm` to display a progress bar, making it easier to track the training process.

## Saving the Model

After training, we save the model's state along with the feature size to a file. This allows us to easily load the model for future inference.

```python
torch.save(model_info, model_save_path)
```

## Conclusion

Transfer learning with Wav2Vec 2.0 offers a powerful approach for audio classification tasks. By leveraging pre-trained models, we can achieve high performance with relatively small datasets and less computational effort.

In the next lesson, we will explore deploying our trained audio models, making them accessible for real-world applications.
