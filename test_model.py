import torch
import torchaudio
from model_defs import Wav2Vec2ForAudioClassification

CLASS_NAMES = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music"
]

TARGET_SAMPLE_RATE= 16000

# Load the model (ensure this matches how you've trained and saved it)
model_path = "models/pretrained_audio_classification_model.pth"
model_info = torch.load(model_path, map_location=torch.device('cpu'))
model_state_dict = model_info["model_state"]
feature_size = model_info["feature_size"]
num_classes = 10  # Adjust this based on your specific model

model = Wav2Vec2ForAudioClassification(pretrained_wav2vec2=torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model(), num_classes=num_classes, feature_size=feature_size)
model.load_state_dict(model_state_dict)
model.eval()

# Load an audio file
audio_path = 'data/audio/fold1/7383-3-0-0.wav'
waveform, sample_rate = torchaudio.load(audio_path)

# Convert stereo to mono if necessary
if waveform.size(0) > 1:
    waveform = waveform.mean(dim=0).unsqueeze(0)  # Average channels to convert to mono

# Ensure the sample rate matches the model's expected input
if sample_rate != TARGET_SAMPLE_RATE:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE)
    waveform = resampler(waveform)

# Predict
with torch.no_grad():
    prediction = model(waveform)  # waveform is now [1, sequence_length]
    _, predicted_label = torch.max(prediction, dim=1)
    predicted_class = CLASS_NAMES[predicted_label.item()]

print(f"Predicted class: {predicted_class}")
