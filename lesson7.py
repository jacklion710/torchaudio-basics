from flask import Flask, request, jsonify, render_template
import torch
import torchaudio
from model_defs import Wav2Vec2ForAudioClassification

app = Flask(__name__)

TARGET_SAMPLE_RATE = 16000
NUM_CLASSES = 10  # Adjust based on your model
FEATURE_SIZE = 768  # Adjust based on your model's feature size

# Load the trained model and feature_size
model_path = "models/pretrained_audio_classification_model.pth"
model_info = torch.load(model_path, map_location=torch.device('cpu'))
model_state_dict = model_info["model_state"]
feature_size = model_info["feature_size"]  # Dynamically loaded feature size

# Initialize the model with the loaded feature_size
model = Wav2Vec2ForAudioClassification(pretrained_wav2vec2=torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model(), num_classes=NUM_CLASSES, feature_size=feature_size)

# Load the trained model
model.load_state_dict(model_state_dict)
model.eval()  # Set the model to evaluation mode for inferencing

@app.route('/', methods=['GET'])
def index():
    # Render the upload form
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction="No file part")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', prediction="No selected file")

    waveform, sample_rate = torchaudio.load(file)
    if sample_rate != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE)
        waveform = resampler(waveform)
    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    waveform = waveform.squeeze(0)

    with torch.no_grad():
        waveform = waveform.unsqueeze(0)  # Add a batch dimension
        prediction = model(waveform)
        _, predicted_label = torch.max(prediction, dim=1)
    
        return jsonify({'prediction': str(predicted_label)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
