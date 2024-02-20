from flask import Flask, request, jsonify, render_template
import torch
import torchaudio
from lesson6 import Wav2Vec2ForAudioClassification 

app = Flask(__name__)

TARGET_SAMPLE_RATE = 16000
NUM_CLASSES = 10  # Adjust based on your model
FEATURE_SIZE = 768  # Adjust based on your model's feature size

# Load the trained model
model_path = "models/pretrained_audio_classification_model.pth"
model = Wav2Vec2ForAudioClassification(NUM_CLASSES, FEATURE_SIZE)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

@app.route('/', methods=['GET'])
def index():
    # Render the upload form
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Handle file upload and prediction as before
    if 'file' not in request.files:
        return render_template('index.html', prediction="No file part")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', prediction="No selected file")

    # Load and preprocess the audio file
    waveform, sample_rate = torchaudio.load(file)
    # Resample if necessary
    if sample_rate != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE)
        waveform = resampler(waveform)
    # Convert to mono by averaging the channels if not already mono
    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Make a prediction
    with torch.no_grad():
        waveform = waveform.unsqueeze(0)  # Add a batch dimension
        prediction = model(waveform)
        _, predicted_label = torch.max(prediction, dim=1)
    
    # Convert prediction to meaningful response
    response = {"prediction": predicted_label.item()}  # Adjust as necessary for your application
    
    return render_template('index.html', prediction=predicted_label.item())  # Adjust based on your model's output

if __name__ == '__main__':
    app.run(debug=True, port=5000)
