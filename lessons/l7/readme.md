# Lesson 7: Deploying Audio Models

In this lesson, we'll cover how to deploy our trained audio classification model using Flask, a lightweight WSGI web application framework in Python. This allows us to create a simple web interface for uploading audio files and getting predictions.

## Model Definitions

First, we have our model definition in `model_defs.py`. This is where we define our `Wav2Vec2ForAudioClassification` class, integrating the pre-trained Wav2Vec 2.0 model with a linear classifier.

```py
import torch

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
```

## Frontend Setup

Our frontend consists of a simple HTML form for file upload, styled with CSS, and a bit of JavaScript to handle the file upload asynchronously.

## HTML (`/templates/index.html`)
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Audio Classification</title>
    <link rel="stylesheet" href="/static/css/styles.css">
</head>
<body>
    <h1>Upload an Audio File</h1>
    <div id="drop-area">Drag and drop audio file here</div>
    <form id="file-form" action="/predict" method="post" enctype="multipart/form-data">
        <input type="file" id="fileElem" name="file" accept="audio/*" onchange="submitForm()">
        <label for="fileElem">Upload and Predict</label>
    </form>
    <div id="prediction-result" style="display:none;">
        <h2 id="prediction-text">Prediction: </h2>
    </div>
    <script src="/static/js/script.js"></script>
</body>
</html>
```

If you're not too familiar with HTML, CSS or JavaScript then have no fear! For the sake of this series, all you need to do is copy the code and follow along.

## CSS (`/static/css/styles.css`)
```css
body {
    font-family: Arial, sans-serif;
    text-align: center;
    margin: 20px;
}

#drop-area {
    width: 300px;
    height: 100px;
    border: 2px dashed #ccc;
    border-radius: 20px;
    text-align: center;
    line-height: 100px;
    margin: 20px auto;
    color: #ccc;
    cursor: pointer;
}

#drop-area.highlight {
    border-color: blue;
    color: blue;
}

input[type="submit"] {
    margin-top: 20px;
    padding: 10px 20px;
    cursor: pointer;
    background-color: #007bff;
    color: white;
    border: none;
    border-radius: 5px;
    font-size: 16px;
}

input[type="submit"]:hover {
    background-color: #0056b3;
}

h1 {
    color: #333;
}

label {
    cursor: pointer;
    color: #007bff;
    text-decoration: underline;
}

label:hover {
    color: #0056b3;
}

#prediction-result {
    margin-top: 20px;
    color: #007bff;
}
```

We're not too concerned with whats going on here, just know its going to make things on the UI look a little more neat and pleasing to look at.

## JS (`/static/css/script.js`)

```js
// static/js/script.js

document.addEventListener("DOMContentLoaded", function() {
    let dropArea = document.getElementById('drop-area');

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        dropArea.classList.add('highlight');
    }

    function unhighlight(e) {
        dropArea.classList.remove('highlight');
    }

    dropArea.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        let dt = e.dataTransfer;
        let files = dt.files;

        handleFiles(files);
    }

    function handleFiles(files) {
        let fileInput = document.getElementById('fileElem');
        fileInput.files = files;
        let event = new Event('change');
        fileInput.dispatchEvent(event);
        submitForm();
    }

    function submitForm() {
        let form = document.getElementById('file-form');
        let formData = new FormData(form);
    
        fetch('/predict', {
            method: 'POST',
            body: formData,
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                console.error('Error:', data.error);
            } else {
                // Display the prediction result
                document.getElementById('prediction-result').style.display = 'block';
                document.getElementById('prediction-text').innerText = 'Prediction: ' + data.prediction;
            }
        })
        .catch(error => console.error('Error:', error));
    }
});
```

This script is mostly defining the behavior of the drag and drop functionality as an event listener. Theres also a function for hanling the form submission so our users can get an inference when using the drag and drop feature.

## Backend Setup with Flask (`lesson7.py`)

In this section, we'll set up a Flask application to deploy our audio classification model. Flask is a Python web framework that's easy to use and perfect for small to medium web applications. Our Flask app will have two main parts: a route for uploading audio files and a route for predicting the class of uploaded audio files.

### Importing Required Libraries

First, we import the necessary libraries and modules. Flask is used to create our web app, `request` and `jsonify` are for handling HTTP requests and responses, and `render_template` is for rendering HTML templates. We also import our model definition and PyTorch libraries for model loading and inference.

```python
from flask import Flask, request, jsonify, render_template
import torch
import torchaudio
from model_defs import Wav2Vec2ForAudioClassification
import logging
```

## Setting Up Logging

Logging is configured for basic information. This is useful for debugging and tracking requests and responses.

```py
logging.basicConfig(level=logging.INFO)
```

## Flask App Initialization

We initialize our Flask app by creating an instance of the Flask class.

```py
app = Flask(__name__)
```

## Model and Class Names

We define the class names our model predicts, the target sample rate for audio files, the number of classes, and the feature size. This information is crucial for preprocessing and model inference.

```py
CLASS_NAMES = [
    "air_conditioner", "car_horn", "children_playing",
    "dog_bark", "drilling", "engine_idling",
    "gun_shot", "jackhammer", "siren", "street_music"
]
TARGET_SAMPLE_RATE = 16000
NUM_CLASSES = 10
FEATURE_SIZE = 768
```

## Loading the Trained Model

We load our pre-trained model using the model path and set it to evaluation mode. This step is essential for performing inference with the model.

```py
model_path = "../../models/pretrained_audio_classification_model.pth"
model_info = torch.load(model_path, map_location=torch.device('cpu'))
model_state_dict = model_info["model_state"]
feature_size = model_info["feature_size"]

model = Wav2Vec2ForAudioClassification(torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model(), NUM_CLASSES, feature_size)
model.load_state_dict(model_state_dict)
model.eval()
```

## Defining App Routes

Home Route
The home route (`'/'`) uses the `GET` method and serves the HTML form where users can upload audio files.

```py
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')
```

## Prediction Route

The prediction route (`'/predict'`) handles audio file uploads and predictions. It uses the `POST` method to receive the audio file, preprocesses it, runs it through the model, and returns the predicted class name as JSON.

```python
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    waveform, sample_rate = torchaudio.load(file)
    if sample_rate != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE)
        waveform = resampler(waveform)
    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    waveform = waveform.squeeze(0)

    # Use logging statements to debug
    logging.info(f"Sample rate: {sample_rate}, Waveform shape: {waveform.shape}")

    with torch.no_grad():
        waveform = waveform.unsqueeze(0)  # Add a batch dimension
        prediction = model(waveform)
        _, predicted_label = torch.max(prediction, dim=1)
        predicted_class = CLASS_NAMES[predicted_label.item()]
    
    return jsonify({'prediction': predicted_class})
```

## Running the App

Finally, we specify that if this script is executed as the main program, the Flask app will run on the default port 5000 in debug mode.

```py
if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

Once everything is set up, you can build the app and interact with it on a `localhost` server.

```bash
python lesson7.py
```

Once its running go to `http://127.0.0.1:5000` on a browser

This setup allows users to upload audio files through a web interface, and the backend processes these files to return the predicted audio class. Deploying the model in this way makes it accessible for real-world applications and testing.
