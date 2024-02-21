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
