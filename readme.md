# Torchaudio Study Project

This project focuses on exploring audio processing and classification using PyTorch and torchaudio. It consists of a series of lessons ranging from basic audio handling to deploying audio models for real-world applications.

## Dataset

The dataset used in this project is UrbanSound8K, which can be found [here](https://urbansounddataset.weebly.com/urbansound8k.html). It contains 8732 labeled sound excerpts of urban sounds from 10 classes, pre-sorted into ten folds for easier cross-validation and experimentation. No data including the audio data, plots or models are included in this repo.

### Lesson 1: Introduction to PyTorch
- **Objective**: Understand PyTorch's fundamentals, including tensors, operations, and computational graphs.

### Lesson 2: Working with Audio Data in PyTorch
- **Objective**: Learn how to load, visualize, and manipulate audio data in PyTorch.

### Lesson 3: Audio Feature Extraction with PyTorch
- **Objective**: Extract audio features crucial for machine learning models, such as spectrograms, MFCCs, and mel-spectrograms.

### Lesson 4: Building a Simple Audio Classification Model
- **Objective**: Create a basic model to classify audio data into different categories using convolutional neural networks (CNNs).

### Lesson 5: Advanced Audio Processing Techniques
- **Objective**: Explore advanced techniques for improving audio model performance, such as data augmentation and feature engineering.

### Lesson 6: Transfer Learning in Audio with PyTorch
- **Objective**: Utilize pre-trained models for audio-related tasks to improve performance with less data.

### Lesson 7: Deploying Audio Models
- **Objective**: Learn to deploy trained audio models for real-world applications.

## PyTorch GPU Setup

For setting up PyTorch with GPU support, follow these instructions. Note that PyTorch requires Python version 3.7 and above.

Initialize conda env (Note: Pytorch requires python=3.7 and above):
```bash
conda create -n torch-study python=3.8
```

Activate conda env:
```bash
conda activate torch-study
```

Visit the [Pytorch website](https://pytorch.org/get-started/locally/), select your preferences, and use the provided command to install. For example:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=[CUDA_VERSION] -c pytorch -c nvidia
```

How do I know which CUDA_VERSION I need?

CUDA has both a driver API and a runtime API, and their API versions can be entirely different. This CLI command:
```bash
nvcc --version
```

will tell you the runtime API version, while
```bash
nvidia-smi
```

points to the GPU driver, and it’s this CUDA version you need when installing Pytorch

If you install Pytorch through your command line interface (CLI) like so…
```bash
conda install torch
``` 

…a CPU compiled version of pytorch will be installed.

To check if Pytorch can find your GPU, use the following:
```py
import torch
torch.cuda.is_available()
```
This will return True if a GPU is found, False otherwise.

If your GPU cannot be found, it would be helpful to get some more feedback. Try sending something to the GPU. It will fail, and give you the reason:
```py
torch.zeros(1).cuda()
```

Should you want to start over because Pytorch is still not communicating with your GPU, you can remove your current environment and packages through your command line interface like so:
```bash
conda activate base
conda remove -n "YOUR_ENVIRONMENT_NAME" --all
```

If any GPU is recognized, you can now get more info about them or even decide which tensors and operations should go on which GPU.
```py
torch.cuda.current_device()     # The ID of the current GPU.
torch.cuda.get_device_name(id)  # The name of the specified GPU, where id is an integer.
torch.cuda.device(id)           # The memory address of the specified GPU, where id is an integer.
torch.cuda.device_count()       # The amount of GPUs that are accessible.
```

## Running the lessons:

Each lesson is encapsulated in its own Python script (e.g., lesson1.py, lesson2.py, etc.). To run a lesson, activate the conda environment and execute the script:

```bash
conda activate torch-study
python lesson1.py
```

## Acknowledgments
This project uses the UrbanSound8K dataset for demonstrating audio processing and classification techniques with PyTorch and torchaudio.