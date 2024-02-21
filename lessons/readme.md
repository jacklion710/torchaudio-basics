# Intro to Audio Processing and Classification with PyTorch

Welcome to this comprehensive guide on audio processing and classification using PyTorch. This curriculum is designed to introduce you to the fundamentals of working with audio data, leveraging PyTorch's powerful libraries to extract features and build predictive models. Whether you're new to audio analysis or looking to refine your skills, this series offers valuable insights into handling real-world audio processing tasks.

## Curriculum Overview

**Lesson 1: Introduction to PyTorch**
* Objective: Grasp the basics of PyTorch, focusing on tensors, operations, and computational graphs.
* Interactive Coding Example: Exploring tensor operations and understanding PyTorch's dynamic computation graph.

**Lesson 2: Working with Audio Data in PyTorch**
* Objective: Learn to load, visualize, and manipulate audio data using torchaudio.
* Interactive Coding Example: Loading an audio file, visualizing its waveform, and applying basic transformations like resampling.

**Lesson 3: Audio Feature Extraction with PyTorch**
* Objective: Understand and implement audio feature extraction techniques, crucial for machine learning models.
* Interactive Coding Example: Generating spectrograms, Mel-Frequency Cepstral Coefficients (MFCCs), and mel-spectrograms.

**Lesson 4: Building a Simple Audio Classification Model**
* Objective: Develop a basic convolutional neural network (CNN) model for classifying audio data.
* Interactive Coding Example: Defining a CNN in PyTorch, preparing audio data for training, and training the model.

**Lesson 5: Advanced Audio Processing Techniques**
* Objective: Explore techniques to enhance audio model performance, including data augmentation and feature engineering.
* Interactive Coding Example: Implementing audio data augmentation and exploring advanced feature extraction methods.

**Lesson 6: Transfer Learning in Audio with PyTorch**
* Objective: Apply transfer learning using pre-trained models to improve performance on audio classification tasks with limited data.
* Interactive Coding Example: Loading and fine-tuning a pre-trained model for a specific audio classification task.

**Lesson 7: Deploying Audio Models**
* Objective: Learn the basics of deploying trained audio models for practical applications.
* Interactive Coding Example: Exporting models for inference and understanding deployment strategies for audio models.

## Setting Up Your Environment
To get the most out of these lessons, you'll need a Python environment with PyTorch and torchaudio installed. While you can complete most exercises on a CPU, we recommend using a GPU for training models to significantly reduce training time. Here's how to set up your environment:

1. Install PyTorch and torchaudio: Follow the official PyTorch installation guide to install PyTorch. Make sure to select the appropriate version for your system and CUDA version to enable GPU support. 

Initialize conda env (Note: Pytorch requires python=3.7 and above):
```bash
conda create -n torch-study python=3.8
```

Activate conda env:
```bash
conda activate torch-study
```

[Pytorch website](https://pytorch.org/get-started/locally/); enter your platform and package manager, and copy the resulting command

The conda command will look something like:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=[CUDA_VERSION] -c pytorch -c nvidia
```

2. Additional Dependencies: Some lessons might require additional Python packages. Install them using conda:

```bash
conda install matplotlib seaborn pandas flask tqdm
```

GPU Training: If you have access to a GPU, ensure that your PyTorch installation is CUDA-enabled. Check your PyTorch installation with:

```py
import torch
print(torch.cuda.is_available())
```

(Pytorch GPU setup instructions)[https://mct-master.github.io/machine-learning/2023/04/25/olivegr-pytorch-gpu.html]

How do I know which CUDA_VERSION I need?

CUDA has both a driver API and a runtime API, and their API versions can be entirely different. This CLI command:
```bash
# Version of CUDA available in your environment
nvcc --version
```

will tell you the runtime API version, while
```bash
# Max version of CUDA available on your hardware
nvidia-smi
```

If you install Pytorch through your command line interface (CLI) like so…
```bash
conda install torch
``` 

…a CPU compiled version of pytorch will be installed.

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

# Project Structure

```
torchaudio-study/
├── __pycache__
├── audio/
├── data/
│   ├── audio/
│   │   ├── fold1/
│   │   ├── fold10/
│   │   ├── fold2/
│   │   ├── fold3/
│   │   ├── fold4/
│   │   ├── fold5/
│   │   ├── fold6/
│   │   ├── fold7/
│   │   ├── fold8/
│   │   └── fold9/
│   └── UrbanSound8K.csv
├── lessons/
│   ├── conclusion/
|   │   └── readme.md
│   ├── l1/
│   │   ├── lesson1.py
│   │   └── readme.md
│   ├── l2/
│   │   ├── lesson2.py
│   │   └── readme.md
│   ├── l3/
│   │   ├── lesson3.py
│   │   └── readme.md
│   ├── l4/
│   │   ├── lesson4.py
│   │   └── readme.md
│   ├── l5/
│   │   ├── lesson5.py
│   │   └── readme.md
│   ├── l6/
│   │   ├── lesson6.py
│   │   └── readme.md
│   ├── l7/
│   │   ├── lesson7.py
│   │   └── readme.md
│   └── readme.md
├── models/
├── plots/
├── readme.md
├── static/
│   ├── css/
│   │   └── styles.css
│   └── js/
│       └── script.js
├── templates/
│   └── index.html
└── utils/
    ├── model_defs.py
    └── test_model.py
```

Once you have your environment configured, you are good to go!