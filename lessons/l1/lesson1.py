# lesson1.py
# Import PyTorch
import torch

# Create a tensor
x = torch.tensor([1, 2, 3])
print("Tensor:", x)

# Basic operations
y = x + x
print("Tensor addition:", y)

# More operations
z = x * x
print("Element-wise multiplication:", z)

# Create a 2D tensor
x = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Flatten the tensor
flat_x = x.flatten()

print("Flattened Tensor:", flat_x)

# Create tensors of different sizes
a = torch.tensor([1, 2, 3])
b = torch.tensor([[0], [1], [2]])

# Broadcasting allows for element-wise addition
result = a + b

print("Broadcasted Addition:", result)

# 2D tensor (matrix) of shape [2, 3]
matrix = torch.tensor([[1, 2, 3],
                       [4, 5, 6]])

# 1D tensor (vector) of shape [3]
vector = torch.tensor([1, 0, 1])

# Broadcasting the vector across each row of the matrix
result = matrix + vector

print("Broadcasted Addition Result:\n", result)

# Create a tensor
c = torch.tensor([[1, 2], [3, 4]])

# Pad the tensor to a larger size
padded_c = torch.nn.functional.pad(c, (1, 1, 1, 1), 'constant', 0)

print("Padded Tensor:", padded_c)

# In-place addition
x = torch.tensor([1, 2, 3])
x.add_(1)  # Adds 1 to each element in-place
print("In-place addition:", x)

# Example of a stereo audio tensor
stereo_audio = torch.rand(2, 48000)  # Simulating stereo audio with random values
print("Stereo audio tensor shape:", stereo_audio.shape)

# Example of a mono audio tensor
mono_audio = torch.rand(48000)  # Simulating mono audio with random values
print("Mono audio tensor shape:", mono_audio.shape)

# Simulating a stereo audio tensor with random data
# Assume each channel has 1000 samples
stereo_audio = torch.rand((2, 1000))  # Shape is [2, 1000] for stereo

# Collapsing the stereo audio into mono by averaging the two channels
mono_audio = torch.mean(stereo_audio, dim=0)  # Take the mean across the first dimension

print("Mono audio shape:", mono_audio.shape)  # The shape will be [1000], representing mono audio

# Simulating a stereo audio tensor with random data
stereo_audio = torch.rand((2, 1000))  # Shape is [2, 1000] for stereo

# Collapsing the stereo audio into mono by averaging the two channels
mono_audio = torch.mean(stereo_audio, dim=0)  # Take the mean across the first dimension

print("Mono audio shape:", mono_audio.shape)  # The shape will be [1000], representing mono audio