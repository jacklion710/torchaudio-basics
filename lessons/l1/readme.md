# Lesson 1: Introduction to PyTorch Fundamentals

Welcome to the first lesson in our series on audio processing with PyTorch. In this lesson, we'll dive into the basics of PyTorch, a powerful library for building machine learning models. We'll explore how to create tensors, perform basic operations, and understand the core concepts that make PyTorch a favorite among data scientists and AI researchers.

## What is PyTorch?
P
yTorch is an open-source machine learning library developed by Facebook's AI Research lab. It provides a flexible and intuitive framework for building and training neural networks, with a strong focus on deep learning. PyTorch is known for its dynamic computational graph, ease of use, and efficient memory usage.

## Getting Started with Tensors
Tensors are the fundamental building blocks in PyTorch. They are similar to arrays and matrices and are used to encode the inputs and outputs of a model, as well as the model's parameters.

## Creating a Tensor
Let's start by creating a simple tensor. Tensors in PyTorch can be created using the torch.tensor() function.

```py
import torch

# Create a tensor
x = torch.tensor([1, 2, 3])
print("Tensor:", x)
```

In this code snippet, we import the torch module and create a tensor x from a Python list [1, 2, 3]. We can then print the tensor to see its contents.

## Basic Operations

PyTorch tensors support a wide range of operations. Let's perform some basic arithmetic operations on our tensor.

### Tensor Addition

You can add two tensors element-wise using the + operator.

```py
# Basic operations
y = x + x
print("Tensor addition:", y)
```

This operation adds the tensor x to itself, resulting in a new tensor y where each element is the sum of the corresponding elements in x.

### Element-wise Multiplication
Similarly, tensors can be multiplied element-wise using the * operator.

```py
# More operations
z = x * x
print("Element-wise multiplication:", z)
```

This code multiplies each element of the tensor x by itself, demonstrating how to perform element-wise multiplication.

## Conclusion
Congratulations on completing your first lesson on PyTorch! You've learned how to create tensors and perform basic arithmetic operations. These operations are the building blocks for more complex computations and neural network models that you'll encounter as you delve deeper into PyTorch.

In the next lesson we will explore how to work with audio data in PyTorch, including loading, visualizing, and transforming sound files.

