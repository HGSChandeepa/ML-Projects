# Convolutional Neural Networks with PyTorch

This repository contains a notebook that demonstrates the fundamentals of Convolutional Neural Networks (CNNs) using PyTorch.  
It covers tensor shapes, convolution operations, pooling layers, and key building blocks of CNNs.

🧠 [Medium Article: Convolutional Neural Networks](https://medium.com/data-science/applied-deep-learning-part-4-convolutional-neural-networks-584bc134c1e2)


---

## Table of Contents
- [Convolutional Neural Networks with PyTorch](#convolutional-neural-networks-with-pytorch)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Environment Setup](#environment-setup)
  - [Data Representation](#data-representation)
  - [Convolutional Layers](#convolutional-layers)
    - [Key Concepts](#key-concepts)
  - [Pooling Layers](#pooling-layers)
  - [Activation Functions](#activation-functions)
  - [Building a CNN](#building-a-cnn)
  - [Training Workflow](#training-workflow)
  - [Results and Visualizations](#results-and-visualizations)
  - [References](#references)
  - [Next Steps](#next-steps)

---

## Introduction
A Convolutional Neural Network (CNN) is a deep learning model designed for analyzing visual data such as images and videos.  
Instead of processing images as flat vectors, CNNs leverage spatial hierarchies using convolutions, pooling, and feature maps.

This notebook explains:
- How image tensors are represented in PyTorch.  
- What convolution and pooling operations do.  
- How CNN layers transform image dimensions.  
- The role of filters, stride, and padding.  

---

## Environment Setup
We use the following libraries:
- **PyTorch**: For tensor operations and deep learning.  
- **Torchvision**: Provides datasets and image transformations.  
- **tqdm**: Progress bar for training loops.  
- **Matplotlib**: For visualizations.  

```python
import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm.auto import tqdm
import matplotlib.pylab as plt
````

---

## Data Representation

* Images are represented as tensors in the form:

  ```
  [batch_size, channels, height, width]
  ```

* Example:

  ```python
  images = torch.randn(size=(32, 3, 64, 64))
  ```

  * `32`: batch size (number of images)
  * `3`: color channels (RGB)
  * `64 x 64`: image height and width

A single image has shape `[3, 64, 64]`, which can be expanded with `.unsqueeze(dim=0)` to simulate batch size.

---

## Convolutional Layers

A convolutional layer (`nn.Conv2d`) extracts features from an image using small filters (kernels).

```python
conv_layer = nn.Conv2d(
    in_channels=3,
    out_channels=10,
    kernel_size=3,
    stride=1,
    padding=0
)
```

### Key Concepts

* **In Channels**: Number of input channels (3 for RGB).
* **Out Channels**: Number of filters, which defines the number of feature maps.
* **Kernel Size**: Size of filter (e.g., `3x3`).
* **Stride**: Step size of filter movement.
* **Padding**: Extra pixels added around edges to control output size.

The convolution layer outputs a tensor of shape:

```
[batch_size, out_channels, new_height, new_width]
```

---

## Pooling Layers

Pooling reduces spatial dimensions while retaining key features.
The most common type is MaxPooling (`nn.MaxPool2d`).

```python
max_pool_layer = nn.MaxPool2d(kernel_size=2)
```

* Kernel size = 2 halves height and width.
* Helps reduce computations and prevents overfitting.

---

## Activation Functions

After convolution, we apply non-linear activation functions like ReLU:

```python
nn.ReLU()
```

* ReLU (Rectified Linear Unit) replaces negative values with 0.
* Introduces non-linearity, which allows learning complex features.

---

## Building a CNN

A simple CNN architecture includes:

1. Convolution → Activation → Pooling (feature extraction).
2. Fully Connected (Linear) Layers (classification).

Example:

```python
model = nn.Sequential(
    nn.Conv2d(3, 10, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(10 * 31 * 31, 10) # example output classes
)
```

---

## Training Workflow

1. **Forward Pass**: Input → CNN → Predictions.
2. **Loss Function**: Measure error (e.g., CrossEntropyLoss).
3. **Backward Pass**: Gradient calculation via backpropagation.
4. **Optimizer**: Update weights (e.g., Adam, SGD).
5. Repeat for multiple epochs until convergence.

---

## Results and Visualizations

The notebook demonstrates:

* Input vs. transformed feature maps.
* How dimensions change after each layer.
* The effect of convolution and pooling.

Visualizations can help in understanding what CNNs "see" at different layers.

---

## References

* [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
* [Deep Learning with PyTorch](https://pytorch.org/tutorials/)
* [CS231n Convolutional Neural Networks](http://cs231n.github.io/convolutional-networks/)

---

## Next Steps

* Add training with a real dataset (e.g., CIFAR-10, MNIST).
* Experiment with different kernel sizes, strides, and padding.
* Visualize learned filters and feature maps.

