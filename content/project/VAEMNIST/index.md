---
title: Variational Auto Encoder Using MNIST
summary: An exploration into variatonal autoencoders using MNIST. 
tags:
  - Machine Learning
date: '2022-08-19T00:00:00Z'

# Optional external URL for project (replaces project detail page).
external_link: ''

image:
  caption: 2D latent representation of the MNIST dataset. 
  focal_point: Smart

links:
url_code: ''
url_pdf: ''
url_slides: ''
url_video: ''

# Slides (optional).
#   Associate this project with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides = "example-slides"` references `content/slides/example-slides.md`.
#   Otherwise, set `slides = ""`.
slides: ""
---

The goal of a Variational Autoencoder is to create a generative model, such that we can call it and it will generate a sample that mimics the training dataset. Variational Autoencoders make the assumption that each of the samples {{< math >}}$ \mathbf x_i${{< /math >}} from the data set {{< math >}}$\mathbf{D}${{< /math >}} are iid and generated from the same probability distribution. Along with this, each sample has an latent feature vector {{< math >}}$\mathbf{z}_i${{< /math >}}. This implies the existence of a full joint distribution {{< math >}}$p(\mathbf{x},\mathbf{z})${{< /math >}}. When creating a generative model we are trying to discover the distribution {{< math >}}$p(\mathbf x | \mathbf z)${{< /math >}}, whereas when creating a discriminative model we are trying to discover the distribution {{< math >}}$p(\mathbf{z}|\mathbf{x})${{< /math >}}. VAEs are an attempt to utilize this latent structure to make optimization easier.

To start, let us first get our imports as well as our datset. For the purpose of this tutorial we will be using the MNIST dataset. 

```
import matplotlib.pyplot as plt                    
import numpy as np                                 
import torch as torch                              
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor 
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class data(Dataset):
    
  def __init__(self, X, Y):
    self.X = X
    self.Y = Y
    if len(self.X) != len(self.Y):
      raise Exception("len(X) != len(Y)")

  def __len__(self):
    return len(self.X)

  def __getitem__(self, index):
    _x = self.X[index].unsqueeze(dim=0)
    _y = self.Y[index].unsqueeze(dim=0)

    return _x, _y

# Importing MNIST
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=ToTensor())
mnist_testset = datasets.MNIST(root='./data', train=True, download=True, transform=ToTensor())

bs = 200

# Data Loader
train_loader = torch.utils.data.DataLoader(dataset=mnist_trainset, batch_size=bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=mnist_testset, batch_size=bs, shuffle=False)
```
