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

Variational Autoencoders make the assumption that each of the samples {{< math >}}$ \mathbf x_i${{< /math >}} from the data set {{< math >}}$\mathbf{D}${{< /math >}} are iid and generated from the same probability distribution. Along with this, each sample has an latent feature vector {{< math >}}$\mathbf{z}_i${{< /math >}}. This implies the existence of a full joint distribution {{< math >}}$p(\mathbf{x},\mathbf{z})${{< /math >}}. When creating a generative model we are trying to discover the distribution {{< math >}}$p(\mathbf x | \mathbf z)${{< /math >}}, whereas when creating a discriminative model we are trying to discover the distribution {{< math >}}$p(\mathbf{z}|\mathbf{x})${{< /math >}}. VAEs are an attempt to utilize this latent structures joint structure to make optimization easier. 
