---
title: Boltzmann Sampling with Stochastic Interpolants
summary: Using deep generative models to sample Boltzmann distributions for molecular systems, addressing the rare event problem in molecular dynamics.
tags:
  - Machine Learning
  - Physics
date: '2025-05-01T00:00:00Z'
weight: 9

external_link: ''

image:
  caption: Ramachandran plot comparison — data vs generated samples
  focal_point: Smart

links:

url_code: ''
url_pdf: 'BoltzmannSampling.pdf'
url_slides: ''
url_video: ''
---

A joint project with Zichen Huang exploring how deep generative models can efficiently sample Boltzmann distributions for molecular systems. We use stochastic interpolants to learn a transport map from a simple base distribution to the target Boltzmann distribution, bypassing the rare event problem that plagues conventional molecular dynamics. Applied to alanine dipeptide, the model generates samples whose Ramachandran plot free energy surface closely matches the ground truth from long MD trajectories.
