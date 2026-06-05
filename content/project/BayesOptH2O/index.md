---
title: Bayesian Optimization of a Molecular Simulation of H₂O
summary: Using Local Gaussian Process surrogates and Hamiltonian Monte Carlo to find the best classical water model that reproduces experimental scattering measurements.
tags:
  - Machine Learning
  - Physics
date: '2023-12-01T00:00:00Z'
weight: 7

external_link: ''

image:
  caption: ''
  focal_point: Smart

links:

url_code: ''
url_pdf: 'BayesOptH2O.pdf'
url_slides: ''
url_video: ''
---

A joint project with Leqian (Tim) Tan applying Bayesian optimization and uncertainty quantification to classical molecular models of water. We trained a Local Gaussian Process surrogate on the oxygen-oxygen radial distribution function from 512 LAMMPS simulations, then performed inference over the molecular model parameters using Hamiltonian Monte Carlo. The HMC-optimized parameters yielded improved liquid structure prediction compared to TIP4P/2005, with full posterior uncertainty quantification on the parameters and RDF predictions.
