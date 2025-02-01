---
title: "Accelerated Bayesian Inference for Molecular Simulations using Local Gaussian Process Surrogate Models"
publication_types:
  - "2"
authors:
  - admin
  - Harry W. Sullivan
  - Abdur R. Shazed
  - Michael P. Hoepfner
doi: 10.1021/acs.jctc.3c01358
publication: Journal of Chemical Theory and Computation
publication_short: ""
url_code : https://github.com/hoepfnergroup/LGPMD
abstract: While Bayesian inference is the gold standard for uncertainty quantification and propagation, its use within physical chemistry encounters formidable computational barriers. These bottlenecks are magnified for modeling data with many independent variables, such as X-ray/neutron scattering patterns and electromagnetic spectra. To address this challenge, we apply a Bayesian framework accelerated via local Gaussian process (LGP) surrogate models. We show that the time-complexity of LGPs scales linearly in the number of independent variables, in stark contrast to the computationally expensive cubic scaling of conventional Gaussian processes. To illustrate the method, we trained a LGP surrogate model on the experimental radial distribution function of liquid neon, and observed a remarkable 288,000-fold speed-up compared to molecular dynamics with insignificant loss in predictive accuracy. We conclude that LGPs are robust and efficient surrogate models, poised to expand the application of Bayesian inference in molecular simulations to a broad spectrum of ever-advancing experimental data.
draft: false
featured: true
tags:
  - neutron scattering
  - statistical mechanics
  - thermodynamics
  - machine learning
image:
  filename: featured.png
  focal_point: Center
  preview_only: false
date: 2024-03-29T18:29:49.516Z
---
