# Relative Density Ratio (RDR)

This repository contains the official implementation for the paper:

> **"Distributional Evaluation of Generative Models via Relative Density Ratio"**  
> [arXiv link tbc]

The code provides a principled framework to evaluate the distributional discrepancy
between observed and generated data using the **relative density ratio**
\[
r(x) = \frac{2p(x)}{p(x) + q(x)} \in (0, 2),
\]
which measures how much (probability mass) each sample belongs to the real vs. generated distribution.

---

## 🔧 Repository Structure
```
RDR/
├─ checkpoints/
│ └─ ratio_ddim_celeba64_nep2.pt # pretrained ratio estimator for CelebA64-DDIM
│
├─ experiments/
│ ├─ AGP/
│ │ ├─ AGP1_data_preprocessing.py # preprocessing for American Gut Project data
│ │ ├─ AGP2_ICFM.py # main microbiome experiment (ICFM model)
│ │ └─ microbiome_unet.py # auxiliary U-Net model for microbiome data
│ │
│ ├─ CelebA_ddim.py # DDIM experiment on CelebA-64
│ ├─ MNIST_batch.py # MNIST experiment (batch implementation)
│ ├─ MNIST_batch.ipynb # interactive MNIST notebook
│ └─ oneD_toy.py / oneD_toy.ipynb # 1D toy example illustrating RDR behavior
│ 
├─ utils/
│ ├─ AGP_help.py # AGP data loading, CLR transform, diversity metrics
│ ├─ CelebA_help.py # CelebA dataset utilities
│ ├─ DRE_func.py # core density ratio estimation via f-divergence
│ ├─ DRE_batch.py # batch-level SGD-based ratio training utilities
│ ├─ MNIST_help.py # MNIST data helpers
│ ├─ microbiome_help.py # microbiome helper functions
│ ├─ sampler_ddim_celeba64.py # DDIM sampling for CelebA64
│ ├─ gen_ddim_with_pbar.py # DDIM generator with progress bar
│ ├─ help_func.py # general-purpose helper functions
│ └─ vae.py # simple VAE baseline
│
└─ readme.md
```