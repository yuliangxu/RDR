# Relative Density Ratio (RDR)

This repository contains the official implementation for the paper:

> **"Distributional Evaluation of Generative Models via Relative Density Ratio"**  
> (Xu, 2025) — [arXiv link tbc]

The code provides a principled framework to evaluate the distributional discrepancy
between observed and generated data using the **relative density ratio**
\[
r(x) = \frac{2p(x)}{p(x) + q(x)} \in (0, 2),
\]
which measures how much (probability mass) each sample belongs to the real vs. generated distribution.

---

## 🔧 Repository Structure