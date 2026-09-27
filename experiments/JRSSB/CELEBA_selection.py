"""Shared numerical and data code for the final Agent 3 workflow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pytorch_fid.fid_score import calculate_frechet_distance

@dataclass
class FIDResult:
    fid: float
    mean_term: float
    covariance_term: float

@dataclass
class FeatureMoments:
    mean: np.ndarray
    covariance: np.ndarray
    sample_size: int

def fid_from_moments(p: FeatureMoments, q: FeatureMoments) -> FIDResult:
    if p.mean.shape != q.mean.shape or p.covariance.shape != q.covariance.shape:
        raise ValueError("FID moment dimensions do not match.")
    mean_term = float(np.sum((p.mean - q.mean) ** 2))
    fid = float(calculate_frechet_distance(p.mean, p.covariance, q.mean, q.covariance))
    return FIDResult(fid=fid, mean_term=mean_term, covariance_term=fid - mean_term)
