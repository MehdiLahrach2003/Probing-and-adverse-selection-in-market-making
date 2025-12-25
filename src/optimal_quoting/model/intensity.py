from __future__ import annotations

import math


def intensity_exp(A: float, k: float, delta: float) -> float:
    """
    Exponential intensity model:
        λ(δ) = A exp(-k δ), δ >= 0

    Parameters
    ----------
    A : float
        Baseline intensity at δ=0 (must be > 0).
    k : float
        Decay rate (must be > 0).
    delta : float
        Distance to mid (must be >= 0).

    Returns
    -------
    float
        λ(δ)
    """
    if A <= 0:
        raise ValueError("A must be > 0")
    if k <= 0:
        raise ValueError("k must be > 0")
    if delta < 0:
        raise ValueError("delta must be >= 0")
    return A * math.exp(-k * delta)
