import numpy as np


def split_conformal_radii(scores):
    """Return radii at significance levels .01, .05 and .10."""
    scores = np.sort(np.asarray(scores).reshape(-1))
    m = len(scores)
    if not m or not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("Calibration scores must be nonempty, finite and nonnegative")
    ranks = ((m + 1) * np.array([99, 95, 90]) + 99) // 100
    return np.array([scores[k - 1] if k <= m else np.inf for k in ranks])
