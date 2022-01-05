"""
Implements Generalized Seasonal Block Bootstrap for periodic data.
Dudek et al. 2013
See https://sci-hub.st/10.1002/jtsa.12053
"""
import random
import numpy as np


def gsbb_sampler(n: int = 1000, blocksize: int = 25, period: int = 100):
    """Generalized Seasonal Block Bootstrap. Dudek et al. 2013
    blocksize=25 suggested a good value according to their numerical results.

    n = number of samples
    blocksize = number of samples in each block
    period = period in number of samples

    Variables named according to the paper for simpler understanding.
    """
    # Step 0 - sanity check
    if n == period:
        return list(
            range(n)
        )  # if there is a single period in the data, we cannot bootstrap.
    # Step 1
    # Choose a (positive) integer block size (b) < n and let l = b // n
    b = blocksize
    l, d = n // blocksize, period
    # Step 2
    indices = []
    for i in range(l + 1):  # +1 otherwise we miss the last block
        t = i * b

        r1 = t // d
        r2 = (n - b - t) // d

        s = range(t - d * r1, t + d * r2, d)
        k = random.choice(s)
        # Step 3
        indices.extend([*range(k, k + b)])

    return indices[:n]  # we don't need the rest of the last block


def gsbb_bootstrap(data, period, blocksize=25, n_boots=1000):
    """Generate bootstrap samples according to GSBB.
    For now assumes data is 1D.
    PERIOD IN DATA SAMPLES
    """
    samples = [
        data[gsbb_sampler(n=data.shape[0], blocksize=blocksize, period=period)]
        for i in range(n_boots)
    ]
    return np.dstack(samples)


def get_ci(bootstraps, level=95):
    edge = (100 - level) / 2
    percentiles = edge, 100 - edge
    emin, emax = np.percentile(bootstraps, percentiles, axis=2)
    return emin, emax


def gsbb_bootstrap_ci(data, period, blocksize=25, n_boots=1000, level=95):
    bootstraps = gsbb_bootstrap(data, period, blocksize=blocksize, n_boots=n_boots)
    cis = get_ci(bootstraps=bootstraps, level=level)
    return cis
