# Module containing generally handy functions used by simulation module

import numpy as np
import bisect
import sys

import chippr
from chippr import defaults as d

def choice(weights):
    """
    Function sampling discrete distribution

    Parameters
    ----------
    weights: numpy.ndarray
        relative probabilities for each category

    Returns
    -------
    index: int
        chosen category
    """
    cdf_vals = np.cumsum(weights) / np.sum(weights)
    x = np.random.random()
    index = bisect.bisect(cdf_vals, x)
    return index
