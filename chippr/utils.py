# Module containing generally handy functions used by simulation and inference modules

import numpy as np

import chippr
from chippr import defaults as d

np.random.seed(d.seed)

def safe_log(arr, threshold=d.eps):
    """
    Takes the natural logarithm of an array that might contain zeros.

    Parameters
    ----------
    arr: ndarray
        values to be logged
    threshold: float, optional
        small, positive value to replace zeros and negative numbers

    Returns
    -------
    logged: ndarray
        logarithms, with approximation in place of zeros and negative numbers
    """
    shape = np.shape(arr)
    flat = arr.flatten()
    logged = np.log(np.array([max(a,threshold) for a in flat])).reshape(shape)
    return logged
