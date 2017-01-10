# Module containing generally handy functions used by simulation module

import numpy as np
import bisect
import sys

import chippr
from chippr import defaults as d

def ingest(in_info):
    """
    Function reading in parameter file to define functions necessary for generation of posterior probability distributions

    Parameters
    ----------
    in_info: string or dict
        string containing path to plaintext input file or dict containing likelihood input parameters

    Returns
    -------
    in_dict: dict
        dict containing keys and values necessary for posterior probability distributions
    """
    if type(in_info) == str:
        with open(in_info) as infile:
            lines = (line.split(None) for line in infile)
            in_dict = {defn[0] : defn[1:] for defn in lines}
    else:
        in_dict = in_info
    return in_dict

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
