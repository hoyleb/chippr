import numpy as np
import os

import matplotlib as mpl
mpl.use('PS')
import matplotlib.pyplot as plt

import chippr
from chippr import utils as u
from chippr import plot_utils as pu

def plot_true_histogram(true_samps, n_bins=50, plot_loc='', plot_name='true_hist.png'):
    """
    Plots a histogram of true input values

    Parameters
    ----------
    true_samps: numpy.ndarray, float
        vector of true values of scalar input
    n_bins: int, optional
        number of histogram bins in which to place input values
    plot_loc: string, optional
        location in which to store plot
    plot_name: string, optional
        filename for plot
    """
    pu.set_up_plot()
    f = plt.figure(figsize=(5, 5))
    sps = f.add_subplot(1, 1, 1)
    sps.hist(true_samps, bins=n_bins, normed=1)
    sps.set_xlabel(r'$z_{true}$')
    sps.set_ylabel(r'$n(z_{true})$')
    f.savefig(os.path.join(plot_loc, plot_name))

def plot_obs_scatter(true_samps, obs_samps, plot_loc='', plot_name='obs_scatter.png'):
    """
    Plots a scatterplot of true and observed redshift values

    Parameters
    ----------
    true_samps: numpy.ndarray, float
        vector of true values of scalar input
    obs_samps: numpy.ndarray, float
        vector of observed values of scalar input
    plot_loc: string, optional
        location in which to store plot
    plot_name: string, optional
        filename for plot
    """
    pu.set_up_plot()
    f = plt.figure(figsize=(5, 5))
    sps = f.add_subplot(1, 1, 1)
    sps.scatter(true_samps, obs_samps)
    sps.set_xlabel(r'$z_{true}$')
    sps.set_ylabel(r'$z_{obs}$')
    f.savefig(os.path.join(plot_loc, plot_name))
