import numpy as np
import os

import matplotlib as mpl
mpl.use('PS')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import chippr
from chippr import defaults as d
from chippr import utils as u
from chippr import plot_utils as pu

def plot_true_histogram(true_samps, n_bins=(10, 50), plot_loc='', prepend='', plot_name='true_hist.png'):
    """
    Plots a histogram of true input values

    Parameters
    ----------
    true_samps: numpy.ndarray, float
        vector of true values of scalar input
    n_bins: tuple, int, optional
        number of histogram bins in which to place input values, coarse and fine
    plot_loc: string, optional
        location in which to store plot
    plot_name: string, optional
        filename for plot
    prepend: str, optional
        prepend string to plot name
    """
    pu.set_up_plot()
    f = plt.figure(figsize=(5, 5))
    sps = f.add_subplot(1, 1, 1)
    sps.hist(true_samps, bins=n_bins[1], normed=1, color='k', alpha=0.5, log=True)
    sps.hist(true_samps, bins=n_bins[0], normed=1, color='y', alpha=0.5, log=True)
    sps.set_xlabel(r'$z_{true}$')
    sps.set_ylabel(r'$n(z_{true})$')
    f.savefig(os.path.join(plot_loc, prepend+plot_name), bbox_inches='tight', pad_inches = 0, dpi=d.dpi)

    return

def plot_prob_space(z_grid, p_space, plot_loc='', prepend='', plot_name='prob_space.png'):
    """
    Plots the 2D probability space of z_spec, z_phot

    Parameters
    ----------
    p_space: numpy.ndarray, float
        probabilities on the grid
    z_grid: numpy.ndarray, float
        fine grid of redshifts
    plot_loc: string, optional
        location in which to store plot
    plot_name: string, optional
        filename for plot
    prepend: str, optional
        prepend string to plot name
    """
    pu.set_up_plot()
    f = plt.figure(figsize=(5, 5))
    plt.subplot(1, 1, 1)
    grid_len = len(z_grid)
    grid_range = range(grid_len)
    # to_plot = u.safe_log(p_space.evaluate(all_points.reshape((grid_len**2, 2))).reshape((grid_len, grid_len)))
    # to_plot.reshape((len(z_grid), len(z_grid)))
    # print('trying to plot '+str(np.shape(to_plot)))
    # plt.pcolormesh(z_grid, z_grid, to_plot, cmap='viridis')
    all_points = np.array([[(z_grid[kk], z_grid[jj]) for kk in grid_range] for jj in grid_range])
    orig_shape = np.shape(all_points)
    # all_vals = np.array([[p_space.evaluate_one(np.array([z_grid[jj], z_grid[kk]])) for jj in range(len(z_grid))] for kk in range(len(z_grid))])
    all_vals = p_space.evaluate(all_points.reshape((orig_shape[0]*orig_shape[1], orig_shape[2]))).reshape((orig_shape[0], orig_shape[1]))
    plt.pcolormesh(z_grid, z_grid, u.safe_log(all_vals), cmap='viridis')
    plt.plot(z_grid, z_grid, color='k')
    plt.colorbar()
    plt.xlabel(r'$z_{spec}$')
    plt.ylabel(r'$z_{phot}$')
    plt.axis([z_grid[0], z_grid[-1], z_grid[0], z_grid[-1]])
    f.savefig(os.path.join(plot_loc, prepend+plot_name), bbox_inches='tight', pad_inches = 0, dpi=d.dpi)
    return

def plot_mega_scatter(zs, pfs, z_grid, grid_ends, plot_loc='', prepend='', plot_name='mega_scatter.png'):
    """
    Plots a scatterplot of true and observed redshift values

    Parameters
    ----------
    zs: numpy.ndarray, float
        matrix of spec, phot values
    z_grid: numpy.ndarray, float
        fine grid of redshifts
    grid_ends: numpy.ndarray, float
        coarse bin ends
    pfs: numpy.ndarray, float
        matrix of posteriors evaluated on a fine grid
    plot_loc: string, optional
        location in which to store plot
    plot_name: string, optional
        filename for plot
    prepend: str, optional
        prepend string to plot name
    """
    n = len(zs)
    zs = zs.T
    true_zs = zs[0]
    obs_zs = zs[1]

    pu.set_up_plot()
    f, scatplot = plt.subplots(figsize=(7.5, 7.5))
    f.subplots_adjust(hspace=0)

    # true_hist = np.hist(true_zs, grid_ends)
    # sps_x.step(true_hist[0], true_hist[1], c='k', where='mid')
    # sps_x.hist(obs_zs, bins=info['bin_ends'][0])
    # sps_y.hist(true_zs)

    scatplot.plot(z_grid, z_grid, color='k', alpha=0.5, linewidth=1.)
    scatplot.scatter(true_zs, obs_zs, c='k', marker='.', s=1., alpha=0.1)
    randos = np.floor(n / (d.plot_colors + 1)) * np.arange(1., d.plot_colors + 1)# np.random.choice(range(len(z_grid)), d.plot_colors)
    randos = randos.astype(int)
    max_pfs = np.max(pfs)
    sort_inds = np.argsort(obs_zs)
    sorted_pfs = pfs[sort_inds]
    sorted_true = true_zs[sort_inds]
    sorted_obs = obs_zs[sort_inds]
    for r in range(d.plot_colors):
        pf = sorted_pfs[randos[r]]
        norm_pf = pf / max_pfs
        scatplot.step(z_grid, norm_pf + sorted_obs[randos[r]], c=pu.colors[r], where='mid')# plt.plot(z_grid, norm_pf + sorted_obs[randos[r]], c='k')
        scatplot.hlines(sorted_obs[randos[r]], min(z_grid), max(z_grid), color='k', alpha=0.1, linestyle='--', linewidth=0.5)
        scatplot.scatter(sorted_true[randos[r]], sorted_obs[randos[r]], marker='+', c=pu.colors[r])
    scatplot.set_xlabel(r'$z_{true}$')
    scatplot.set_ylabel(r'$z_{obs}$')

    scatplot.set_aspect(1.)
    # gs = gridspec.GridSpec(3, 3)
    # sps = f.add_subplot(gs[:-1, :-1])
    # sps_x = f.add_subplot(gs[:-1, -1], share)
    # sps_y = f.add_subplot(gs[-1, :-1])
    divider = make_axes_locatable(scatplot)
    histx = divider.append_axes('top', 1.2, pad=0., sharex=scatplot)
    histy = divider.append_axes('right', 1.2, pad=0., sharey=scatplot)

    histx.xaxis.set_tick_params(labelbottom=False)
    histy.yaxis.set_tick_params(labelleft=False)
    histx.hist(true_zs, bins=grid_ends)
    histy.hist(obs_zs, bins=grid_ends, orientation='horizontal')

    # plt.draw()
    f.savefig(os.path.join(plot_loc, prepend+plot_name), bbox_inches='tight', pad_inches=0, dpi=d.dpi)
    # plt.close()
    return

def plot_scatter(zs, pfs, z_grid, plot_loc='', prepend='', plot_name='scatter.png'):
    """
    Plots a scatterplot of true and observed redshift values

    Parameters
    ----------
    zs: numpy.ndarray, float
        matrix of spec, phot values
    z_grid: numpy.ndarray, float
        fine grid of redshifts
    pfs: numpy.ndarray, float
        matrix of posteriors evaluated on a fine grid
    plot_loc: string, optional
        location in which to store plot
    plot_name: string, optional
        filename for plot
    prepend: str, optional
        prepend string to plot name
    """
    n = len(zs)
    zs = zs.T
    true_zs = zs[0]
    obs_zs = zs[1]
    pu.set_up_plot()
    f = plt.figure(figsize=(5, 5))
    sps = f.add_subplot(1, 1, 1)
    sps.plot(z_grid, z_grid, color='r', alpha=0.5, linewidth=2.)
    sps.scatter(true_zs, obs_zs, c='g', marker='.', s = 1., alpha=0.1)
    randos = np.floor(n / (d.plot_colors + 1)) * np.arange(1., d.plot_colors + 1)# np.random.choice(range(len(z_grid)), d.plot_colors)
    randos = randos.astype(int)
    max_pfs = np.max(pfs)
    sort_inds = np.argsort(obs_zs)
    sorted_pfs = pfs[sort_inds]
    sorted_true = true_zs[sort_inds]
    sorted_obs = obs_zs[sort_inds]
    for r in range(d.plot_colors):
        pf = sorted_pfs[randos[r]]
        norm_pf = pf / max_pfs / (d.plot_colors + 1)
        plt.step(z_grid, norm_pf + sorted_obs[randos[r]], c='k', where='mid')# plt.plot(z_grid, norm_pf + sorted_obs[randos[r]], c='k')
        plt.hlines(sorted_obs[randos[r]], min(z_grid), max(z_grid), color='k', alpha=0.5, linestyle='--')
        plt.scatter(sorted_true[randos[r]], sorted_obs[randos[r]], marker='+', c='r')
    sps.set_xlabel(r'$z_{true}$')
    sps.set_ylabel(r'$z_{obs}$')
    f.savefig(os.path.join(plot_loc, prepend+plot_name), bbox_inches='tight', pad_inches = 0, dpi=d.dpi)
    # plt.close()
    return

def plot_obs_scatter(true_vals, pfs, z_grid, plot_loc='', prepend='', plot_name='obs_scatter.png'):
    """
    Plots a scatterplot of true and observed redshift values

    Parameters
    ----------
    true_vals: numpy.ndarray, float
        2 * n_gals array of true values of scalar input
    pfs: numpy.ndarray, float
        matrix of interim posteriors evaluated on a fine grid
    z_grid: numpy.ndarray, float
        vector of bin midpoints
    plot_loc: string, optional
        location in which to store plot
    plot_name: string, optional
        filename for plot
    prepend: str, optional
        prepend string to plot name
    """
    true_zs = true_vals[0]
    obs_zs = true_vals[1]#np.array([z_grid[np.argmax(pf)] for pf in pfs])
    max_pfs = np.max(pfs)
    n = len(true_zs)
    dz = (max(z_grid) - min(z_grid)) / len(z_grid)
    jitters = np.random.uniform(-1. * dz / 2., dz / 2., n)
    obs_zs = obs_zs + jitters
    pu.set_up_plot()
    f = plt.figure(figsize=(5, 5))
    sps = f.add_subplot(1, 1, 1)
    sps.scatter(true_zs, obs_zs, c='k', marker='.', s = 1., alpha=0.1)
    randos = np.floor(n / (d.plot_colors + 1)) * np.arange(1., d.plot_colors + 1)# np.random.choice(range(len(z_grid)), d.plot_colors)
    randos = randos.astype(int)
    ordered = np.argsort(obs_zs)
    sorted_pfs = pfs[ordered]
    sorted_true = true_zs[ordered]
    sorted_obs = obs_zs[ordered]
    for r in range(d.plot_colors):
        pf = sorted_pfs[randos[r]]
        plt.scatter(sorted_true[randos[r]], sorted_obs[randos[r]], marker='+', c='r')
        norm_pf = pf / max_pfs / (d.plot_colors + 1)
        plt.step(z_grid, norm_pf + sorted_obs[randos[r]], c='k', where='mid')
        plt.hlines(sorted_obs[randos[r]], min(z_grid), max(z_grid), color='k', alpha=0.5, linestyle='--')
    sps.set_xlabel(r'$z_{true}$')
    sps.set_ylabel(r'$\hat{z}_{MAP}$')
    f.savefig(os.path.join(plot_loc, prepend+plot_name), bbox_inches='tight', pad_inches = 0, dpi=d.dpi)

    return
