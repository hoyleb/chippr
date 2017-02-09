import numpy as np
import os
import scipy as sp

import matplotlib as mpl
mpl.use('PS')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import chippr
from chippr import defaults as d
from chippr import utils as u
from chippr import plot_utils as pu
from chippr import stat_utils as s

lnz, nz = r'$\ln[n(z)]$', r'$n(z)$'

s_tru, w_tru, a_tru, c_tru, d_tru, l_tru = '--', 0.5, 1., 'k', [(0, (1, 1))], 'True '
s_int, w_int, a_int, c_int, d_int, l_int = '--', 0.5, 0.5, 'k', [(0, (1, 1))], 'Interim '
s_stk, w_stk, a_stk, c_stk, d_stk, l_stk = '--', 1.5, 1., 'g', [(0, (7.5, 7.5))], 'Stacked '
s_map, w_map, a_map, c_map, d_map, l_map = '--', 1., 1., 'r', [(0, (5., 5.))], 'MMAP '
s_exp, w_exp, a_exp, c_exp, d_exp, l_exp = '--', 0.5, 1., 'r', [(0, (2.5, 2.5))], 'MExp '
s_mle, w_mle, a_mle, c_mle, d_mle, l_mle = '--', 2.5, 1., 'b', [(0, (1, 1))], 'MMLE '
s_smp, w_smp, a_smp, c_smp, d_smp, l_smp = '--', 1., 1., 'k', [(0, (1, 1))], 'Sampled '
s_bfe, w_bfe, a_bfe, c_bfe, d_bfe, l_bfe = '--', 1.5, 1., 'b', [(0, (1, 1))], 'Mean of\n Samples '

cmap = np.linspace(0., 1., d.plot_colors)
colors = [cm.viridis(i) for i in cmap]

def plot_ivals(info):
    """
    Plots the initial values given to the sampler

    Parameters
    ----------
    info: dict
        dictionary of stored information from log_z_dens object

    Returns
    -------
    f: matplotlib figure
        figure object
    """
    return f

def set_up_burn_in_plots(n_bins, n_walkers):
    """
    Creates plotting objects for sampler progress

    Parameters
    ----------
    n_bins: int
        number of parameters defining n(z)

    Returns
    -------
    plot_information: tuple
        contains figure and subplot objects for Gelman-Rubin evolution,
        autocorrelation times, acceptance fractions, posterior probabilities,
        and chain evolution
    """
    pu.set_up_plot()

    f_gelman_rubin_evolution = plt.figure(figsize=(5, 5))
    sps_gelman_rubin_evolution = f_gelman_rubin_evolution.add_subplot(1, 1, 1)
    f_gelman_rubin_evolution.subplots_adjust(hspace=0, wspace=0)
    sps_gelman_rubin_evolution.set_ylabel(r'Gelman-Rubin Statistic')
    sps_gelman_rubin_evolution.set_xlabel(r'accepted sample number')
    gelman_rubin_evolution_plot = [f_gelman_rubin_evolution, sps_gelman_rubin_evolution]

    f_autocorrelation_times = plt.figure(figsize=(5, 5))
    sps_autocorrelation_times = f_autocorrelation_times.add_subplot(1, 1, 1)
    f_autocorrelation_times.subplots_adjust(hspace=0, wspace=0)
    sps_autocorrelation_times.set_ylabel(r'autocorrelation time')
    sps_autocorrelation_times.set_xlabel(r'accepted sample number')
    sps_autocorrelation_times.set_ylim(0, 100)
    autocorrelation_times_plot = [f_autocorrelation_times, sps_autocorrelation_times]

    f_acceptance_fractions = plt.figure(figsize=(5, 5))
    sps_acceptance_fractions = f_acceptance_fractions.add_subplot(1, 1, 1)
    f_acceptance_fractions.subplots_adjust(hspace=0, wspace=0)
    sps_acceptance_fractions.set_ylim(0, 1)
    sps_acceptance_fractions.set_ylabel('acceptance fraction per bin')
    sps_acceptance_fractions.set_xlabel('number of iterations')
    acceptance_fractions_plot = [f_acceptance_fractions, sps_acceptance_fractions]

    f_posterior_probabilities = plt.figure(figsize=(5, 5))
    sps_posterior_probabilities = f_posterior_probabilities.add_subplot(1, 1, 1)
    f_posterior_probabilities.subplots_adjust(hspace=0, wspace=0)
    sps_posterior_probabilities.set_ylabel(r'log probability per walker')
    sps_posterior_probabilities.set_xlabel(r'accepted sample number')
    posterior_probabilities_plot = [f_posterior_probabilities, sps_posterior_probabilities]

    bin_range = range(n_bins)
    f_chain_evolution = plt.figure(figsize=(5, 5 * n_bins))
    sps_chain_evolution = [f_chain_evolution.add_subplot(n_bins, 1, k+1) for k in bin_range]
    for k in bin_range:
        sps_chain_evolution[k].set_ylabel(r'parameter value '+str(k))
        sps_chain_evolution[k].set_xlabel(r'accepted sample number')
    f_chain_evolution.subplots_adjust(hspace=0, wspace=0)
    random_walkers = [np.random.randint(0, n_walkers) for i in range(d.plot_colors)]
    chain_evolution_plot = [f_chain_evolution, sps_chain_evolution, random_walkers]

    plot_information = (gelman_rubin_evolution_plot, autocorrelation_times_plot, acceptance_fractions_plot, posterior_probabilities_plot, chain_evolution_plot)
    return plot_information

def plot_sampler_progress(plot_information, sampler_output, full_chain, burn_ins, plot_dir):
    """
    Plots new information into burn-in progress plots

    Parameters
    ----------
    plot_information: tuple
        contains figure and subplot objects for Gelman-Rubin evolution,
        autocorrelation times, acceptance fractions, posterior probabilities,
        and chain evolution
    sampler_output: dict
        dictionary containing array of sampled redshift density function bin values as well as posterior probabilities, acceptance fractions, and autocorrelation times
    full_chain: ndarray, float
        array of all accepted samples so far
    burn_ins: int
        number of between-convergence-check intervals that have already been performed
    plot_dir: string
        location in which to store the plots

    Returns
    -------
    plot_information: tuple
        contains figure and subplot objects for Gelman-Rubin evolution,
        autocorrelation times, acceptance fractions, posterior probabilities,
        and chain evolution
    """
    (n_walkers, n_burn_test, n_bins) = np.shape(sampler_output['chains'])

    burn_test_range = range(n_burn_test)
    bin_range = range(n_bins)

    (gelman_rubin_evolution_plot, autocorrelation_times_plot, acceptance_fractions_plot, posterior_probabilities_plot, chain_evolution_plot) = plot_information

    [f_gelman_rubin_evolution, sps_gelman_rubin_evolution] = gelman_rubin_evolution_plot
    gelman_rubin = s.multi_parameter_gr_stat(full_chain)
    x_some = [(burn_ins + 1) * n_burn_test] * n_bins
    sps_gelman_rubin_evolution.scatter(x_some,
                           gelman_rubin,
                           c='k',
                           alpha=0.5,
                           linewidth=0.1,
                           s=2)
    sps_gelman_rubin_evolution.set_xlim(0, (burn_ins + 2) * n_burn_test)
    gelman_rubin_evolution_plot = [f_gelman_rubin_evolution, sps_gelman_rubin_evolution]
    f_gelman_rubin_evolution.savefig(os.path.join(plot_dir, 'gelman_rubin_evolution.png'), bbox_inches='tight', pad_inches = 0)

    [f_autocorrelation_times, sps_autocorrelation_times] = autocorrelation_times_plot
    autocorrelation_times = s.acors(full_chain)# sampler_output['acors']
    # default to bins mode for autocorrelation times, will need to fix this later
    # if something == 'bins':
    x_some = [(burn_ins + 1) * n_burn_test] * n_bins
    # if something == 'walkers':
    #     x_some = [(burn_ins + 1) * n_burn_test] * n_walkers
    sps_autocorrelation_times.scatter(x_some,
                           autocorrelation_times,
                           c='k',
                           alpha=0.5,
                           linewidth=0.1,
                           s=2)
    sps_autocorrelation_times.set_xlim(0, (burn_ins + 2) * n_burn_test)
    autocorrelation_times_plot = [f_autocorrelation_times, sps_autocorrelation_times]
    f_autocorrelation_times.savefig(os.path.join(plot_dir, 'autocorrelation_times.png'), bbox_inches='tight', pad_inches = 0)

    [f_acceptance_fractions, sps_acceptance_fractions] = acceptance_fractions_plot
    acceptance_fractions = sampler_output['fracs'].T
    sps_acceptance_fractions.scatter([(burn_ins + 1) * n_burn_test] * n_walkers,
                                   acceptance_fractions,
                                   c='k',
                                   alpha=0.5,
                                   linewidth=0.1,
                                   s=n_bins)
    sps_acceptance_fractions.set_xlim(0, (burn_ins + 2) * n_burn_test)
    acceptance_fractions_plot = [f_acceptance_fractions, sps_acceptance_fractions]
    f_acceptance_fractions.savefig(os.path.join(plot_dir, 'acceptance_fractions.png'), bbox_inches='tight', pad_inches = 0)

    [f_posterior_probabilities, sps_posterior_probabilities] = posterior_probabilities_plot
    posterior_probabilities = np.swapaxes(sampler_output['probs'], 0, 1)
    max_posterior_probabilities = np.max(posterior_probabilities)
    locs, scales = [], []
    for x in burn_test_range:
        loc, scale = sp.stats.norm.fit_loc_scale(posterior_probabilities[x])
        locs.append(loc)
        scales.append(scale)
    locs = np.array(locs)
    scales = np.array(scales)
    # (locs, scales) = s.norm_fit(posterior_probabilities[:])
    x_all = np.arange(burn_ins * n_burn_test, (burn_ins + 1) * n_burn_test + 1)
    pu.plot_step(sps_posterior_probabilities, x_all, locs)
    x_cor = [x_all[:-1], x_all[:-1], x_all[1:], x_all[1:]]
    y_cor = np.array([locs - scales, locs + scales, locs + scales, locs - scales])
    y_cor2 = np.array([locs - 2. * scales, locs + 2. * scales, locs + 2. * scales, locs - 2. * scales])
    sps_posterior_probabilities.fill(x_cor, y_cor, color='k', alpha=0.5, linewidth=0.)
    sps_posterior_probabilities.fill(x_cor, y_cor2, color='k', alpha=0.25, linewidth=0.)
    sps_posterior_probabilities.set_xlim(0, (burn_ins + 1) * n_burn_test)
    posterior_probabilities_plot = [f_posterior_probabilities, sps_posterior_probabilities]
    f_posterior_probabilities.savefig(os.path.join(plot_dir, 'posterior_probabilities.png'), bbox_inches='tight', pad_inches = 0)

    [f_chain_evolution, sps_chain_evolution, random_walkers] = chain_evolution_plot
    chains = sampler_output['chains']
    (n_walkers, n_burn_test, n_bins) = np.shape(chains)
    chains = np.swapaxes(chains, 1, 2)
    x_all = np.arange(burn_ins * n_burn_test, (burn_ins + 1) * n_burn_test)
    for k in bin_range:
        for i in range(d.plot_colors):
            sps_chain_evolution[k].plot(x_all, chains[random_walkers[i]][k], c=colors[i])
            sps_chain_evolution[k].set_xlim(0, (burn_ins + 1) * n_burn_test)
    chain_evolution_plot = [f_chain_evolution, sps_chain_evolution, random_walkers]
    f_chain_evolution.savefig(os.path.join(plot_dir, 'chain_evolution.png'), bbox_inches='tight', pad_inches = 0)

    plot_information = (gelman_rubin_evolution_plot, autocorrelation_times_plot, acceptance_fractions_plot, posterior_probabilities_plot, chain_evolution_plot)
    return plot_information

def plot_estimators(info, plot_dir):
    """
    Makes a log and linear plot of n(z) estimators from a log_z_dens object

    Parameters
    ----------
    info: dict
        dictionary of stored information from log_z_dens object
    plot_dir: string
        location into which the plot will be saved

    Returns
    -------
    f: matplotlib figure
        figure object
    """
    pu.set_up_plot()

    f = plt.figure(figsize=(5, 10))
    sps = [f.add_subplot(2, 1, l+1) for l in xrange(0, 2)]
    f.subplots_adjust(hspace=0, wspace=0)
    sps_log = sps[0]
    sps = sps[1]

    sps_log.set_xlim(info['bin_ends'][0], info['bin_ends'][-1])
    sps_log.set_ylabel(r'$\ln n(z)$')
    sps.set_xlim(info['bin_ends'][0], info['bin_ends'][-1])
    sps.set_xlabel(r'$z$')
    sps.set_ylabel(r'$n(z)$')
    sps.ticklabel_format(style='sci',axis='y')

    pu.plot_step(sps, info['bin_ends'], np.exp(info['log_interim_prior']), w=w_int, s=s_int, a=a_int, c=c_int, d=d_int, l=l_int+nz)
    pu.plot_step(sps_log, info['bin_ends'], info['log_interim_prior'], w=w_int, s=s_int, a=a_int, c=c_int, d=d_int, l=l_int+lnz)

    if info['truth'] is not None:
        sps.plot(info['truth']['z_grid'], info['truth']['nz_grid'], linewidth=w_tru, alpha=a_tru, color=c_tru, label=l_tru+nz)
        sps_log.plot(info['truth']['z_grid'], u.safe_log(info['truth']['nz_grid']), linewidth=w_tru, alpha=a_tru, color=c_tru, label=l_tru+lnz)

    if 'log_mean_sampled_nz' in info['estimators']:
        plot_samples(info, plot_dir)
        (locs, scales) = s.norm_fit(info['log_sampled_nz_meta_data']['chains'])
        for k in range(len(info['bin_ends'])-1):
            x_errs = [info['bin_ends'][k], info['bin_ends'][k], info['bin_ends'][k+1], info['bin_ends'][k+1]]
            log_y_errs = [locs[k] - scales[k], locs[k] + scales[k], locs[k] + scales[k], locs[k] - scales[k]]
            sps_log.fill(x_errs, log_y_errs, color='k', alpha=0.1, linewidth=0.)
            sps.fill(x_errs, np.exp(log_y_errs), color='k', alpha=0.1, linewidth=0.)
        pu.plot_step(sps, info['bin_ends'], np.exp(info['estimators']['log_mean_sampled_nz']), w=w_bfe, s=s_bfe, a=a_bfe, c=c_bfe, d=d_bfe, l=l_bfe+nz)
        pu.plot_step(sps_log, info['bin_ends'], info['estimators']['log_mean_sampled_nz'], w=w_bfe, s=s_bfe, a=a_bfe, c=c_bfe, d=d_bfe, l=l_bfe+lnz)

    if 'log_mmle_nz' in info['estimators']:
        pu.plot_step(sps, info['bin_ends'], np.exp(info['estimators']['log_mmle_nz']), w=w_mle, s=s_mle, a=a_mle, c=c_mle, d=d_mle, l=l_mle+nz)
        pu.plot_step(sps_log, info['bin_ends'], info['estimators']['log_mmle_nz'], w=w_mle, s=s_mle, a=a_mle, c=c_mle, d=d_mle, l=l_mle+lnz)

    if 'log_stacked_nz' in info['estimators']:
        pu.plot_step(sps, info['bin_ends'], np.exp(info['estimators']['log_stacked_nz']), w=w_stk, s=s_stk, a=a_stk, c=c_stk, d=d_stk, l=l_stk+nz)
        pu.plot_step(sps_log, info['bin_ends'], info['estimators']['log_stacked_nz'], w=w_stk, s=s_stk, a=a_stk, c=c_stk, d=d_stk, l=l_stk+lnz)

    if 'log_mmap_nz' in info['estimators']:
        pu.plot_step(sps, info['bin_ends'], np.exp(info['estimators']['log_mmap_nz']), w=w_map, s=s_map, a=a_map, c=c_map, d=d_map, l=l_map+nz)
        pu.plot_step(sps_log, info['bin_ends'], info['estimators']['log_mmap_nz'], w=w_map, s=s_map, a=a_map, c=c_map, d=d_map, l=l_map+lnz)

    if 'log_mexp_nz' in info['estimators']:
        pu.plot_step(sps, info['bin_ends'], np.exp(info['estimators']['log_mexp_nz']), w=w_exp, s=s_exp, a=a_exp, c=c_exp, d=d_exp, l=l_exp+nz)
        pu.plot_step(sps_log, info['bin_ends'], info['estimators']['log_mexp_nz'], w=w_exp, s=s_exp, a=a_exp, c=c_exp, d=d_exp, l=l_exp+lnz)

    sps_log.legend(fontsize='x-small', loc='lower left')
    sps.set_xlabel('x')
    sps_log.set_ylabel('Log probability density')
    sps.set_ylabel('Probability density')
    f.savefig(os.path.join(plot_dir, 'all_estimators.png'), bbox_inches='tight', pad_inches = 0)

    return

def plot_samples(info, plot_dir):
    """
    Plots a few random samples from the posterior distribution

    Parameters
    ----------
    info: dict
        dictionary of stored information from log_z_dens object
    """
    pu.set_up_plot()

    f = plt.figure(figsize=(5, 10))
    sps = [f.add_subplot(2, 1, l+1) for l in xrange(0, 2)]
    f.subplots_adjust(hspace=0, wspace=0)
    sps_log = sps[0]
    sps = sps[1]

    sps_log.set_xlim(info['bin_ends'][0], info['bin_ends'][-1])
    sps_log.set_ylabel(r'$\ln n(z)$')
    sps.set_xlim(info['bin_ends'][0], info['bin_ends'][-1])
    sps.set_xlabel(r'$z$')
    sps.set_ylabel(r'$n(z)$')
    sps.ticklabel_format(style='sci',axis='y')

    pu.plot_step(sps, info['bin_ends'], np.exp(info['log_interim_prior']), w=w_int, s=s_int, a=a_int, c=c_int, d=d_int, l=l_int+nz)
    pu.plot_step(sps_log, info['bin_ends'], info['log_interim_prior'], w=w_int, s=s_int, a=a_int, c=c_int, d=d_int, l=l_int+lnz)
    if info['truth'] is not None:
        sps.plot(info['truth']['z_grid'], info['truth']['nz_grid'], linewidth=w_tru, alpha=a_tru, color=c_tru, label=l_tru+nz)
        sps_log.plot(info['truth']['z_grid'], u.safe_log(info['truth']['nz_grid']), linewidth=w_tru, alpha=a_tru, color=c_tru, label=l_tru+lnz)

    (locs, scales) = s.norm_fit(info['log_sampled_nz_meta_data']['chains'])
    for k in range(len(info['bin_ends'])-1):
        x_errs = [info['bin_ends'][k], info['bin_ends'][k], info['bin_ends'][k+1], info['bin_ends'][k+1]]
        log_y_errs = [locs[k] - scales[k], locs[k] + scales[k], locs[k] + scales[k], locs[k] - scales[k]]
        sps_log.fill(x_errs, log_y_errs, color='k', alpha=0.1, linewidth=0.)
        sps.fill(x_errs, np.exp(log_y_errs), color='k', alpha=0.1, linewidth=0.)
    shape = np.shape(info['log_sampled_nz_meta_data']['chains'])
    flat = info['log_sampled_nz_meta_data']['chains'].reshape(np.prod(shape[:-1]), shape[-1])
    random_samples = [np.random.randint(0, len(flat)) for i in range(d.plot_colors)]
    for i in range(d.plot_colors):
        pu.plot_step(sps_log, info['bin_ends'], flat[random_samples[i]], s=s_smp, d=d_smp, w=w_smp, a=1., c=colors[i])
        pu.plot_step(sps, info['bin_ends'], np.exp(flat[random_samples[i]]), s=s_smp, d=d_smp, w=w_smp, a=1., c=colors[i])
    pu.plot_step(sps_log, info['bin_ends'], locs, s=s_smp, d=d_smp, w=2., a=1., c='k', l=l_bfe+lnz)
    pu.plot_step(sps, info['bin_ends'], np.exp(locs), s=s_smp, d=d_smp, w=2., a=1., c='k', l=l_bfe+nz)

    sps_log.legend(fontsize='x-small', loc='lower left')
    sps.set_xlabel('x')
    sps_log.set_ylabel('Log probability density')
    sps.set_ylabel('Probability density')
    f.savefig(os.path.join(plot_dir, 'samples.png'), bbox_inches='tight', pad_inches = 0)
    return
