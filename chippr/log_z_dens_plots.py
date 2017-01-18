import numpy as np

import matplotlib as mpl
mpl.use('PS')
import matplotlib.pyplot as plt

import chippr
from chippr import utils as u
from chippr import plot_utils as pu

lnz, nz = r'$\ln[n(z)]$', r'$n(z)$'

s_tru, w_tru, a_tru, c_tru, d_tru, l_tru = '--', 0.5, 1., 'k', [(0, (1, 1))], 'True '
s_int, w_int, a_int, c_int, d_int, l_int = '--', 0.5, 0.5, 'k', [(0, (1, 1))], 'Interim '
s_stk, w_stk, a_stk, c_stk, d_stk, l_stk = '--', 1.5, 1., 'k', [(0, (7.5, 7.5))], 'Stacked '
s_map, w_map, a_map, c_map, d_map, l_map = '--', 1., 1., 'k', [(0, (7.5, 7.5))], 'MMAP '
s_exp, w_exp, a_exp, c_exp, d_exp, l_exp = '--', 1., 1., 'k', [(0, (2.5, 2.5))], 'MExp '
s_mle, w_mle, a_mle, c_mle, d_mle, l_mle = '--', 2., 1., 'k', [(0, (2.5, 2.5))], 'MMLE '
# s_smp, w_smp, a_smp, c_smp, d_smp, l_smp = '--', 1., 1., 'k', [(0, (1, 1))], 'Sampled '
s_bfe, w_bfe, a_bfe, c_bfe, d_bfe, l_bfe = '--', 2., 1., 'k', [(0, (1, 1))], 'Mean of\n Samples '

def plot_estimators(info):
    """
    Makes a log and linear plot of n(z) estimators from a log_z_dens object

    Parameters
    ----------
    info: dict
        dictionary of stored information from log_z_dens object

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

    if 'log_stacked_nz' in info['estimators']:
        pu.plot_step(sps, info['bin_ends'], np.exp(info['estimators']['log_stacked_nz']), w=w_stk, s=s_stk, a=a_stk, c=c_stk, d=d_stk, l=l_stk+nz)
        pu.plot_step(sps_log, info['bin_ends'], info['estimators']['log_stacked_nz'], w=w_stk, s=s_stk, a=a_stk, c=c_stk, d=d_stk, l=l_stk+lnz)

    if 'log_mmap_nz' in info['estimators']:
        pu.plot_step(sps, info['bin_ends'], np.exp(info['estimators']['log_mmap_nz']), w=w_map, s=s_map, a=a_map, c=c_map, d=d_map, l=l_map+nz)
        pu.plot_step(sps_log, info['bin_ends'], info['estimators']['log_mmap_nz'], w=w_map, s=s_map, a=a_map, c=c_map, d=d_map, l=l_map+lnz)

    if 'log_mexp_nz' in info['estimators']:
        pu.plot_step(sps, info['bin_ends'], np.exp(info['estimators']['log_mexp_nz']), w=w_exp, s=s_exp, a=a_exp, c=c_exp, d=d_exp, l=l_exp+nz)
        pu.plot_step(sps_log, info['bin_ends'], info['estimators']['log_mexp_nz'], w=w_exp, s=s_exp, a=a_exp, c=c_exp, d=d_exp, l=l_exp+lnz)

    if 'log_mmle_nz' in info['estimators']:
        pu.plot_step(sps, info['bin_ends'], np.exp(info['estimators']['log_mmle_nz']), w=w_mle, s=s_mle, a=a_mle, c=c_mle, d=d_mle, l=l_mle+nz)
        pu.plot_step(sps_log, info['bin_ends'], info['estimators']['log_mmle_nz'], w=w_mle, s=s_mle, a=a_mle, c=c_mle, d=d_mle, l=l_mle+lnz)

    if 'log_mean_sampled_nz' in info['estimators']:
        pu.plot_step(sps, info['bin_ends'], np.exp(info['estimators']['log_mean_sampled_nz']), w=w_bfe, s=s_bfe, a=a_bfe, c=c_bfe, d=d_bfe, l=l_bfe+nz)
        pu.plot_step(sps_log, info['bin_ends'], info['estimators']['log_mean_sampled_nz'], w=w_bfe, s=s_bfe, a=a_bfe, c=c_bfe, d=d_bfe, l=l_bfe+lnz)

    sps_log.legend(fontsize='x-small', loc='lower left')
    sps.set_xlabel('x')
    sps_log.set_ylabel('Log probability density')
    sps.set_ylabel('Probability density')

    return f
