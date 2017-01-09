import numpy as np
import scipy as sp
import scipy.optimize as op

import matplotlib as mpl
mpl.use('PS')
import matplotlib.pyplot as plt

import chippr
from chippr import plot_utils as pu
from chippr import utils as u

class log_z_dens(object):

    def __init__(self, catalog, hyperprior, truth=None, vb=True):
        """
        An object representing the redshift density function (normalized redshift distribution function)

        Parameters
        ----------
        catalog: chippr.catalog object
            dict containing bin endpoints, interim prior bin values, and interim posterior PDF bin values
        hyperprior: chippr.mvn object
            multivariate Gaussian distribution for hyperprior distribution
        truth: chippr.gmix object, optional
            true redshift density function expressed as univariate Gaussian mixture
        vb: boolean, optional
            True to print progress messages to stdout, False to suppress
        """

        self.bin_ends = np.array(catalog['bin_ends'])
        self.bin_range = self.bin_ends[:-1]-self.bin_ends[0]
        self.bin_mids = (self.bin_ends[1:]+self.bin_ends[:-1])/2.
        self.bin_difs = self.bin_ends[1:]-self.bin_ends[:-1]
        self.n_bins = len(self.bin_mids)

        self.int_pr = np.array(catalog['interim_prior'])
        self.log_int_pr = u.safe_log(self.int_pr)

        if vb:
            print(np.dot(np.exp(self.log_int_pr), self.bin_difs))

        self.pdfs = np.array(catalog['interim_posteriors'])
        self.log_pdfs = u.safe_log(self.pdfs)
        self.n_pdfs = len(self.log_pdfs)

        if vb:
            print(str(len(self.bin_ends)-1)+' bins, '+str(len(self.log_pdfs))+' interim posterior PDFs')

        self.hyper_prior = hyperprior

        self.truth = truth

        self.stack_nz = None
        self.mmle_nz = None

        return

    def calc_log_hyper_lf(self, log_nz):
        """
        Function to evaluate log hyperlikelihood

        Parameters
        ----------
        log_nz: numpy.ndarray, float
            vector of logged redshift density bin values at which to evaluate the hyperlikelihood

        Returns
        -------
        log_hyper_lf: float
            log likelihood probability associated with parameters in log_nz
        """
        norm_nz = np.exp(log_nz - np.max(log_nz))
        norm_nz /= np.sum(norm_nz)#, self.bin_difs)
        hyper_lfs = np.sum(norm_nz[None,:] * self.pdfs / self.int_pr[None,:], axis=1)
        log_hyper_lf = np.sum(u.safe_log(hyper_lfs))

        return log_hyper_lf

    def calc_log_hyper_pr(self, log_nz):
        """
        Function to evaluate log hyperprior

        Parameters
        ----------
        log_nz: numpy.ndarray, float
            vector of logged redshift density bin values at which to evaluate the hyperprior

        Returns
        -------
        log_hyper_pr: float
            log prior probability associated with parameters in log_nz
        """
        log_hyper_pr = -0.5 * np.dot(np.dot(self.hyper_prior.invvar, log_nz), log_nz)

        return log_hyper_pr

    def calc_log_hyper_posterior(self, log_nz):
        """
        Function to evaluate log hyperposterior

        Parameters
        ----------
        log_nz: numpy.ndarray, float
            vector of logged redshift density bin values at which to evaluate the full posterior

        Returns
        -------
        log_prob: float
            log posterior probability associated with parameters in log_nz
        """
        log_hyper_lf = self.calc_log_hyper_lf(log_nz)
        log_hyper_pr = self.calc_log_hyper_pr(log_nz)
        log_hyper_post = log_hyper_lf + log_hyper_pr
        return log_hyper_post

    def optimize(self, start, vb=True):
        """
        Calculates the marginalized maximum likelihood estimator of the redshift density function

        Parameters
        ----------
        start: numpy.ndarray
            array of log redshift density function bin values at which to begin optimization
        vb: boolean, optional
            True to print progress messages to stdout, False to suppress

        Returns
        -------
        mmle_dens: numpy.ndarray
            array of redshift density function bin values
        """
        def _objective(log_nz):
            return -2. * self.calc_log_hyper_posterior(log_nz)

        if vb:
            print("starting at", start, _objective(start))

        mmle = op.minimize(_objective, start, method="Nelder-Mead", options={"maxfev": 1e5, "maxiter":1e5})

        # if vb:
        #     print(mmle)
        #     print(np.dot(np.exp(mmle.x), self.bin_difs))

        mmle_nz = np.exp(mmle.x)
        norm_mmle = mmle_nz / np.dot(mmle_nz, self.bin_difs)
        self.mmle_nz = u.safe_log(norm_mmle)

        # if vb:
        #     print(np.dot(np.exp(self.mmle_nz), self.bin_difs))

        return self.mmle_nz

    def stack(self, vb=True):
        """
        Calculates the stacked estimator of the redshift density function

        Parameters
        ----------
        vb: boolean, optional
            True to print progress messages to stdout, False to suppress

        Returns
        -------
        log_stack: ndarray
            array of logged redshift density function bin values
        """
        stack = np.sum(self.pdfs, axis=0)
        stack /= np.dot(stack, self.bin_difs)
        log_stack = u.safe_log(stack)
        self.stack_nz = log_stack

        # if vb:
        #     print(np.dot(np.exp(self.stack_nz), self.bin_difs))

        return self.stack_nz

    def mmap(self):
        """
        Calculates the marginalized maximum a posteriori estimator of the redshift density function

        Returns
        -------
        mmap_dens: ndarray
            array of redshift density function bin values
        """
        self.mmap_nz = np.zeros(self.n_bins)
        mmappreps = [np.argmax(l) for l in self.log_pdfs]
        for m in mmappreps:
              self.mmap_nz[m] += 1.
        self.mmap_nz /= self.bin_difs[m] * self.n_pdfs
        self.mmap_nz = u.safe_log(self.mmap_nz)
        return self.mmap_nz

    def mexp(self):
        """
        Calculates the marginalized expected value estimator of the redshift density function

        Returns
        -------
        mexp_dens: ndarray
            array of redshift density function bin values
        """

        return

    def sample(self, n_samps, vb=True):
        """
        Calculates samples estimating the redshift density function

        Parameters
        ----------
        n_samps: int
            number of samples to accept before stopping
        vb: boolean, optional
            True to print progress messages to stdout, False to suppress

        Returns
        -------
        samp_dens: ndarray
            array of sampled redshift density function bin values
        """

        return

    def plot(self, plot_loc=''):
        """
        Plots all available estimators of the redshift density function.
        """

        # set up for better looking plots
        title = 10
        label = 10
        mpl.rcParams['text.usetex'] = True
        mpl.rcParams['axes.titlesize'] = title
        mpl.rcParams['axes.labelsize'] = label
        mpl.rcParams['figure.subplot.left'] = 0.2
        mpl.rcParams['figure.subplot.right'] = 0.9
        mpl.rcParams['figure.subplot.bottom'] = 0.2
        mpl.rcParams['figure.subplot.top'] = 0.9
        mpl.rcParams['figure.subplot.wspace'] = 0.5
        mpl.rcParams['figure.subplot.hspace'] = 0.5

        self.f = plt.figure(figsize=(5, 10))
        self.sps = [self.f.add_subplot(2, 1, l+1) for l in xrange(0, 2)]
        self.f.subplots_adjust(hspace=0, wspace=0)
        sps_log = self.sps[0]
        sps = self.sps[1]

        sps_log.set_xlim(self.bin_ends[0], self.bin_ends[-1])
        sps_log.set_ylabel(r'$\ln n(z)$')
        sps.set_xlim(self.bin_ends[0], self.bin_ends[-1])
        sps.set_xlabel(r'$z$')
        sps.set_ylabel(r'$n(z)$')
        sps.ticklabel_format(style='sci',axis='y')

        pu.plot_step(sps, self.bin_ends, self.int_pr, w=pu.w_int, s=pu.s_int, a=pu.a_int, c=pu.c_int, d=pu.d_int, l=pu.l_int+pu.nz)
        pu.plot_step(sps_log, self.bin_ends, self.log_int_pr, w=pu.w_int, s=pu.s_int, a=pu.a_int, c=pu.c_int, d=pu.d_int, l=pu.l_int+pu.lnz)

        if self.truth is not None:
            z = np.linspace(self.bin_ends[0], self.bin_ends[-1], self.n_bins**2)
            fun = self.truth.evaluate(z)
            log_fun = u.safe_log(fun)
            pu.plot_step(sps, z, fun, w=pu.w_tru, s=pu.s_tru, a=pu.a_tru, c=pu.c_tru, d=pu.d_tru, l=pu.l_tru+pu.nz)
            pu.plot_step(sps_log, z, log_fun, w=pu.w_tru, s=pu.s_tru, a=pu.a_tru, c=pu.c_tru, d=pu.d_tru, l=pu.l_tru+pu.lnz)

        if self.mmap_nz is not None:
            pu.plot_step(sps, self.bin_ends, np.exp(self.mmap_nz), w=pu.w_map, s=pu.s_map, a=pu.a_map, c=pu.c_map, d=pu.d_map, l=pu.l_map+pu.nz)
            pu.plot_step(sps_log, self.bin_ends, self.mmap_nz, w=pu.w_map, s=pu.s_map, a=pu.a_map, c=pu.c_map, d=pu.d_map, l=pu.l_map+pu.lnz)

        if self.stack_nz is not None:
            pu.plot_step(sps, self.bin_ends, np.exp(self.stack_nz), w=pu.w_stk, s=pu.s_stk, a=pu.a_stk, c=pu.c_stk, d=pu.d_stk, l=pu.l_stk+pu.nz)
            pu.plot_step(sps_log, self.bin_ends, self.stack_nz, w=pu.w_stk, s=pu.s_stk, a=pu.a_stk, c=pu.c_stk, d=pu.d_stk, l=pu.l_stk+pu.lnz)

        if self.mmle_nz is not None:
            pu.plot_step(sps, self.bin_ends, np.exp(self.mmle_nz), w=pu.w_mle, s=pu.s_mle, a=pu.a_mle, c=pu.c_mle, d=pu.d_mle, l=pu.l_mle+pu.nz)
            pu.plot_step(sps_log, self.bin_ends, self.mmle_nz, w=pu.w_mle, s=pu.s_mle, a=pu.a_mle, c=pu.c_mle, d=pu.d_mle, l=pu.l_mle+pu.lnz)

        sps_log.legend()
        sps.set_xlabel('x')
        sps_log.set_ylabel('Log probability density')
        sps.set_ylabel('Probability density')
        self.f.savefig(plot_loc+'plot.png')
