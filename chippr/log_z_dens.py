import numpy as np
import scipy as sp
import scipy.optimize as op
import cPickle as cpkl
import emcee

import matplotlib as mpl
mpl.use('PS')
import matplotlib.pyplot as plt

import chippr
from chippr import plot_utils as pu
from chippr import utils as u
from chippr import stats as stats

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
        self.info = {}

        self.bin_ends = np.array(catalog['bin_ends'])
        self.bin_range = self.bin_ends[:-1]-self.bin_ends[0]
        self.bin_mids = (self.bin_ends[1:]+self.bin_ends[:-1])/2.
        self.bin_difs = self.bin_ends[1:]-self.bin_ends[:-1]
        self.n_bins = len(self.bin_mids)
        self.info['bin_ends'] = self.bin_ends

        self.log_int_pr = np.array(catalog['log_interim_prior'])
        self.int_pr = np.exp(self.log_int_pr)
        self.info['log_interim_prior'] = self.log_int_pr

        self.log_pdfs = np.array(catalog['log_interim_posteriors'])
        self.pdfs = np.exp(self.log_pdfs)
        self.n_pdfs = len(self.log_pdfs)
        self.info['log_interim_posteriors'] = self.log_pdfs

        if vb:
            print(str(len(self.bin_ends)-1)+' bins, '+str(len(self.log_pdfs))+' interim posterior PDFs')

        self.hyper_prior = hyperprior

        self.truth = truth

        self.stk_nz = None
        self.map_nz = None
        self.exp_nz = None
        self.mle_nz = None

        return

    def evaluate_log_hyper_likelihood(self, log_nz):
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
        log_hyper_likelihood = np.sum(u.safe_log(hyper_lfs))

        return log_hyper_likelihood

    def evaluate_log_hyper_prior(self, log_nz):
        """
        Function to evaluate log hyperprior

        Parameters
        ----------
        log_nz: numpy.ndarray, float
            vector of logged redshift density bin values at which to evaluate the hyperprior

        Returns
        -------
        log_hyper_prior: float
            log prior probability associated with parameters in log_nz
        """
        log_hyper_prior = -0.5 * np.dot(np.dot(self.hyper_prior.invvar, log_nz), log_nz)

        return log_hyper_prior

    def evaluate_log_hyper_posterior(self, log_nz):
        """
        Function to evaluate log hyperposterior

        Parameters
        ----------
        log_nz: numpy.ndarray, float
            vector of logged redshift density bin values at which to evaluate the full posterior

        Returns
        -------
        log_hyper_posterior: float
            log hyperposterior probability associated with parameters in log_nz
        """
        log_hyper_likelihood = self.evaluate_log_hyper_likelihood(log_nz)
        log_hyper_prior = self.evaluate_log_hyper_prior(log_nz)
        log_hyper_posterior = log_hyper_likelihood + log_hyper_prior
        return log_hyper_posterior

    def optimize(self, start, vb=True):
        """
        Maximizes the hyperposterior of the redshift density

        Parameters
        ----------
        start: numpy.ndarray
            array of log redshift density function bin values at which to begin optimization
        vb: boolean, optional
            True to print progress messages to stdout, False to suppress

        Returns
        -------
        res.x: numpy.ndarray
            array of logged redshift density function bin values maximizing hyperposterior
        """
        def _objective(log_nz):
            return -2. * self.evaluate_log_hyper_posterior(log_nz)

        if vb:
            print("starting at", start, _objective(start))

        res = op.minimize(_objective, start, method="Nelder-Mead", options={"maxfev": 1e5, "maxiter":1e5})

        if vb:
            print(res)
        return res.x

    def calculate_mmle(self, start, vb=True):
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
        log_mle_nz: numpy.ndarray
            array of logged redshift density function bin values maximizing hyperposterior
        """

        log_mle = self.optimize(start)
        mle_nz = np.exp(log_mle)
        self.mle_nz = mle_nz / np.dot(mle_nz, self.bin_difs)
        self.log_mle_nz = u.safe_log(self.mle_nz)
        self.info['log_mmle_nz'] = self.log_mle_nz

        return self.log_mle_nz

    def calculate_stacked(self, vb=True):
        """
        Calculates the stacked estimator of the redshift density function

        Parameters
        ----------
        vb: boolean, optional
            True to print progress messages to stdout, False to suppress

        Returns
        -------
        log_stk_nz: ndarray
            array of logged redshift density function bin values
        """
        self.stk_nz = np.sum(self.pdfs, axis=0)
        self.stk_nz /= np.dot(self.stk_nz, self.bin_difs)
        self.log_stk_nz = u.safe_log(self.stk_nz)
        self.info['log_stacked_nz'] = self.log_stk_nz

        return self.log_stk_nz

    def calculate_mmap(self, vb=True):
        """
        Calculates the marginalized maximum a posteriori estimator of the redshift density function

        Parameters
        ----------
        vb: boolean, optional
            True to print progress messages to stdout, False to suppress

        Returns
        -------
        log_map_nz: ndarray
            array of logged redshift density function bin values
        """
        self.map_nz = np.zeros(self.n_bins)
        mappreps = [np.argmax(l) for l in self.log_pdfs]
        for m in mappreps:
              self.map_nz[m] += 1.
        self.map_nz /= self.bin_difs[m] * self.n_pdfs
        self.log_map_nz = u.safe_log(self.map_nz)
        self.info['log_mmap_nz'] = self.log_map_nz

        return self.log_map_nz

    def calculate_mexp(self, vb=True):
        """
        Calculates the marginalized expected value estimator of the redshift density function

        Returns
        -------
        log_exp_nz: ndarray
            array of logged redshift density function bin values
        """
        expprep = [sum(z) for z in self.bin_mids * self.pdfs * self.bin_difs]
        self.exp_nz = np.zeros(self.n_bins)
        for z in expprep:
            for k in range(self.n_bins):
                if z > self.bin_ends[k] and z < self.bin_ends[k+1]:
                    self.exp_nz[k] += 1.
        self.exp_nz /= self.bin_difs * self.n_pdfs
        self.log_exp_nz = u.safe_log(self.exp_nz)
        self.info['log_mexp_nz'] = self.log_exp_nz

        return self.log_exp_nz

    def sample(self, ivals, n_samps, vb=True):
        """
        Samples the redshift density hyperposterior

        Parameters
        ----------
        n_samps: int
            number of samples to accept before stopping
        vb: boolean, optional
            True to print progress messages to stdout, False to suppress

        Returns
        -------
        samples: ndarray
            array of sampled redshift density function bin values
        """
        pos, prob, state = self.sampler.run_mcmc(ivals, n_samps)
        chains = self.sampler.chain
        return chains

    def calculate_samples(self, ivals, n_samps=None, vb=True):
        """
        Calculates samples estimating the redshift density function

        Parameters
        ----------
        n_samps: int, optional
            number of samples to accept before stopping
        ivals:
        vb: boolean, optional
            True to print progress messages to stdout, False to suppress

        Returns
        -------
        log_samples_nz: ndarray
            array of sampled log redshift density function bin values
        """
        self.n_walkers = len(ivals)
        self.sampler = emcee.EnsembleSampler(self.n_walkers, self.n_bins, self.evaluate_log_hyper_posterior)
        if n_samps is None:
            n_samps = self.n_pdfs
        self.log_samples_nz = self.sample(ivals, n_samps)
        self.samples_nz = np.exp(self.log_samples_nz)
        self.info['log_sampled_nz'] = self.log_samples_nz

        return self.log_samples_nz

    def plot(self, plot_loc=''):
        """
        Plots all available estimators of the redshift density function.

        Parameters
        ----------
        plot_loc: string
            destination where plot should be stored
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

        if self.stk_nz is not None:
            pu.plot_step(sps, self.bin_ends, self.stk_nz, w=pu.w_stk, s=pu.s_stk, a=pu.a_stk, c=pu.c_stk, d=pu.d_stk, l=pu.l_stk+pu.nz)
            pu.plot_step(sps_log, self.bin_ends, self.log_stk_nz, w=pu.w_stk, s=pu.s_stk, a=pu.a_stk, c=pu.c_stk, d=pu.d_stk, l=pu.l_stk+pu.lnz)

        if self.map_nz is not None:
            pu.plot_step(sps, self.bin_ends, self.map_nz, w=pu.w_map, s=pu.s_map, a=pu.a_map, c=pu.c_map, d=pu.d_map, l=pu.l_map+pu.nz)
            pu.plot_step(sps_log, self.bin_ends, self.log_map_nz, w=pu.w_map, s=pu.s_map, a=pu.a_map, c=pu.c_map, d=pu.d_map, l=pu.l_map+pu.lnz)

        if self.exp_nz is not None:
            pu.plot_step(sps, self.bin_ends, self.exp_nz, w=pu.w_exp, s=pu.s_exp, a=pu.a_exp, c=pu.c_exp, d=pu.d_exp, l=pu.l_exp+pu.nz)
            pu.plot_step(sps_log, self.bin_ends, self.log_exp_nz, w=pu.w_exp, s=pu.s_exp, a=pu.a_exp, c=pu.c_exp, d=pu.d_exp, l=pu.l_exp+pu.lnz)

        if self.mle_nz is not None:
            pu.plot_step(sps, self.bin_ends, self.mle_nz, w=pu.w_mle, s=pu.s_mle, a=pu.a_mle, c=pu.c_mle, d=pu.d_mle, l=pu.l_mle+pu.nz)
            pu.plot_step(sps_log, self.bin_ends, self.log_mle_nz, w=pu.w_mle, s=pu.s_mle, a=pu.a_mle, c=pu.c_mle, d=pu.d_mle, l=pu.l_mle+pu.lnz)

        if self.samples_nz is not None:
            self.log_bfe_nz = stats.mean(self.log_samples_nz)
            self.bfe_nz = np.exp(self.log_bfe_nz)
            pu.plot_step(sps, self.bin_ends, self.bfe_nz, w=pu.w_bfe, s=pu.s_bfe, a=pu.a_bfe, c=pu.c_bfe, d=pu.d_bfe, l=pu.l_bfe+pu.nz)
            pu.plot_step(sps_log, self.bin_ends, self.log_bfe_nz, w=pu.w_bfe, s=pu.s_bfe, a=pu.a_bfe, c=pu.c_bfe, d=pu.d_bfe, l=pu.l_bfe+pu.lnz)

        sps_log.legend(fontsize='x-small')
        sps.set_xlabel('x')
        sps_log.set_ylabel('Log probability density')
        sps.set_ylabel('Probability density')
        self.f.savefig(plot_loc+'plot.png')

    def read(self, loc, style='pickle'):
        """
        Function to load inferred quantities from files.

        Parameters
        ----------
        loc: string
            filepath where inferred redshift density function is stored
        style: string, optional
            keyword for file format

        Returns
        -------
        self: log_z_dens object
            returns the redshift density function object itself
        """
        return self

    def write(self, loc, style='pickle'):
        """
        Function to write results of inference to files.

        Parameters
        ----------
        loc: string
            filepath where results of inference should be saved.
        style: string, optional
            keyword for file format
        """
        with open(loc) as file_location:
            cpkl.dump(self.info, loc)
        return
