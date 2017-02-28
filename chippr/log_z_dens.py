import numpy as np
import scipy as sp
import os
import scipy.optimize as op
import cPickle as cpkl
import emcee

import matplotlib as mpl
mpl.use('PS')
import matplotlib.pyplot as plt

import chippr
from chippr import defaults as d
from chippr import plot_utils as pu
from chippr import utils as u
from chippr import stat_utils as s
from chippr import log_z_dens_plots as plots

class log_z_dens(object):

    def __init__(self, catalog, hyperprior, truth=None, loc='', vb=True):
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
        loc: string, optional
            directory into which to save results and plots made along the way
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
            print(str(self.n_bins)+' bins, '+str(len(self.log_pdfs))+' interim posterior PDFs')

        self.hyper_prior = hyperprior

        self.truth = truth
        self.info['truth'] = None
        if self.truth is not None:
            self.info['truth'] = {}
            self.tru_nz = np.zeros(self.n_bins)
            self.fine_zs = []
            self.fine_nz = []
            for b in range(self.n_bins):
                fine_z = np.linspace(self.bin_ends[b], self.bin_ends[b+1], self.n_bins)
                self.fine_zs.extend(fine_z)
                fine_dz = (self.bin_ends[b+1] - self.bin_ends[b]) / self.n_bins
                fine_n = self.truth.evaluate(fine_z)
                self.fine_nz.extend(fine_n)
                coarse_nz = np.sum(fine_n) * fine_dz
                self.tru_nz[b] += coarse_nz
            self.log_tru_nz = u.safe_log(self.tru_nz)
            self.info['log_tru_nz'] = self.log_tru_nz
            self.info['truth']['z_grid'] = np.array(self.fine_zs)
            self.info['truth']['nz_grid'] = np.array(self.fine_nz)

        self.info['estimators'] = {}
        self.info['stats'] = {}

        self.dir = loc
        self.data_dir = os.path.join(loc, 'data')
        self.plot_dir = os.path.join(loc, 'plots')
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)
        self.res_dir = os.path.join(loc, 'results')
        if not os.path.exists(self.res_dir):
            os.makedirs(self.res_dir)

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
        norm_nz = np.exp(log_nz)# - np.max(log_nz))
        #norm_nz /= np.sum(norm_nz)
        hyper_lfs = np.sum(norm_nz[None,:] * self.pdfs / self.int_pr[None,:] * self.bin_difs, axis=1)
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
        log_hyper_prior -= np.dot(np.exp(log_nz), self.bin_difs)

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
        if 'log_mmle_nz' not in self.info['estimators']:
            log_mle = self.optimize(start)
            mle_nz = np.exp(log_mle)
            self.mle_nz = mle_nz# / np.dot(mle_nz, self.bin_difs)
            self.log_mle_nz = log_mle#u.safe_log(self.mle_nz)
            self.info['estimators']['log_mmle_nz'] = self.log_mle_nz
        else:
            self.log_mle_nz = self.info['estimators']['log_mmle_nz']
            self.mle_nz = np.exp(self.log_mle_nz)

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
        if 'log_stacked_nz' not in self.info['estimators']:
            self.stk_nz = np.sum(self.pdfs, axis=0)
            self.stk_nz /= np.dot(self.stk_nz, self.bin_difs)
            self.log_stk_nz = u.safe_log(self.stk_nz)
            self.info['estimators']['log_stacked_nz'] = self.log_stk_nz
        else:
            self.log_stk_nz = self.info['estimators']['log_stacked_nz']
            self.stk_nz = np.exp(self.log_stk_nz)

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
        if 'log_mmap_nz' not in self.info['estimators']:
            self.map_nz = np.zeros(self.n_bins)
            mappreps = [np.argmax(l) for l in self.log_pdfs]
            for m in mappreps:
                self.map_nz[m] += 1.
            self.map_nz /= self.bin_difs[m] * self.n_pdfs
            self.log_map_nz = u.safe_log(self.map_nz)
            self.info['estimators']['log_mmap_nz'] = self.log_map_nz
        else:
            self.log_map_nz = self.info['estimators']['log_mmap_nz']
            self.map_nz = np.exp(self.log_map_nz)

        return self.log_map_nz

    def calculate_mexp(self, vb=True):
        """
        Calculates the marginalized expected value estimator of the redshift density function

        Parameters
        ----------
        vb: boolean, optional
            True to print progress messages to stdout, False to suppress

        Returns
        -------
        log_exp_nz: ndarray
            array of logged redshift density function bin values
        """
        if 'log_mexp_nz' not in self.info['estimators']:
            expprep = [sum(z) for z in self.bin_mids * self.pdfs * self.bin_difs]
            self.exp_nz = np.zeros(self.n_bins)
            for z in expprep:
                for k in range(self.n_bins):
                    if z > self.bin_ends[k] and z < self.bin_ends[k+1]:
                        self.exp_nz[k] += 1.
            self.exp_nz /= self.bin_difs * self.n_pdfs
            self.log_exp_nz = u.safe_log(self.exp_nz)
            self.info['estimators']['log_mexp_nz'] = self.log_exp_nz
        else:
            self.log_exp_nz = self.info['estimators']['log_mexp_nz']
            self.exp_nz = np.exp(self.log_exp_nz)

        return self.log_exp_nz

    def sample(self, ivals, n_samps, vb=True):
        """
        Samples the redshift density hyperposterior

        Parameters
        ----------
        ivals: numpy.ndarray, float
            initial values of the walkers
        n_samps: int
            number of samples to accept before stopping
        vb: boolean, optional
            True to print progress messages to stdout, False to suppress

        Returns
        -------
        mcmc_outputs: dict
            dictionary containing array of sampled redshift density function bin values as well as posterior probabilities, acceptance fractions, and autocorrelation times
        """
        self.sampler.reset()
        pos, prob, state = self.sampler.run_mcmc(ivals, n_samps)
        chains = self.sampler.chain
        probs = self.sampler.lnprobability
        fracs = self.sampler.acceptance_fraction
        acors = s.acors(chains)
        mcmc_outputs = {}
        mcmc_outputs['chains'] = chains
        mcmc_outputs['probs'] = probs
        mcmc_outputs['fracs'] = fracs
        mcmc_outputs['acors'] = acors
        return mcmc_outputs

    def calculate_samples(self, ivals, n_accepted=d.n_accepted, n_burn_test=d.n_burned, vb=True):
        """
        Calculates samples estimating the redshift density function

        Parameters
        ----------
        n_accepted: int, optional
            number of samples to accept per walker
        n_intermediate: int, optional
            duration of interval between burn-in tests
        ivals: numpy.ndarray, float
            initial values of log n(z) for each walker
        vb: boolean, optional
            True to print progress messages to stdout, False to suppress

        Returns
        -------
        log_samples_nz: ndarray
            array of sampled log redshift density function bin values
        """
        if 'log_mean_sampled_nz' not in self.info['estimators']:
            self.n_walkers = len(ivals)
            self.sampler = emcee.EnsembleSampler(self.n_walkers, self.n_bins, self.evaluate_log_hyper_posterior)
            self.burn_ins = 0
            self.burning_in = True
            vals = ivals
            if vb:
                canvas = plots.set_up_burn_in_plots(self.n_bins, self.n_walkers)
            full_chain = np.array([[ivals[w]] for w in range(self.n_walkers)])
            while self.burning_in:
                if vb:
                    print('beginning sampling '+str(self.burn_ins))
                burn_in_mcmc_outputs = self.sample(vals, n_burn_test)
                with open(os.path.join(self.res_dir, 'mcmc'+str(self.burn_ins)+'.p'), 'wb') as file_location:
                    cpkl.dump(burn_in_mcmc_outputs, file_location)
                full_chain = np.concatenate((full_chain, burn_in_mcmc_outputs['chains']), axis=1)
                if vb:
                    canvas = plots.plot_sampler_progress(canvas, burn_in_mcmc_outputs, full_chain, self.burn_ins, self.plot_dir)
                self.burning_in = s.gr_test(full_chain)
                vals = np.array([item[-1] for item in burn_in_mcmc_outputs['chains']])
                self.burn_ins += 1

            mcmc_outputs = self.sample(vals, n_accepted)
            full_chain = np.concatenate((full_chain, mcmc_outputs['chains']), axis=1)
            with open(os.path.join(self.res_dir, 'full_chain.p'), 'wb') as file_location:
                cpkl.dump(full_chain, file_location)

            self.log_smp_nz = mcmc_outputs['chains']
            self.smp_nz = np.exp(self.log_smp_nz)
            #self.info['estimators']['log_sampled_nz'] = self.log_smp_nz
            self.info['log_sampled_nz_meta_data'] = mcmc_outputs
            self.log_bfe_nz = s.norm_fit(self.log_smp_nz)[0]# s.mean(self.log_smp_nz)
            self.bfe_nz = np.exp(self.log_bfe_nz)
            self.info['estimators']['log_mean_sampled_nz'] = self.log_bfe_nz
        else:
            self.log_smp_nz = self.info['log_sampled_nz_meta_data']
            self.smp_nz = np.exp(self.log_smp_nz)
            self.log_bfe_nz = self.info['estimators']['log_mean_sampled_nz']
            self.bfe_nz = np.exp(self.log_smp_nz)

        if vb:
            plots.plot_samples(self.info, self.plot_dir)

        return self.log_smp_nz

    def compare(self, vb=True):
        """
        Calculates all available goodness of fit measures

        Parameters
        ----------
        vb: boolean, optional
            True to print progress messages to stdout, False to suppress

        Returns
        -------
        info['stats']: dict
            dictionary of all available statistics
        """
        self.info['stats']['kld'] = {}
        if self.truth is not None:
            for key in self.info['estimators']:
                self.info['stats']['kld'][key] = s.calculate_kld(self.tru_nz, self.info['estimators'][key])

        self.info['stats']['rms'], self.info['stats']['log_rms'] = {}, {}
        for key_1 in self.info['estimators']:
            for key_2 in self.info['estimators']:
                self.info['stats']['log_rms'][key_1 + '__' + key_2] = s.calculate_rms(self.info['estimators'][key_1], self.info['estimators'][key_2])
                self.info['stats']['rms'][key_1[4:] + '__' + key_2[4:]] = s.calculate_rms(np.exp(self.info['estimators'][key_1]), np.exp(self.info['estimators'][key_2]))

        if vb:
            print(self.info['stats'])
        return self.info['stats']

    def plot_estimators(self):
        """
        Plots all available estimators of the redshift density function.
        """
        plots.plot_estimators(self.info, self.plot_dir)
        return

    def read(self, read_loc, style='pickle', vb=True):
        """
        Function to load inferred quantities from files.

        Parameters
        ----------
        read_loc: string
            filepath where inferred redshift density function is stored
        style: string, optional
            keyword for file format, currently only 'pickle' supported
        vb: boolean, optional
            True to print progress messages to stdout, False to suppress

        Returns
        -------
        self.info: dict
            returns the log_z_dens information dictionary object
        """
        with open(os.path.join(self.res_dir, read_loc), 'rb') as file_location:
            self.info = cpkl.load(file_location)
        if vb:
            print('The following quantities were read from '+read_loc+' in the '+style+' format:')
            for key in self.info:
                print(key)
        return self.info

    def write(self, write_loc, style='pickle', vb=True):
        """
        Function to write results of inference to files.

        Parameters
        ----------
        write_loc: string
            filepath where results of inference should be saved.
        style: string, optional
            keyword for file format, currently only 'pickle' supported
        vb: boolean, optional
            True to print progress messages to stdout, False to suppress
        """
        with open(os.path.join(self.res_dir, write_loc), 'wb') as file_location:
            cpkl.dump(self.info, file_location)
        if vb:
            print('The following quantities were written to '+write_loc+' in the '+style+' format:')
            for key in self.info:
                print(key)
        return
