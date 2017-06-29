import numpy as np
import csv
import timeit
import os

import matplotlib as mpl
mpl.use('PS')
import matplotlib.pyplot as plt

import chippr
from chippr import defaults as d
from chippr import utils as u
from chippr import sim_utils as su
from chippr import gauss
from chippr import discrete
from chippr import catalog_plots as plots

class catalog(object):

    def __init__(self, params={}, vb=True, loc='.'):
        """
        Object containing catalog of photo-z interim posteriors

        Parameters
        ----------
        params: dict or string, optional
            dictionary containing parameter values for catalog creation or
            string containing location of parameter file
        vb: boolean, optional
            True to print progress messages to stdout, False to suppress
        loc: string, optional
            directory into which to save data and plots made along the way
        """
        if type(params) == str:
            self.params = u.ingest(params)
        else:
            self.params = params

        self.params = d.check_sim_params(self.params)

        if vb:
            print self.params

        self.cat = {}

        self.dir = loc
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        self.plot_dir = os.path.join(loc, 'plots')
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)
        self.data_dir = os.path.join(loc, 'data')
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def proc_bins(self, vb=True):
        """
        Function to process binning

        Parameters
        ----------
        vb: boolean, optional
            True to print progress messages to stdout, False to suppress
        """
        self.n_coarse = self.params['n_bins']
        self.z_min = self.params['bin_min']
        self.z_max = self.params['bin_max']
        self.n_fine = self.n_coarse
        self.n_tot = self.n_coarse * self.n_fine
        z_range = self.z_max - self.z_min

        self.dz_coarse = z_range / self.n_coarse
        self.dz_fine = z_range / self.n_tot

        self.z_coarse = np.arange(self.z_min + 0.5 * self.dz_coarse, self.z_max, self.dz_coarse)
        self.z_fine = np.arange(self.z_min + 0.5 * self.dz_fine, self.z_max, self.dz_fine)

        self.bin_ends = np.arange(self.z_min, self.z_max + self.dz_coarse, self.dz_coarse)

        return

    def coarsify(self, fine):
        """
        Function to bin function evaluated on fine grid

        Parameters
        ----------
        fine: numpy.ndarray, float
            vector of values of function on fine grid

        Returns
        -------
        coarse: numpy.ndarray, float
            vector of binned values of function
        """
        coarse = fine / (np.sum(fine) * self.dz_fine)
        coarse = np.array([np.sum(coarse[k * self.n_fine : (k+1) * self.n_fine]) * self.dz_fine for k in range(self.n_coarse)])
        coarse /= np.sum(coarse * self.dz_coarse)

        return coarse

    def create(self, truth, int_pr, N=d.n_gals, vb=True):
        """
        Function creating a catalog of interim posterior probability
        distributions, will split this up into helper functions

        Parameters
        ----------
        truth: chippr.gmix object or chippr.gauss object or chippr.discrete
        object
            true redshift distribution object
        int_pr: chippr.gmix object or chippr.gauss object or chippr.discrete
        object
            interim prior distribution object
        vb: boolean, optional
            True to print progress messages to stdout, False to suppress

        Returns
        -------
        self.cat: dict
            dictionary comprising catalog information
        """
        self.truth = truth
        # self.true_samps = self.truth.sample(10**N)
        # if vb:
        #     plots.plot_true_histogram(self.true_samps, plot_loc=self.plot_dir)
        # self.n_items = len(self.true_samps)
        # self.samp_range = range(self.n_items)

        self.proc_bins()
        # self.obs_samps = self.sample_obs()

        self.prob_space = self.make_probs()
        if vb:
            plots.plot_prob_space(self.z_fine, self.prob_space, plot_loc=self.plot_dir)

        ## this far!
        ## next, sample discrete to get z_true, z_obs
        ## then literally take slices!

        self.obs_lfs = self.evaluate_lfs()
        for n in self.samp_range:
            self.obs_lfs[n] /= np.sum(self.obs_lfs[n]) * self.dz_fine

        # if vb:
        #     plots.plot_scatter(self.true_samps, self.obs_samps, self.obs_lfs, self.z_fine, plot_loc=self.plot_dir)
        #

        self.int_pr = int_pr
        int_pr_fine = self.int_pr.evaluate(self.z_fine)
        int_pr_coarse = self.coarsify(int_pr_fine)

        # rewrite to take advantage of numpy array manipulation
        pfs_fine = np.zeros((self.n_items, self.n_tot))
        for n in self.samp_range:
            pfs_fine[n] += int_pr_fine * self.obs_lfs[n]
            pfs_fine[n] /= np.sum(pfs_fine[n]) * self.dz_fine

        if vb:
            plots.plot_obs_scatter(self.true_samps, pfs_fine, self.z_fine, plot_loc=self.plot_dir)

        # rewrite to take advantage of numpy array manipulation
        pfs_coarse = np.zeros((self.n_items, self.n_coarse))
        for n in self.samp_range:
            pfs_coarse[n] += self.coarsify(pfs_fine[n])
            pfs_coarse[n] /= np.sum(pfs_coarse[n]) * self.dz_coarse

        self.cat['bin_ends'] = self.bin_ends
        self.cat['log_interim_prior'] = u.safe_log(int_pr_coarse)
        self.cat['log_interim_posteriors'] = u.safe_log(pfs_coarse)

        return self.cat

    def make_probs(self, vb=True):
        """
        Makes the continuous 2D probability distribution over z_spec, z_phot

        Parameters
        ----------
        vb: boolean
            print progress to stdout?

        Returns
        -------

        Notes
        -----
        Does not currently support variable sigmas, only one outlier population at a time
        """
        p_space = np.zeros((self.n_tot, self.n_tot))

        if not self.params['variable_sigmas']:
            identity = [gauss(self.z_fine[kk], self.params['constant_sigma']**2) for kk in range(self.n_tot)]
            p_space += np.array([identity[kk].evaluate(self.z_fine) for kk in range(self.n_tot)])

        if self.params['catastrophic_outliers'] != 0:
            self.outlier_lf = gauss(self.params['outlier_mean'], self.params['outlier_sigma']**2)
            self.uniform_lf = discrete(np.array([self.z_min, self.z_max]), np.array([1.]))

        if self.params['catastrophic_outliers'] == 'template':
            p_space += self.params['outlier_fraction'] * self.outlier_lf.evaluate(self.z_fine)[np.newaxis, :]

        elif self.params['catastrophic_outliers'] == 'training':
            p_space += self.params['outlier_fraction'] * self.outlier_lf.evaluate(self.z_fine)[:, np.newaxis]

        p_space *= self.truth.evaluate(self.z_fine)[:, np.newaxis]

        return p_space

    def sample_obs(self, vb=True):
        """
        Samples observed values from true values

        Parameters
        ----------
        vb: boolean
            print progress to stdout?

        Returns
        -------
        obs_samps: numpy.ndarray, float
            "observed" values

        Notes
        -----
        Currently, template-like outlier populations and training-like outlier
        populations cannot both be enabled in the same dataset.  Also, only one
        population of a type is permitted at this time.
        """
        if not self.params['variable_sigmas']:
            self.true_lfs = [gauss(self.true_samps[n], self.params['constant_sigma']**2) for n in self.samp_range]
            self.sigmas = np.array([self.params['constant_sigma']] * self.n_items)
        else:
            sigma_fun = gauss(self.params['constant_sigma'], self.params['constant_sigma'])
            self.sigmas = sigma_fun.sample(self.n_items)
            self.true_lfs = [gauss(self.true_samps[n], self.sigmas[n]**2) for n in self.samp_range]

        obs_samps = np.zeros(self.n_items)
        for n in self.samp_range:
            obs_samps[n] = self.true_lfs[n].sample_one()
        # self.limit = None

        if self.params['catastrophic_outliers'] != 0:
            # will add in functionality for multiple outlier populations soon!
            self.outlier_lf = gauss(self.params['outlier_mean'], self.params['outlier_sigma']**2)
            self.uniform_lf = discrete(np.array([self.z_min, self.z_max]), np.array([1.]))
            self.limits = np.zeros(self.n_items)

        if self.params['catastrophic_outliers'] == 'template':
            factors = self.uniform_lf.evaluate(self.true_samps)
            self.limits = self.params['outlier_fraction'] * factors
            for n in self.samp_range:
                if np.random.uniform() < self.limits[n]:
                    obs_samps[n] = self.outlier_lf.sample_one()
            # self.limit = np.mean(self.limits)

        elif self.params['catastrophic_outliers'] == 'training':
            factors = self.outlier_lf.evaluate(self.true_samps)
            self.limits = self.params['outlier_fraction'] * factors
            for n in self.samp_range:
                if np.random.uniform() < self.limits[n]:
                    obs_samps[n] = self.uniform_lf.sample_one()
            # self.limit = np.mean(self.limits)

        if vb:
            print('photo-zs calculated ')#with limit='+str(self.limit))
        return obs_samps

    def evaluate_lfs(self, vb=True):
        """
        Evaluates likelihoods based on observed sample values

        Parameters
        ----------
        vb: boolean
            print progress to stdout?

        Returns
        -------
        obs_lfs.T: numpy.ndarray, float
            array of likelihood values for each item as a function of fine
            binning
        """
        if not self.params['variable_sigmas']:
            lfs_fine = [gauss(self.z_fine[kk], self.sigmas**2) for kk in range(self.n_tot)]
            obs_lfs = np.array([lfs_fine[kk].evaluate(self.obs_samps) for kk in range(self.n_tot)])
            obs_lfs = obs_lfs.T
        else:
            lfs_fine = [[gauss(self.z_fine[kk], self.sigmas[n]**2) for kk in range(self.n_tot)] for n in self.samp_range]
            obs_lfs = np.array([[lfs_fine[n][kk].evaluate(self.obs_samps[n]) for kk in range(self.n_tot)] for n in self.samp_range])

        if type(self.params['catastrophic_outliers']) == str:
            if self.params['catastrophic_outliers'] == 'template':
                # factors = self.outlier_lf.evaluate(self.true_samps)
                # self.limits = self.params['outlier_fraction'] * factors
                for n in self.samp_range:
                    obs_lfs[n] = (1.-self.limits[n]) * obs_lfs[n]
                    obs_lfs[n] += self.limits[n] * self.uniform_lf.evaluate(self.z_fine)
            elif self.params['catastrophic_outliers'] == 'training':
                # factors = self.uniform_lf.evaluate(self.true_samps)
                # self.limits = self.params['outlier_fraction'] * factors
                for n in self.samp_range:
                    obs_lfs[n] = (1.-self.limits[n]) * obs_lfs[n]
                    obs_lfs[n] += self.limits[n] * self.outlier_lf.evaluate(self.z_fine)

        if vb:
            print('photo-z posteriors calculated')

        return obs_lfs

    def write(self, loc='data', style='.txt'):
        """
        Function to write newly-created catalog to file

        Parameters
        ----------
        loc: string, optional
            file name into which to save catalog
        style: string, optional
            file format in which to save the catalog
        """
        if style == '.txt':
            with open(os.path.join(self.data_dir, loc + style), 'wb') as csvfile:
                out = csv.writer(csvfile, delimiter=' ')
                out.writerow(self.cat['bin_ends'])
                out.writerow(self.cat['log_interim_prior'])
                for line in self.cat['log_interim_posteriors']:
                    out.writerow(line)
        return

    def read(self, loc='data', style='.txt'):
        """
        Function to read in catalog file

        Parameters
        ----------
        loc: string, optional
            location of catalog file
        """
        if style == '.txt':
            with open(os.path.join(self.data_dir, loc + style), 'rb') as csvfile:
                tuples = (line.split(None) for line in csvfile)
                alldata = [[float(pair[k]) for k in range(0,len(pair))] for pair in tuples]
        self.cat['bin_ends'] = np.array(alldata[0])
        self.cat['log_interim_prior'] = np.array(alldata[1])
        self.cat['log_interim_posteriors'] = np.array(alldata[2:])
        return self.cat
