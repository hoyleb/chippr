import numpy as np
import csv
import timeit

import matplotlib as mpl
mpl.use('PS')
import matplotlib.pyplot as plt

import chippr
from chippr import defaults as d
from chippr import utils as u
from chippr import sim_utils as su
from chippr import gauss
from chippr import catalog_plots as plots

class catalog(object):

    def __init__(self, params={}, vb=True):
        """
        Object containing catalog of photo-z interim posteriors

        Parameters
        ----------
        params: dict or string, optional
            dictionary containing parameter values for catalog creation or string containing location of parameter file
        vb: boolean, optional
            True to print progress messages to stdout, False to suppress
        """
        if type(params) == str:
            self.params = su.ingest(params)
        else:
            self.params = params

        self.params = d.check_sim_params(self.params)

        if vb:
            print self.params

        self.cat = {}

    def proc_bins(self, bins, limits=(d.min_x, d.max_x), vb=True):
        """
        Function to process binning

        Parameters
        ----------
        bins: int
            number of evenly spaced bins
        limits: tuple or list or numpy.ndarray, float, optional
            endpoints over which binning will be defined
        vb: boolean, optional
            True to print progress messages to stdout, False to suppress
        """
        if type(bins) == int:
            self.n_coarse = bins
        else:
            self.n_coarse = 10
        x_min = limits[0]#np.min(self.obs_samps)
        x_max = limits[-1]#np.max(self.obs_samps)
        self.n_fine = self.n_coarse
        self.n_tot = self.n_coarse * self.n_fine
        x_range = x_max-x_min

        self.dx_coarse = x_range / self.n_coarse
        self.dx_fine = x_range / self.n_tot

        self.x_coarse = np.arange(x_min+0.5*self.dx_coarse, x_max, self.dx_coarse)
        self.x_fine = np.arange(x_min+0.5*self.dx_fine, x_max, self.dx_fine)

        self.bin_ends = np.arange(x_min, x_max+self.dx_coarse, self.dx_coarse)

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
        coarse = fine / (np.sum(fine) * self.dx_fine)
        coarse = np.array([np.sum(coarse[k * self.n_fine : (k+1) * self.n_fine]) * self.dx_fine for k in range(self.n_coarse)])
        coarse /= self.dx_coarse

        return coarse

    def create(self, truth, int_pr, vb=True):
        """
        Function creating a catalog of interim posterior probability distributions, will split this up into helper functions

        Parameters
        ----------
        truth: numpy.ndarray, float
            vector of true redshifts
        int_pr: chippr.gmix object or chippr.gauss object or chippr.discrete object
            interim prior distribution object
        bins: int, optional
            number of evenly spaced bins
        vb: boolean, optional
            True to print progress messages to stdout, False to suppress

        Returns
        -------
        self.cat: dict
            dictionary comprising catalog information
        """
        self.true_samps = truth
        if vb:
            plots.plot_true_histogram(self.true_samps)
        self.n_items = len(self.true_samps)
        self.samp_range = range(self.n_items)

        self.obs_samps = self.sample_obs()
        if vb:
            plots.plot_obs_scatter(self.true_samps, self.obs_samps)

        self.int_pr = int_pr
        self.proc_bins(self.params['n_bins'], limits=(self.int_pr.bin_ends[0], self.int_pr.bin_ends[-1]))

        self.obs_lfs = self.evaluate_lfs()

        int_pr_fine = int_pr.evaluate(self.x_fine)
        int_pr_coarse = self.coarsify(int_pr_fine)

        pfs = np.zeros((self.n_items, self.n_coarse))
        for n in self.samp_range:
            pf = int_pr_fine * self.obs_lfs[n]
            pf = self.coarsify(pf)
            pfs[n] += pf

        self.cat['bin_ends'] = self.bin_ends
        self.cat['log_interim_prior'] = u.safe_log(int_pr_coarse)
        self.cat['log_interim_posteriors'] = u.safe_log(pfs)

        return self.cat

    def sample_obs(self):
        """
        Samples observed values from true values

        Returns
        -------
        obs_samps: numpy.ndarray, float
            observed values
        """
        if not self.params['variable_sigma']:
            true_lfs = [gauss(self.true_samps[n], self.params['constant_sigma']**2) for n in self.samp_range]
        if self.params['catastrophic_outliers']:
            outlier_lf = gauss(self.params['outlier_mean'], self.params['outlier_sigma']**2)
        obs_samps = np.zeros(self.n_items)
        for n in self.samp_range:
            if np.random.uniform() < self.params['outlier_fraction']:
                obs_samps[n] = outlier_lf.sample_one()
            else:
                obs_samps[n] = true_lfs[n].sample_one()
        return obs_samps

    def evaluate_lfs(self):
        """
        Evaluates likelihoods based on observed sample values

        Returns
        -------
        obs_lfs: numpy.ndarray, float
            array of likelihood values for each item as a function of fine binning
        """
        if not self.params['variable_sigma']:
            lfs_fine = [gauss(self.x_fine[kk], self.params['constant_sigma']**2) for kk in range(self.n_tot)]
            obs_lfs = (1.-self.params['outlier_fraction']) * np.array([lfs_fine[kk].evaluate(self.obs_samps) for kk in range(self.n_tot)])
        if self.params['catastrophic_outliers']:
            outlier_lf = gauss(self.params['outlier_mean'], self.params['outlier_sigma']**2)
            obs_lfs += self.params['outlier_fraction'] * outlier_lf.evaluate(self.obs_samps)
        return obs_lfs.T

    def write(self, loc, style='plaintext'):
        """
        Function to write newly-created catalog to file

        Parameters
        ----------
        loc: string
            location into which to save catalog files
        style: string, optional
            file format in which to save the catalog
        """
        if style == 'plaintext':
            with open(loc, 'wb') as csvfile:
                out = csv.writer(csvfile, delimiter=' ')
                out.writerow(self.cat['bin_ends'])
                out.writerow(self.cat['log_interim_prior'])
                for line in self.cat['log_interim_posteriors']:
                    out.writerow(line)
        return

    def read(self, loc, style='plaintext'):
        """
        Function to read in catalog file

        Parameters
        ----------
        loc: string
            location of catalog file(s)
        """
        if style == 'plaintext':
            with open(loc, 'rb') as csvfile:
                tuples = (line.split(None) for line in csvfile)
                alldata = [[float(pair[k]) for k in range(0,len(pair))] for pair in tuples]
        self.cat['bin_ends'] = np.array(alldata[0])
        self.cat['log_interim_prior'] = np.array(alldata[1])
        self.cat['log_interim_posteriors'] = np.array(alldata[2:])
        return self.cat
