import numpy as np
import csv
import timeit
import os
import pickle as pkl

import matplotlib as mpl
mpl.use('PS')
import matplotlib.pyplot as plt

import chippr
from chippr import defaults as d
from chippr import utils as u
from chippr import sim_utils as su
from chippr import gauss
from chippr import discrete
from chippr import multi_dist
from chippr import gmix
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
        with open(os.path.join(self.data_dir, 'params.p'), 'wb') as paramfile:
            pkl.dump(self.params, paramfile)

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
        self.n_fine = 2#self.n_coarse
        self.n_tot = self.n_coarse * self.n_fine
        z_range = self.z_max - self.z_min

        self.dz_coarse = z_range / self.n_coarse
        self.dz_fine = z_range / self.n_tot

        self.z_coarse = np.linspace(self.z_min + 0.5 * self.dz_coarse, self.z_max - 0.5 * self.dz_coarse, self.n_coarse)
        self.z_fine = np.linspace(self.z_min + 0.5 * self.dz_fine, self.z_max - 0.5 * self.dz_fine, self.n_tot)
        self.z_all = np.linspace(self.z_min, self.z_max, self.n_tot + 1)

        self.bin_ends = np.linspace(self.z_min, self.z_max, self.n_coarse+1)
        self.bin_difs_coarse = self.dz_coarse * np.ones(self.n_coarse)
        self.bin_difs_fine = self.dz_fine * np.ones(self.n_tot)

        return

    def coarsify(self, fine):
        """
        Function to bin function evaluated on fine grid

        Parameters
        ----------
        fine: numpy.ndarray, float
            matrix of probability values of function on fine grid for N galaxies

        Returns
        -------
        coarse: numpy.ndarray, float
            vector of binned values of function
        """
        fine = fine.T
        fine /= np.sum(fine, axis=0)[np.newaxis, :] * self.dz_fine
        coarse = np.array([np.sum(fine[k * self.n_fine : (k+1) * self.n_fine], axis=0) * self.dz_fine for k in range(self.n_coarse)])
        coarse /= np.sum(coarse, axis=0)[np.newaxis, :]  * self.dz_coarse
        coarse = coarse.T
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
        self.N = 10**N
        self.N_range = range(self.N)
        self.truth = truth

        self.proc_bins()

        self.prob_space = self.make_probs()
        # print('make_probs returns '+str(type(self.prob_space)))
        if vb:
            plots.plot_prob_space(self.z_fine, self.prob_space, plot_loc=self.plot_dir)

        ## next, sample discrete to get z_true, z_obs
        self.samps = self.prob_space.sample(self.N)
        self.cat['true_vals'] = self.samps
        if vb:
            plots.plot_true_histogram(self.samps.T[0], plot_loc=self.plot_dir)

        ## then literally take slices (evaluate at constant z_phot)
        self.obs_lfs = self.evaluate_lfs()
        #self.obs_lfs /= np.sum(self.obs_lfs, axis=1)[:, np.newaxis] * self.dz_fine

        if vb:
            plots.plot_scatter(self.samps, self.obs_lfs, self.z_fine, plot_loc=self.plot_dir)

        self.int_pr = int_pr
        int_pr_fine = np.array([self.int_pr.pdf(self.z_fine)])
        int_pr_coarse = self.coarsify(int_pr_fine)
        truth_fine = self.truth.pdf(self.z_fine)

        pfs_fine = self.obs_lfs * int_pr_fine[np.newaxis, :] / truth_fine[np.newaxis, :]
        pfs_coarse = self.coarsify(pfs_fine)

        self.cat['bin_ends'] = self.bin_ends
        self.cat['log_interim_prior'] = u.safe_log(int_pr_coarse[0])
        self.cat['log_interim_posteriors'] = u.safe_log(pfs_coarse[0])

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
        only one outlier population at a time for now
        """
        x_vals = self.z_fine#self.truth.sample(self.N)
        hor_amps = self.truth.evaluate(x_vals) * self.bin_difs_fine
        hor_funcs = [discrete(np.array([self.z_all[kk], self.z_all[kk+1]]), np.array([1.])) for kk in range(self.n_tot)]

        sigma = self.params['constant_sigma']
        if not self.params['variable_sigmas']:
            sigmas = np.ones(self.N) * sigma
        else:
            sigmas = sigma * (1. + x_vals)

        if self.params['ez_bias']:
            if self.params['variable_bias']:
                y_means = x_vals + self.params['ez_bias_val'] * (1. + x_vals)
            else:
                y_means = x_vals + self.params['ez_bias_val']
        else:
            y_means = x_vals

        vert_funcs = [gauss(y_means[kk], sigmas[kk]) for kk in range(self.n_tot)]

        # grid_amps = self.truth.evaluate(x_vals)
        #
        # grid_means, grid_amps, uniform_lfs = self._setup_prob_space(true_func)
        #
        # pdf_means = self._make_bias(grid_means)

        # WILL REFACTOR THIS TO ADD BACK CATASTROPHIC OUTLIER SUPPORT
        # if self.params['catastrophic_outliers'] != '0':
        #     if self.params['catastrophic_outliers'] == 'uniform':
        #
        #     else:
        #         self.outlier_lf = gauss(self.params['outlier_mean'], self.params['outlier_sigma']**2)
        #         in_amps = np.ones(self.n_tot)
        #         if self.params['catastrophic_outliers'] == 'template':
        #             out_funcs = [multi_dist([uniform_lfs[kk], self.outlier_lf]) for kk in range(self.n_tot)]
        #             out_amps = uniform_lf.pdf(grid_means)
        #
        #         elif self.params['catastrophic_outliers'] == 'training':
        #             out_funcs = [multi_dist([uniform_lfs[kk], uniform_lf]) for kk in range(self.n_tot)]
        #             out_amps = self.outlier_lf.pdf(grid_means)
        #
        #         out_amps /= np.dot(out_amps, self.bin_difs_fine)
        #         in_amps *= (1. - self.params['outlier_fraction'])
        #         out_amps *= self.params['outlier_fraction']
        #         try:
        #             test_out_frac = np.dot(out_amps, self.bin_difs_fine)
        #             assert np.isclose(test_out_frac, self.params['outlier_fraction'])
        #         except:
        #             print('outlier fraction not normalized: '+str(test_out_frac))
        #         grid_funcs = [gmix(np.array([in_amps[kk], out_amps[kk]]), [grid_funcs[kk], out_funcs[kk]]) for kk in range(self.n_tot)]
        #         # np.append(grid_means, [self.params['outlier_mean'], self.uniform_lf.sample_one()])

        # true n(z) in z_spec, uniform in z_phot
        # grid_amps *= true_func.evaluate(grid_means)
        p_space = gmix(hor_amps, [multi_dist([hor_funcs[kk], vert_funcs[kk]]) for kk in range(self.n_tot)])

        return p_space

    # def _setup_prob_space(self):
    #     """
    #     Helper function for make_probs
    #
    #     Parameters
    #     ----------
    #
    #     Returns
    #     -------
    #     grid_means: numpy.ndarray, float
    #         average redshift of a bin, not weighted by probability
    #     grid_amps: numpy.ndarray, float
    #         model probabilities at grid_means
    #     uniform_lfs: chippr.discrete object
    #         uniform distribution for each redshift bin
    #     """
    #     x_grid = self.z_fine
    #
    #     if self.params['ez_bias']:
    #         if self.params['variable_bias']:
    #             means = x_grid + self.params['ez_bias_val'] * (1. + x_grid)
    #         else:
    #             means = x_grid + self.params['ez_bias_val']
    #     else:
    #         means = x_grid
    #
    #     sigma = self.params['constant_sigma']
    #     if not self.params['variable_sigmas']:
    #         sigmas = np.ones(self.N) * sigma
    #     else:
    #         sigmas = sigma * (1. + means)
    #
    #
    #
    #     if self.params['catastrophic_outliers'] != '0':
    #         if self.params['catastrophic_outliers'] == 'uniform':
    #             uniform_lf = discrete(np.array([self.z_min, self.z_max]), np.array([1.]))
    #             out_amps =
    #         else:
    #             self.outlier_lf = gauss(self.params['outlier_mean'], self.params['outlier_sigma']**2)
    #             in_amps = np.ones(self.n_tot)
    #             if self.params['catastrophic_outliers'] == 'template':
    #                 out_funcs = [multi_dist([uniform_lfs[kk], self.outlier_lf]) for kk in range(self.n_tot)]
    #                 out_amps = uniform_lf.pdf(grid_means)
    #
    #             elif self.params['catastrophic_outliers'] == 'training':
    #                 out_funcs = [multi_dist([uniform_lfs[kk], uniform_lf]) for kk in range(self.n_tot)]
    #                 out_amps = self.outlier_lf.pdf(grid_means)
    #
    #             out_amps /= np.dot(out_amps, self.bin_difs_fine)
    #             in_amps *= (1. - self.params['outlier_fraction'])
    #             out_amps *= self.params['outlier_fraction']
    #             try:
    #                 test_out_frac = np.dot(out_amps, self.bin_difs_fine)
    #                 assert np.isclose(test_out_frac, self.params['outlier_fraction'])
    #             except:
    #                 print('outlier fraction not normalized: '+str(test_out_frac))
    #             grid_funcs = [gmix(np.array([in_amps[kk], out_amps[kk]]), [grid_funcs[kk], out_funcs[kk]]) for kk in range(self.n_tot)]
    #
    #     # grid_means = self.z_fine#np.array([(self.z_fine[kk], self.z_fine[kk]]) for kk in range(self.n_tot)])
    #     # grid_amps = true_func.pdf(grid_means)
    #     # grid_amps /= np.dot(grid_amps, self.bin_difs_fine)
    #     # try:
    #     #     test_norm = np.dot(grid_amps, self.bin_difs_fine)
    #     #     assert np.isclose(test_norm, 1.)
    #     # except:
    #     #     print('n(z) PDF amplitudes not normalized: '+str(test_norm))
    #
    #     # uniform_lf = discrete(np.array([self.z_min, self.z_max]), np.array([1.]))
    #     uniform_lfs = [discrete(np.array([grid_means[kk] - self.dz_fine / 2., grid_means[kk] + self.dz_fine / 2.]), np.array([1.])) for kk in range(self.n_tot)]
    #     # return(x_grid, x_amps)
    #     return(grid_means, grid_amps, uniform_lfs)

    # def _make_bias(self, grid_means):
        # """
        # Introduces global redshift bias
        #
        # Parameters
        # ----------
        # grid_means: numpy.ndarray, float
        #     cental redshifts of each bin
        #
        # Returns
        # -------
        # pdf_means: numpy.ndarray, float
        #     cental redshifts to use as Gaussian means
        # """
        # pdf_means = grid_means
        # if self.params['ez_bias']:
        #     pdf_means += self.params['ez_bias_val'] * (1. + grid_means)
        # return(pdf_means)

    # def _make_scatter(self):
    #     """
    #     """

    def sample(self, N, vb=False):
        """
        Samples (z_spec, z_phot) pairs

        Parameters
        ----------
        N: int
            number of samples to take
        vb: boolean
            print progress to stdout?

        Returns
        -------
        samps: numpy.ndarray, float
            (z_spec, z_phot) pairs
        """
        self.n_gals = N
        if vb: print(self.n_gals)
        samps = self.prob_space.sample(self.n_gals)
        return samps

    def evaluate_lfs(self, vb=True):
        """
        Evaluates likelihoods based on observed sample values

        Parameters
        ----------
        vb: boolean
            print progress to stdout?

        Returns
        -------
        lfs: numpy.ndarray, float
            array of likelihood values for each item as a function of fine
            binning
        """
        lfs = []
        for n in self.N_range:
            points = zip(self.z_fine, [self.samps[n][1]] * self.n_tot)
            lfs.append(self.prob_space.pdf(np.array(points)))
        lfs = np.array(lfs)
        lfs /= np.sum(lfs, axis=-1)[:, np.newaxis] * self.dz_fine
        return lfs

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
            with open(os.path.join(self.data_dir, 'true_vals' + style), 'wb') as csvfile:
                out = csv.writer(csvfile, delimiter=' ')
                for line in self.cat['true_vals']:
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
