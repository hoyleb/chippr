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
            self.params['raw'] = 0

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
        self.n_fine = 10#self.n_coarse
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
        # self.true_samps = self.truth.sample(10**N)
        # self.n_items = len(self.true_samps)
        # self.samp_range = range(self.n_items)

        self.proc_bins()

        self.prob_space = self.make_probs()
        if vb:
            plots.plot_prob_space(self.z_fine, self.prob_space, plot_loc=self.plot_dir)

        ## this far!
        ## next, sample discrete to get z_true, z_obs
        self.samps = self.sample(self.N)
        if vb:
            self.cat['true_vals'] = self.samps
            plots.plot_true_histogram(self.samps.T[0], plot_loc=self.plot_dir)

        ## then literally take slices (evaluate at constant z_phot)
        self.obs_lfs = self.evaluate_lfs()
        #self.obs_lfs /= np.sum(self.obs_lfs, axis=1)[:, np.newaxis] * self.dz_fine

        if vb:
            plots.plot_scatter(self.samps, self.obs_lfs, self.z_fine, plot_loc=self.plot_dir)

        self.int_pr = int_pr
        int_pr_fine = np.array([self.int_pr.evaluate(self.z_fine)])
        int_pr_coarse = self.coarsify(int_pr_fine)
        truth_fine = self.truth.evaluate(self.z_fine)

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
        Does not currently support variable sigmas, only one outlier population at a time
        """
        # this is one Gaussian for each z_spec, to be evaluated at each z_phot
        true_func = self.truth#multi_dist([self.truth, self.uniform_lf])
        mins = [true_func.min_x, -100.]
        maxs = [true_func.max_x, 100.]
        grid_means = self.z_fine#np.array([(self.z_fine[kk], self.z_fine[kk]]) for kk in range(self.n_tot)])
        grid_amps = true_func.evaluate(grid_means)#np.ones(self.n_tot)#
        grid_amps /= (np.sum(grid_amps) * self.dz_fine)
        assert np.isclose(np.sum(grid_amps) * self.dz_fine, 1.)

        uniform_lf = discrete(np.array([self.z_min, self.z_max]), np.array([1.]))
        uniform_lfs = [discrete(np.array([grid_means[kk] - self.dz_fine / 2., grid_means[kk] + self.dz_fine / 2.]), np.array([1.])) for kk in range(self.n_tot)]

        if not self.params['variable_sigmas']:
            grid_sigma = self.params['constant_sigma']# * np.identity(2)
            # grid_sigmas = self.params['constant_sigma'] * np.ones(self.n_tot)#np.ones((self.n_tot, self.n_tot, 2))
            grid_funcs = [gauss(grid_means[kk], grid_sigma**2) for kk in range(self.n_tot)]#[mvn(grid_means[kk], grid_sigma**2) for kk in range(self.n_tot)]#[[mvn(grid_means[kk][jj], grid_sigmas[kk][jj]**2) for jj in range(self.n_tot)] for kk in range(self.n_tot)]
            grid_funcs = [multi_dist([uniform_lfs[kk], grid_funcs[kk]]) for kk in range(self.n_tot)]
        else:
            print('variable sigmas is not yet implemented')
            return

        if self.params['catastrophic_outliers'] != '0':
            # np.append(grid_amps, [self.params['outlier_fraction'] / self.n_tot])
            self.outlier_lf = gauss(self.params['outlier_mean'], self.params['outlier_sigma']**2)
            # out_amps = np.ones(self.n_tot) * self.params['outlier_fraction'] / self.n_items
            # grid_amps *= (1. - out_amp) / self.n_items
            # grid_amps.append(out_amp)

            in_amps = np.ones(self.n_tot)#grid_amps# * (1. - self.params['outlier_fraction'])
            # out_amps = np.ones(self.n_tot)#grid_amps# * self.params['outlier_fraction']
            if self.params['catastrophic_outliers'] == 'template':
                out_funcs = [multi_dist([uniform_lfs[kk], self.outlier_lf]) for kk in range(self.n_tot)]
                out_amps = uniform_lf.evaluate(grid_means)

            elif self.params['catastrophic_outliers'] == 'training':
                out_funcs = [multi_dist([uniform_lfs[kk], uniform_lf]) for kk in range(self.n_tot)]
                out_amps = self.outlier_lf.evaluate(grid_means)

            out_amps /= (np.sum(out_amps) * self.dz_fine)
            in_amps *= (1. - self.params['outlier_fraction'])
            out_amps *= self.params['outlier_fraction']
            assert np.isclose(np.sum(in_amps) * self.dz_fine, (1. - self.params['outlier_fraction']))
            assert np.isclose(np.sum(out_amps) * self.dz_fine, self.params['outlier_fraction'])
            grid_funcs = [gmix(np.array([in_amps[kk], out_amps[kk]]), [grid_funcs[kk], out_funcs[kk]], limits=(mins, maxs)) for kk in range(self.n_tot)]
                # np.append(grid_means, [self.params['outlier_mean'], self.uniform_lf.sample_one()])

        # true n(z) in z_spec, uniform in z_phot
        # grid_amps *= true_func.evaluate(grid_means)
        p_space = gmix(grid_amps, grid_funcs, limits=(mins, maxs))

        return p_space

    def sample(self, N, vb=True):
        """
        Samples (z_spec, z_phot) pairs

        Parameters
        ----------
        N: int
            number og samples to take
        vb: boolean
            print progress to stdout?

        Returns
        -------
        samps: numpy.ndarray, float
            (z_spec, z_phot) pairs
        """
        self.n_gals = N
        samps = self.prob_space.sample(N)
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
            lfs.append(self.prob_space.evaluate(np.array(points)))
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
