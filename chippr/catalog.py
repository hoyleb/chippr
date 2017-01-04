import numpy as np

import chippr
from chippr import sim_utils as su
from chippr import gauss

class catalog(object):

    def __init__(self, params=None, vb=True):
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

        if self.params is None:
            self.params = {}
            self.params['constant_sigma'] = 0.05

        if vb:
            print self.params

    def proc_bins(self, bins, vb=True):
        """
        Function to process binning

        Parameters
        ----------
        bins: int
            number of evenly spaced bins
        vb: boolean, optional
            True to print progress messages to stdout, False to suppress
        """
        if type(bins) == int:
            self.n_coarse = bins
        else:
            self.n_coarse = 10
        x_min = np.min(self.obs_samps)
        x_max = np.max(self.obs_samps)
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
        coarse = np.array([np.sum(coarse[k*self.n_fine:(k+1)*self.n_fine])*self.dx_fine for k in range(self.n_coarse)])
        coarse /= self.dx_coarse

        return coarse

    def create(self, truth, int_pr, bins=10):
        """
        Function creating a catalog of interim posterior probability distributions, will split this up into helper functions

        Parameters
        ----------
        truth: numpy.ndarray, float
            vector of true redshifts
        int_pr: chippr.gmix object or chippr.gauss object or chippr.binned object
            interim prior distribution object
        bins: int, optional
            number of evenly spaced bins

        Returns
        -------
        self.cat: dict
            dictionary comprising catalog information
        """
        true_samps = truth
        n_items = len(true_samps)
        samp_range = range(n_items)

        true_sigma = self.params['constant_sigma']
        true_lfs = [gauss(true_samps[n], true_sigma**2, limits=(0., 1.)) for n in samp_range]
        self.obs_samps = np.array([true_lfs[n].sample_one() for n in samp_range])

        self.proc_bins(bins)

        self.obs_lfs = np.array([[gauss(self.x_fine[kk], true_sigma**2).evaluate(self.obs_samps[n]) for kk in range(self.n_tot)] for n in samp_range])
        int_pr_fine = int_pr.evaluate(self.x_fine)
        int_pr_coarse = self.coarsify(int_pr_fine)

        pfs = np.zeros((n_items, self.n_coarse))
        for n in samp_range:
            pf = int_pr_fine * self.obs_lfs[n]
            pf = self.coarsify(pf)
            pfs[n] += pf

        self.cat = {}
        self.cat['bin_ends'] = self.bin_ends
        self.cat['interim_prior'] = int_pr_coarse
        self.cat['interim_posteriors'] = pfs

        return self.cat

    def write(self, loc):
        """
        Function to write newly-created catalog to file

        Parameters
        ----------
        loc: string
            location into which to save catalog files
        """
        return

    def read(self, loc):
        """
        Function to read in catalog file

        Parameters
        ----------
        loc: string
            location of catalog file(s)
        """
        return
