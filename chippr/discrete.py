import numpy as np
import bisect

import chippr
from chippr import defaults as d
from chippr import sim_utils as su

class discrete(object):
    def __init__(self, bin_ends, weights):
        """
        Binned function class for any discrete function

        Parameters
        ----------
        bin_ends: numpy.ndarray, float
            endpoints of bins
        weights: numpy.ndarray, float
            relative weights associated with each bin
        """
        self.bin_ends = bin_ends
        self.dbins = self.bin_ends[1:] - self.bin_ends[:-1]
        self.n_bins = len(self.bin_ends)-1
        self.bin_range = range(self.n_bins)

        self.weights = weights
        self.normweights = np.cumsum(self.weights) / np.sum(self.weights)
        self.distweights = np.cumsum(self.weights) / np.dot(self.weights, self.dbins)

    def pdf(self, xs):
        return self.evaluate(xs)

    def evaluate_one(self, x):
        """
        Function to evaluate the discrete probability distribution at one point

        Parameters
        ----------
        x: float
            value at which to evaluate discrete probability distribution

        Returns
        -------
        p: float
            value of discrete probability distribution at x
        """
        p = d.eps
        for k in self.bin_range:
            try:
                if x > self.bin_ends[k] and x < self.bin_ends[k+1]:
                    p = self.distweights[k]
            except ValueError:
                print('x should be a float: '+str(x))
        return p

    def evaluate(self, xs):
        """
        Function to evaluate the discrete probability distribution at many points

        Parameters
        ----------
        xs: ndarray, float
            values at which to evaluate discrete probability distribution

        Returns
        -------
        ps: ndarray, float
            values of discrete probability distribution at xs
        """
        ps = np.array([self.evaluate_one(x) for x in xs])
        return ps

    def sample_one(self):
        """
        Function to sample a single value from discrete probability distribution

        Returns
        -------
        x: float
            a single point sampled from the discrete probability distribution
        """
        r = np.random.random()
        k = bisect.bisect(self.normweights, r)

        x = np.random.uniform(low=self.bin_ends[k], high=self.bin_ends[k+1])
        return x

    def sample(self, n_samps):
        """
        Function to take samples from discrete probability distribution

        Parameters
        ----------
        n_samps: int
            number of samples to take

        Returns
        -------
        xs: ndarray, float
            array of points sampled from the discrete probability distribution
        """
        xs = np.array([self.sample_one() for n in range(n_samps)])
        return xs
