import numpy as np
import sys

import chippr
from chipper import utils as u

class gmix(object):

    def __init__(self, amps, means, sigmas):
        """
        Object to define a Gaussian mixture probability distribution

        Parameters
        ----------
        amps: ndarray, float
            array with one relative amplitude per component
        means: ndarray, float
            array with one mean per component
        sigmas: ndarray, float
            array with one standard deviation per component
        """

        self.amps = amps/np.sum(amps)
        self.cumamps = np.cumsum(self.amps)
        self.means = means
        self.sigmas = sigmas
        self.n_comps = len(self.amps)

        self.funcs = [chippr.gauss(self.means[c], self.sigmas[c]) for c in range(self.n_comps)]

    def evaluate_one(self, x):
        """
        Function to evaluate the Gaussian mixture probability distribution at one point

        Parameters
        ----------
        x: float
            value at which to evaluate Gaussian mixture probability distribution

        Returns
        -------
        p: float
            value of Gaussian mixture probability distribution at x
        """
        p = u.eps
        for c in range(self.n_comps):
            p += self.amps[c] * self.funcs[c].evaluate_one(x)
        return p

    def evaluate(self, xs):
        """
        Function to evaluate the Gaussian mixture probability distribution at many points

        Parameters
        ----------
        xs: ndarray, float
            values at which to evaluate Gaussian mixture probability distribution

        Returns
        -------
        ps: ndarray, float
            values of Gaussian mixture probability distribution at xs
        """
        n_vals = len(xs)
        ps = np.array([self.evaluate_one(x) for x in xs])
        return ps

    def sample_one(self):
        """
        Function to sample a single value from Gaussian mixture probability distribution

        Returns
        -------
        x: float
            a single point sampled from the Gaussian mixture probability distribution
        """
        r = np.random.uniform(0., self.cumamps[-1])
        c = 0
        for k in range(1, self.n_comps):
            if r > self.cumamps[k-1]:
                c = k
        x = self.funcs[c].sample_one()
        return x

    def sample(self, n_samps):
        """
        Function to take samples from Gaussian mixture probability distribution

        Returns
        -------
        xs: ndarray, float
            array of points sampled from the Gaussian mixture probability distribution
        """
        xs = np.array([self.sample_one() for n in range(n_samps)])
        return xs