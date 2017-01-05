import numpy as np
import sys

import chippr
from chippr import utils as u

class gauss(object):

    def __init__(self, mean, var, limits=(-1./u.eps, 1./u.eps)):
        """
        A univariate Gaussian probability distribution object

        Parameters
        ----------
        mean: float
            mean of Gaussian probability distribution
        var: float
            variance of Gaussian probability distribution
        limits: tuple or list, optional
            minimum and maximum sample values to return
        """
        self.mean = mean
        self.var = var
        self.sigma = self.norm_var()
        self.invvar = self.invert_var()

        self.min_x = limits[0]
        self.max_x = limits[1]

    def norm_var(self):
        """
        Function to create standard deviation from variance
        """
        return np.sqrt(self.var)

    def invert_var(self):
        """
        Function to invert variance
        """
        return 1./self.var

    def evaluate_one(self, x):
        """
        Function to evaluate univariate Gaussian probability distribution once

        Parameters
        ----------
        x: float
            value at which to evaluate Gaussian probability distribution

        Returns
        -------
        p: float
            probability associated with x
        """
        p = u.eps
        if x > self.min_x and x < self.max_x:
            p = 1. / (np.sqrt(2. * np.pi) * self.sigma) * \
                np.exp(-0.5 * (x - self.mean) * self.invvar * (x - self.mean))
        p = max(u.eps, p)
        return p

    def evaluate(self, xs):
        """
        Function to evaluate univariate Gaussian probability distribution at multiple points

        Parameters
        ----------
        xs: float
            input values at which to evaluate probability

        Returns
        -------
        ps: ndarray, float
            output probabilities
        """
        ps = np.zeros_like(xs)
        for n, x in enumerate(xs):
            ps[n] += self.evaluate_one(x)
        return ps

    def sample_one(self):
        """
        Function to take one sample from univariate Gaussian probability distribution

        Returns
        -------
        x: float
            single sample from Gaussian probability distribution
        """
        x = self.mean + self.sigma * np.random.normal()
        return x

    def sample(self, n_samps):
        """
        Function to sample univariate Gaussian probability distribution

        Parameters
        ----------
        n_samps: positive int
            number of samples to take

        Returns
        -------
        xs: ndarray, float
            array of n_samps samples from Gaussian probability distribution
        """
        xs = np.array([self.sample_one() for n in range(n_samps)])
        return xs
