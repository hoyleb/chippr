import numpy as np
import sys

import chippr
from chippr import utils as u

class gauss(object):

    def __init__(self, mean, var, limits=(-1./u.eps, 1./u.eps)):
        """
        A Gaussian probability distribution object

        Parameters
        ----------
        mean: float or numpy.ndarray, float
            mean of Gaussian probability distribution
        var: float or numpy.ndarray, float
            variance or covariance matrix of Gaussian probability distribution
        limits: tuple or list or numpy.ndarray, float, optional
            minimum and maximum sample values to return
        """
        self.mean = mean
        self.var = var
        self.sigma = self.norm_var()
        self.invvar = self.invert_var()

        if type(self.var) == np.ndarray:
            assert np.linalg.eig(self.var) > 0.

        self.min_x = limits[0]
        self.max_x = limits[1]

    def norm_var(self):
        """
        Function to normalize variance in the cases of float or matrix
        """
        if type(self.var) == np.float64 or type(self.var) == float:
            return np.sqrt(self.var)

        if type(self.var) == np.ndarray:
            return np.linalg.det(self.var)

    def invert_var(self):
        """
        Function to invert variance in the cases of float or matrix
        """
        if type(self.var) == np.float64 or type(self.var) == float:
            return 1./self.var

        if type(self.var) == np.ndarray:
            return np.linalg.inv(self.var)

    def evaluate_one(self, x):
        """
        Function to evaluate Gaussian probability distribution once

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
        p += 1. / (np.sqrt(2. * np.pi) * self.sigma) * \
            np.exp(-0.5 * (x - self.mean) * self.invvar * (x - self.mean))
        return p

    def evaluate(self, xs):
        """
        Function to evaluate Gaussian probability distribution at multiple points

        Parameters
        ----------
        xs: float or ndarray, float
            input values at which to evaluate probability

        Returns
        -------
        ps: float or ndarray, float
            output probabilities
        """
        if type(xs) != np.ndarray:
            ps = self.evaluate_one(xs)
        else:
            ps = np.zeros_like(xs)
            for n,z in enumerate(xs):
                ps[n] += self.evaluate_one(x)
        return ps

    def sample_one(self):
        """
        Function to take one sample from Gaussian probability distribution

        Returns
        -------
        x: float
            single sample from Gaussian probability distribution
        """
        x = -1.
        while x < self.min_x or x > self.max_x:
            x = self.mean + self.sigma * np.random.normal()
        return x

    def sample(self, n_samps):
        """
        Function to sample from Gaussian probability distribution

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
