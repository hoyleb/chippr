import numpy as np
import sys

import chippr
from chippr import defaults as d
from chippr import utils as u

class mvn(object):

    def __init__(self, mean, var):
        """
        A multivariate Gaussian probability distribution object

        Parameters
        ----------
        mean: numpy.ndarray, float
            mean of multivariate Gaussian probability distribution
        var: numpy.ndarray, float
            covariance matrix of multivariate Gaussian probability distribution
        """
        self.mean = mean
        self.dim = len(self.mean)
        self.var = var
        self.sigma = self.norm_var()
        self.invvar = self.invert_var()

        assert np.linalg.eig(self.var) > 0.

    def norm_var(self):
        """
        Function to normalize covariance matrix
        """
        return np.linalg.det(self.var)

    def invert_var(self):
        """
        Function to invert covariance matrix
        """
        return np.linalg.inv(self.var)

    def evaluate_one(self, x):
        """
        Function to evaluate multivariate Gaussian probability distribution once

        Parameters
        ----------
        x: numpy.ndarray, float
            value at which to evaluate multivariate Gaussian probability distribution

        Returns
        -------
        p: float
            probability associated with x
        """
        p = max(d.eps, 1. / (np.sqrt(2. * np.pi) * self.sigma) * \
                np.exp(-0.5 * (x - self.mean) * self.invvar * (x - self.mean)))
        return p

    def evaluate(self, xs):
        """
        Function to evaluate multivariate Gaussian probability distribution at multiple points

        Parameters
        ----------
        xs: ndarray, float
            input vectors at which to evaluate probability

        Returns
        -------
        ps: ndarray, float
            output probabilities
        """
        ps = np.zeros(len(xs))
        for n, x in enumerate(xs):
            ps[n] += self.evaluate_one(x)
        return ps

    def sample_one(self):
        """
        Function to take one sample from multivariate Gaussian probability distribution

        Returns
        -------
        x: numpy.ndarray, float
            single sample from multivariate Gaussian probability distribution
        """
        x = self.mean + self.var * np.array([np.random.normal() for k in range(self.dim)])
        return x

    def sample(self, n_samps):
        """
        Function to sample from multivariate Gaussian probability distribution

        Parameters
        ----------
        n_samps: positive int
            number of samples to take

        Returns
        -------
        xs: ndarray, float
            array of n_samps samples from multivariate Gaussian probability distribution
        """
        xs = np.array([self.sample_one() for n in range(n_samps)])
        return xs
