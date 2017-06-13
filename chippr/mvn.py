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

        Returns
        -------
        det: float
            determinant of variance
        """
        det = np.linalg.det(self.var)
        return det

    def invert_var(self):
        """
        Function to invert covariance matrix

        Returns
        -------
        inv: numpy.ndarray, float
            inverse variance
        """
        inv = np.linalg.inv(self.var)
        return inv

    def evaluate_one(self, z):
        """
        Function to evaluate multivariate Gaussian probability distribution
        once

        Parameters
        ----------
        z: numpy.ndarray, float
            value at which to evaluate multivariate Gaussian probability
            distribution

        Returns
        -------
        p: float
            probability associated with z
        """
        norm_z = z - self.mean
        p = max(d.eps, 1. / (np.sqrt(2. * np.pi) * self.sigma) * \
                np.exp(-0.5 * np.dot(np.dot(norm_z, self.invvar), norm_z)))
        return p

    def evaluate(self, zs):
        """
        Function to evaluate multivariate Gaussian probability distribution at
        multiple points

        Parameters
        ----------
        zs: ndarray, float
            input vectors at which to evaluate probability

        Returns
        -------
        ps: ndarray, float
            output probabilities
        """
        ps = np.zeros(len(zs))
        for n, z in enumerate(zs):
            ps[n] += self.evaluate_one(z)
        return ps

    def sample_one(self):
        """
        Function to take one sample from multivariate Gaussian probability
        distribution

        Returns
        -------
        z: numpy.ndarray, float
            single sample from multivariate Gaussian probability distribution
        """
        z = np.random.multivariate_normal(self.mean, self.var, 1)[0]#self.mean + np.dot(self.var, np.random.normal(size = self.dim))
        return z

    def sample(self, n_samps):
        """
        Function to sample from multivariate Gaussian probability distribution

        Parameters
        ----------
        n_samps: positive int
            number of samples to take

        Returns
        -------
        zs: ndarray, float
            array of n_samps samples from multivariate Gaussian probability
            distribution
        """
        zs = np.array([self.sample_one() for n in range(n_samps)])
        return zs
