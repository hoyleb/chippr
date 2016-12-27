import numpy as np
import sys

import chippr
from chippr import utils as u

class gauss(object):

    def __init__(self, mean, sigma):
        """
        A Gaussian probability distribution object

        Parameters
        ----------
        mean: float
            mean of Gaussian probability distribution
        sigma: float
            standard deviation of Gaussian probability distribution
        """
        self.epsilon = sys.float_info.epsilon
        self.mean = mean
        self.sigma = sigma

    def evaluate_one(self, x):
        """
        Function to evaluate Gaussian probability distribution once

        Parameters
        ----------
        x: float
            value at which to evaluate Gaussian probability distribution
        xmin: float, optional
            minimum value below which probability is set to epsilon
        xmax: float, optional
            maximum value above which probability is set to epsilon

        Returns
        -------
        p: float
            probability associated with x
        """
        p = u.eps
        p += 1. / (np.sqrt(2. * np.pi) * self.sigma) * \
            np.exp(-0.5 * (x - self.mean) ** 2 / self.sigma ** 2)
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
        if type(zs) != np.ndarray:
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
