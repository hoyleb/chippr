# Wrapper for pomegranate.distributions.NormalDistribution

import sys

import numpy as np

from pomegranate.distributions import NormalDistribution as ND

import chippr

class gauss(object):

    def __init__(self, mean, var, bounds=None):
        """
        A univariate Gaussian probability distribution object

        Parameters
        ----------
        mean: float
            mean of Gaussian probability distribution
        var: float
            variance of Gaussian probability distribution
        """
        self.mean = mean
        self.var = var
        self.sigma = self.norm_var()
        self.invvar = self.invert_var()
        self.dist = ND(self.mean, self.sigma)            

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

    def pdf(self, xs):
        return self.evaluate(xs)

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
        # p = 1. / (np.sqrt(2. * np.pi) * self.sigma) * \
        # np.exp(-0.5 * (self.mean - x) * self.invvar * (self.mean - x))
        p = self.dist.probability(x)
        return p

    def evaluate(self, xs):
        """
        Function to evaluate univariate Gaussian probability distribution at multiple points

        Parameters
        ----------
        xs: numpy.ndarray, float
            input values at which to evaluate probability

        Returns
        -------
        ps: ndarray, float
            output probabilities
        """
        # ps = 1. / (np.sqrt(2. * np.pi) * self.sigma) * \
        # np.exp(-0.5 * (self.mean - xs) * self.invvar * (self.mean - xs))
        # ps = np.zeros_like(xs)
        # for n, x in enumerate(xs):
        #     ps[n] += self.evaluate_one(x)
        ps = self.dist.probability(xs)
        return ps

    def sample_one(self):
        """
        Function to take one sample from univariate Gaussian probability distribution

        Returns
        -------
        x: float
            single sample from Gaussian probability distribution
        """
        # x = self.mean + self.sigma * np.random.normal()
        x = self.dist.sample(1)
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
        # print('gauss trying to sample '+str(n_samps)+' from '+str(self.dist))
        # xs = np.array([self.sample_one() for n in range(n_samps)])
        xs = np.array(self.dist.sample(n_samps))
        # print('gauss sampled '+str(n_samps)+' from '+str(self.dist))
        return xs
