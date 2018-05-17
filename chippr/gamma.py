# Wrapper for pomegranate.distributions.GammaDistribution

import sys

import numpy as np
import scipy.misc as spm

from pomegranate.distributions import GammaDistribution as GD

import chippr
from chippr import defaults as d

class gamma(object):

    def __init__(self, shape, ratesqr, bounds=None):
        """
        A univariate Gamma (Erlang) probability distribution object

        Parameters
        ----------
        shape: int
            shape parameter of gamma (erlang) probability distribution
        ratesqr: float
            square of rate parameter of gamma (erlang) probability distribution
        bounds: tuple, float, optional
            limits of the distribution
        """
        self.shape = shape
        self.rate = np.sqrt(ratesqr)
        self.dist = GD(self.shape, self.rate)

        self.bounds = bounds
        if self.bounds is not None:
            def cdf(x):
                return 1. - sum([1. / spm.factorial(n) * np.exp(-1. * self.rate * x) * (self.rate * x) ** n for n in np.arange(self.shape)])
            self.integral = cdf(self.bounds[1]) - cdf(self.bounds[0])
            self.scale = 1. / self.integral#integrate between bounds
        else:
            self.scale = 1.

    def check_bounds(self, x):
        """
        Checks if point of evaluation is within the bounds

        Parameters
        ----------
        x: float or array
            value at which to check boundedness

        Returns
        -------
        y: boolean
            true if within bounds
        """
        x = np.asarray(x)
        if self.bounds is not None:
            low = x >= self.bounds[0]
            high = x <= self.bounds[1]
            y = low * high
        else:
            y = np.ones_like(x, dtype=bool)
        return y

    def pdf(self, xs):
        return self.evaluate(xs)

    def evaluate_one(self, x):
        """
        Function to evaluate Gamma probability distribution once

        Parameters
        ----------
        x: float
            value at which to evaluate Gamma probability distribution

        Returns
        -------
        p: float
            probability associated with x
        """
        # p = 1. / (np.sqrt(2. * np.pi) * self.sigma) * \
        # np.exp(-0.5 * (self.mean - x) * self.invvar * (self.mean - x))
        if self.check_bounds(x):
            p = self.scale * self.dist.probability(x)
        else:
            p = d.eps
        return p

    def evaluate(self, xs):
        """
        Function to evaluate univariate Gamma probability distribution at multiple points

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
        if self.check_bounds(xs).all():
            ps = self.scale * self.dist.probability(xs)
        else:
            ps = d.eps * np.ones(len(xs))
        return ps

    def sample_one(self):
        """
        Function to take one sample from univariate Gamma probability distribution

        Returns
        -------
        x: float
            single sample from Gamma probability distribution
        """
        # x = self.mean + self.sigma * np.random.normal()
        bounded = False
        while not bounded:
            x = self.dist.sample(1)
            bounded = self.check_bounds(x)
        return x

    def sample(self, n_samps):
        """
        Function to sample univariate Gamma probability distribution

        Parameters
        ----------
        n_samps: positive int
            number of samples to take

        Returns
        -------
        xs: ndarray, float
            array of n_samps samples from Gamma probability distribution
        """
        # print('gauss trying to sample '+str(n_samps)+' from '+str(self.dist))
        # xs = np.array([self.sample_one() for n in range(n_samps)])
        def partial_sample(n_s, x=None):
            # if x is None:
            #     x = np.array(self.dist.sample(n_s))
            # elif type(x) is not np.ndarray:
            # #     x = np.asarray(x)
            # else:
            x_again = np.array([self.dist.sample() for i in range(n_s)]).flatten()
            if x is None:
                x = x_again
            else:
                x = np.concatenate((x, x_again))
            return (n_s, x)
        i = 0
        xs = None
        while i < n_samps:
            (i, xs) = partial_sample(n_samps - i, x=xs)
            xs = xs[self.check_bounds(xs)]
            i = len(xs)
        # print('gauss sampled '+str(n_samps)+' from '+str(self.dist))
        return xs[:n_samps]
