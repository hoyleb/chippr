# Wrapper for pomegranate.GeneralMixtureModel

import sys
import numpy as np

from pomegranate import GeneralMixtureModel as GMM

import chippr
from chippr import defaults as d
from chippr import utils as u

class gmix(object):

    def __init__(self, amps, funcs, limits=(d.min_x, d.max_x)):
        """
        Object to define a mixture probability distribution

        Parameters
        ----------
        amps: ndarray, float
            array with one relative amplitude per component
        funcs: list, chippr.gauss or chippr.discrete objects
            list of components
        limits: tuple or list or numpy.ndarray, float, optional
            minimum and maximum sample values to return
        """

        self.amps = amps/np.sum(amps)
        self.cumamps = np.cumsum(self.amps)
        self.n_comps = len(self.amps)

        self.funcs = funcs#[chippr.gauss(self.means[c], self.sigmas[c]**2) for c in range(self.n_comps)]
        # print('gmix before:')
        # for c in range(self.n_comps):
        #     print('gmix '+str((c, type(self.funcs[c]))))
        self.funcs = [func.dist for func in self.funcs]
        # print('gmix after:')
        # for c in range(self.n_comps):
        #     print('gmix '+str((c, type(self.funcs[c]))))

        self.dims = np.shape(np.array(limits).T)[0]
        self.min_x = limits[0]
        self.max_x = limits[1]
        # print("amps="+str(self.amps))
        self.dist = GMM(self.funcs, weights=self.amps)

    def pdf(self, xs):
        return self.evaluate(xs)

    def evaluate_one(self, x):
        """
        Function to evaluate Gaussian mixture
        once

        Parameters
        ----------
        x: float
            value at which to evaluate Gaussian mixture

        Returns
        -------
        p: float
            probability associated with x
        """
        # p = 0.
        # for c in range(self.n_comps):
        #     p += self.amps[c] * self.funcs[c].evaluate_one(x)
        p = self.dist.probability(x)
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
        # ps = np.zeros(len(xs))
        # for c in range(self.n_comps):
        #     ps += self.amps[c] * self.funcs[c].evaluate(xs)
        ps = self.dist.probability(xs)
        return ps

    def sample_one(self):
        """
        Function to sample a single value from Gaussian mixture probability distribution

        Returns
        -------
        x: float
            a single point sampled from the Gaussian mixture probability distribution
        """

        # x = -1. * np.ones(self.dims)
        # #don't do this every time!
        # min_x = self.min_x * np.ones(self.dims)
        # max_x = self.max_x * np.ones(self.dims)
        #
        # while np.any(np.less(x, min_x)) or np.any(np.greater(x, max_x)):
        #     r = np.random.uniform(0., self.cumamps[-1])
        #     c = 0
        #     for k in range(1, self.n_comps):
        #         if r > self.cumamps[k-1]:
        #             c = k
        #     x = self.funcs[c].sample_one()
        x = self.dist.sample(1)
        return x

    def sample(self, n_samps):
        """
        Function to take samples from Gaussian mixture probability distribution

        Parameters
        ----------
        n_samps: int
            number of samples to take

        Returns
        -------
        xs: ndarray, float
            array of points sampled from the Gaussian mixture probability distribution
        """
        # print('gmix trying to sample '+str(n_samps)+' from '+str(self.dist))
        # xs = np.array([self.sample_one() for n in range(n_samps)])
        # print(self.dist.to_json)
        xs = np.array(self.dist.sample(n_samps))
        # print('gmix sampled '+str(n_samps)+' from '+str(self.dist))
        return xs
