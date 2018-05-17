# Wrapper for pomegranate.distributions.IndependentComponentsDistribution

import sys

import numpy as np
from pomegranate.distributions import IndependentComponentsDistribution as ICD

import chippr
from chippr import defaults as d
from chippr import utils as u

class multi_dist(object):

    def __init__(self, funcs):
        """
        A multidimensional freeform probability distribution object for
        independent dimensions

        Parameters
        ----------
        funcs: list, chippr.gauss or chippr.discrete or chippr.gmix objects
            1D functions, one per dimension
        """
        self.funcs = funcs
        self.dims = len(funcs)
        # for d in range(self.dims):
        #     print('multi dist '+str((d, type(self.funcs[d]))))
        self.funcs = [func.dist for func in self.funcs]
        self.dist = ICD(self.funcs)

    def pdf(self, points):
        return self.evaluate(points)

    def evaluate_one(self, point):
        """
        Function to evaluate the probability at a point in multidimensional
        space

        Parameters
        ----------
        point: numpy.ndarray, float
            coordinate at which to evaluate the probability

        Returns
        -------
        prob: float
            probability associated with point
        """
        # prob = 1.
        # for d in range(self.dims):
        #     prob *= self.funcs[d].evaluate_one(point[d])
        prob = self.dist.probability(point)
        return prob

    def evaluate(self, points):
        """
        Function to evaluate the probability at a point in multidimensional
        space

        Parameters
        ----------
        points: numpy.ndarray, float
            coordinates at which to evaluate the probability

        Returns
        -------
        probs: float
            probabilities associated with points
        """
        # if len(np.shape(points.T)) == 1:
        #     return self.evaluate_one(points)
        # probs = np.ones(len(points))
        # points = points.T
        # for d in range(self.dims):
        #     probs *= self.funcs[d].evaluate(points[d])
        probs = self.dist.probability(points)
        return probs

    def sample_one(self):
        """
        Function to take one sample from independent distributions

        Returns
        -------
        point: numpy.ndarray, float
            single sample from independent distributions
        """
        # point = np.zeros(self.dims)
        # for d in range(self.dims):
        #     point[d] = self.funcs[d].sample_one()
        point = self.dist.sample(1)
        return point

    def sample(self, n_samps):
        """
        Function to sample from multivariate probability distribution with independent components

        Parameters
        ----------
        n_samps: positive int
            number of samples to take

        Returns
        -------
        points: ndarray, float
            array of n_samps samples from independent distributions
        """
        # print('multi_dist trying to sample '+str(n_samps)+' from '+str(self.dist))
        # points = np.array([self.sample_one() for n in range(n_samps)])
        points = np.array([self.dist.sample() for n in range(n_samps)])
        # print('multi_dist sampled '+str(n_samps)+' from '+str(self.dist))
        return points
