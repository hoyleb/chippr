import numpy as np

import chippr
from chippr import sim_utils as su
from chippr import gauss

class likelihood(object):
    def __init__(self, params):
        """
        Likelihood object for simulating individual probability distributions

        Parameters
        ----------
        in_dict: dict or string
            dict containing keywords and values for nontrivial likelihoods or string containing location of
        """
        self.in_dict = su.lf_params(params)

        self.true_sigma = self.in_dict['sigma']

    def sample(self, x_in):
        """
        Function to turn a true value into a sampled value

        Parameters
        ----------
        x_in: float
            'true' value to be sampled

        Returns
        -------
        x_out: float
            sampled value derived from true value
        """
        fun = gauss(x_in, self.true_sigma)
        x_out = fun.sample_one()
        return x_out

    def evaluate_one(self, x, xs):
        """
        Function to evaluate likelihood function

        Parameters
        ----------
        x: float
            'observed' value at which to evaluate likelihood function
        xs: ndarray, float
            'true' values defining likelihood function

        Returns
        -------
        ps: ndarray, float
            probabilities at xs
        """
        n_vals = len(xs)
        funs = [gauss(xs[n], self.true_sigma) for n in range(n_vals)]
        ps = np.array([funs[n].evaluate_one(x) for n in range(n_vals)])
        return ps
