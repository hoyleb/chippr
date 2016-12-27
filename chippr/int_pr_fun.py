import numpy as np

import chippr
from chippr import sim_utils as su

class int_pr_fun(object):
    def __init__(self, params):
        """
        Interim prior object for simulating individual probability distributions

        Parameters
        ----------
        in_dict: dict
            dict containing keywords and values for nontrivial interim priors
        """
        self.in_dict = su.int_pr_params(params)

        self.pr_type = self.in_dict['int']

    def evaluate(self, xs):
        """
        Function evaluating interim prior at values

        Parameters
        ----------
        xs: ndarray, float
            'true' values at which to evaluate interim prior

        Returns
        -------
        ps: ndarray, float
            probabilities associated with values xs
        """
        if self.pr_type == 'flat':
            ps = np.ones_like(xs)

        return ps
