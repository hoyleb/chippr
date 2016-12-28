import numpy as np

import chippr
from chippr import sim_utils as su

class interim_prior(object):
    def __init__(self, params):
        """
        Interim prior object for simulating individual probability distributions

        Parameters
        ----------
        params: dict or string
            dict containing keywords and values for nontrivial interim priors or string containing input parameter text file
        """
        self.in_dict = su.int_pr_params(params)

        self.pr_type = self.in_dict['int_pr']

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
