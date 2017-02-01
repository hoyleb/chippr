"""
util-mcmc module defines handy tools for MCMC
"""

import sys
import numpy as np
import statistics
import cPickle as cpkl
import scipy as sp

class path(object):
    """
    path object takes templates of path style and variables for it and makes os.path objects from them
    """
    def __init__(self, path_template, filled = None):
        self.path_template = path_template
        if filled is None:
            self.filled = {}
        else:
            self.filled = filled

    def construct(self, **args):
        """actually constructs the final path, as a string.  Optionally takes in any missing parameters"""
        nfilled = self.filled.copy()
        nfilled.update(args)
        return self.path_template.format(**nfilled)

    def fill(self, **args):
        """fills any number of missing parameters, returns new object"""
        dct = self.filled.copy()
        dct.update(args)
        return path(self.path_template, dct)
