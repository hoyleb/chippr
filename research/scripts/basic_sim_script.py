import numpy as np
import os

import chippr
from chippr import *

# I will write another script to automate this process. . .
true_amps = np.array([0.20, 0.35, 0.55])
true_means = np.array([0.5, 0.2, 0.75])
true_sigmas = np.array([0.4, 0.2, 0.1])

true_nz = chippr.gmix(true_amps, true_means, true_sigmas, limits=(0., 1.))

N =10**4

true_zs = true_nz.sample(N)

result_dir = os.path.join('..', 'results')
name_file = 'which_tests.txt'

with open(name_file) as tests_to_run:
    for test_name in tests_to_run:
        params = test_name + '.txt'
        params = chippr.sim_utils.ingest(params)

        bin_ends = np.array([params['bin_min'], params['bin_max']])
        weights = np.array([1.])

        interim_prior = chippr.discrete(bin_ends, weights)

        test_dir = os.path.join(resut_dir, test_name)
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        os.makedirs(test_dir)

        posteriors = chippr.catalog(params, loc=test_dir)
        output = posteriors.create(true_zs, interim_prior)

        data = np.exp(output['log_interim_posteriors'])

        posteriors.write()
