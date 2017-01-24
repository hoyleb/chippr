import numpy as np

import chippr
from chippr import *

# I will write another script to automate this process. . .
tru_amps = np.array([0.20, 0.35, 0.55])
tru_means = np.array([0.5, 0.2, 0.75])
tru_sigmas = np.array([0.4, 0.2, 0.1])

tru_nz = gmix(tru_amps, tru_means, tru_sigmas, limits=(0., 1.))

N =10**4

tru_zs = tru_nz.sample(N)

params = 'params.txt'
params = sim_utils.ingest(params)

bin_ends = np.array([0., 1.])
weights = np.array([1.])

int_prior = discrete(bin_ends, weights)

posteriors = catalog(params)
output = posteriors.create(tru_zs, int_prior)

data = np.exp(output['log_interim_posteriors'])

saved_location = 'data.txt'
posteriors.write(saved_location)
