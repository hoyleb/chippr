import numpy as np

def mean(population):
    """
    Calculates the mean of a population

    Parameters
    ----------
    population: np.array, float
        population over which to calculate the mean

    Returns
    -------
    mean: np.array, float
        mean value over population
    """
    shape = np.shape(population)
    flat = population.reshape(np.prod(shape[:-1]), shape[-1])
    mean = np.mean(flat, axis=0)
    return mean
