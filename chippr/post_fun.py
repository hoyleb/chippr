import numpy as np

class post_fun(object):
    def __init__(self, lik_fun, int_pr_fun):
        """
        Object defining a posterior probability distribution in binned parametrization

        Parameters
        ----------
        lik_fun: lik_fun object
            likelihood object
        int_pr_fun: int_pr_fun object
            interim prior object
        """
        self.lik = lik_fun
        self.int_pr = int_pr_fun

        self.binned = False

    def setup_bins(self, x_min=0., x_max=1., n_coarse=10, n_fine=10):
        """
        Function to establish binning for posterior probability distribution parametrization

        Parameters
        ----------
        x_min: float, optional
            minimum of binning range
        x_max: float, optional
            maximum of binning range
        n_coarse: positive int, optional
            number of coarse bins
        n_fine: positive int, optional
            number of fine bins within each coarse bin
        """
        self.x_min = x_min
        self.x_max = x_max
        self.n_coarse = n_coarse
        self.n_fine = n_fine

        self.n_fine_tot = self.n_coarse*self.n_fine
        self.x_range = self.x_max-self.x_min

        self.dx_coarse = self.x_range/self.n_coarse
        self.dx_fine = self.x_range/self.n_fine_tot

        self.x_coarse = np.arange(self.x_min+0.5*self.dx_coarse, self.x_max, self.dx_coarse)
        self.x_fine = np.arange(self.x_min+0.5*self.dx_fine, self.x_max, self.dx_fine)

        self.binned = True

    def bin_post(self, x):
        """
        Function to calculate posterior probability distribution on coarse grid

        Parameters
        ----------
        x: float
            'observed' value defining likelihood function

        Returns
        -------
        cps: ndarray, float
            probabilities at each coarse grid point
        """

        if self.binned == False:
            self.setup_bins()

        ps = self.int_pr.evaluate(self.x_fine)*self.lik.evaluate(x, self.x_fine)
        ps /= np.sum(ps) * self.dx_fine
        cps = np.array([np.sum(ps[k*self.n_fine:(k+1)*self.n_fine])*self.dx_fine for k in range(self.n_coarse)])
        cps = cps/self.dx_coarse
        return cps

    def make_rectangular(self, xs):
        """
        Function to calculate many posterior probability distributions on coarse grid

        Parameters
        ----------
        xs: ndarray, float
            'observed' values defining likelihood functions

        Returns
        -------
        ps: ndarray, float
            posterior probabilities at each point on grid for each input value
        """
        ps = np.zeros((len(xs), self.n_coarse))
        for n, x in enumerate(xs):
            ps[n] = self.bin_post(x)
        return ps