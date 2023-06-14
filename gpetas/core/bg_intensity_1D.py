import numpy as np
from pypolyagamma import PyPolyaGamma
from scipy.linalg import solve_triangular
import gpetas
import matplotlib.pyplot as plt


class BG_Intensity_Sampler():

    def __init__(self, S_borders, X, T, cov_params, lmbda_star=None, X_grid=None, noise=1e-4,
                 mu_upper_bound=None, std_factor=1., mu_nu0=None, mu_length_scale=None, sigma_proposal_hypers=None,
                 kth_sample_obj=None,ard=None):
        """

        :param sigma_proposal_hypers:
        :type sigma_proposal_hypers:
        :type noise: jitter for GP modeling
        """
        self.S_borders = S_borders
        self.S = S_borders[:, 1] - S_borders[:, 0]
        self.R = np.prod(self.S)
        self.D = S_borders.shape[0]
        self.cov_params = cov_params
        self.acc_hyperparams_per_iter = None
        self.sigma_proposal_hypers = sigma_proposal_hypers
        self.noise = noise
        self.T = T
        self.X = X
        self.S_0_idx = None
        self.S_rest_idx = None
        self.S_0 = None
        self.N = 0
        self.M = 0
        self.ard = ard
        self._set_prior_upper_bound(mu_upper_bound=mu_upper_bound, std_factor=std_factor)
        if lmbda_star is None:
            self.lmbda_star = self.alpha0 / self.beta0
        else:
            self.lmbda_star = lmbda_star
        self.lmbda_star_start = self.lmbda_star
        self.mu_nu0 = mu_nu0
        self.mu_length_scale = mu_length_scale

        self.pg = PyPolyaGamma()
        self.sample_unthinned_set()
        self.g_unthinned = np.zeros(self.Pi_unthinned.shape[0])
        self.g = np.zeros(self.N + self.M)
        self.sample_latent_poisson()
        self.g_X = np.zeros(self.X.shape[0])
        self.X_grid = X_grid
        self.iteration = 0
        if kth_sample_obj is not None:
            mu_X = gpetas.some_fun.get_grid_data_for_a_point(grid_values_flattened=kth_sample_obj.mu_grid,
                                                             points_xy=self.X,
                                                             X_borders=self.S_borders)
            self.g_X = -np.log(self.lmbda_star / mu_X - 1.)
            mu_Pi_latent = gpetas.some_fun.get_grid_data_for_a_point(grid_values_flattened=kth_sample_obj.mu_grid,
                                                                     points_xy=self.Pi_unthinned,
                                                                     X_borders=self.S_borders)
            self.g_unthinned = -np.log(self.lmbda_star / mu_Pi_latent - 1.)
            self.N = len(self.g_X)
            self.M = 0
            self.g = np.zeros(self.N + self.M)
            self.sample_latent_poisson()

    def _set_prior_upper_bound(self, mu_upper_bound, std_factor):
        """
        Prior is constructed that the mean is the maximum expected count from a histogramm of all (!) data.
        The standard deviation is then given by std_factor * mean
        """
        if self.D == 2:
            if mu_upper_bound is None:
                counts, x_bins, y_bins = np.histogram2d(self.X[:, 0], self.X[:, 1], bins=10)
                dx, dy = x_bins[1] - x_bins[0], y_bins[1] - y_bins[0]
                mu_upper_bound = np.amax(counts) / dx / dy / self.T
        if self.D == 1:
            counts, x_bins = np.histogram(self.X, bins=10)
            dx = x_bins[1] - x_bins[0]
            mu_upper_bound = np.amax(counts) / dx  # / self.T

        self.beta0 = 1. / std_factor ** 2. / mu_upper_bound
        self.alpha0 = 1. / std_factor ** 2.
        self.prior_mu_upper_bound = mu_upper_bound
        self.prior_std_factor_upper_bound = std_factor
        print('START: hyper prior: self.prior_mu_upper_bound  =', self.prior_mu_upper_bound)
        print('START: self.alpha0                             =', self.alpha0)
        print('START: self.beta0                              =', self.beta0)
        print('START: self.prior_std_factor_upper_bound       =', self.prior_std_factor_upper_bound)

    def sample(self, branching):
        self.set_new_branching(branching)
        self.sample_latent_poisson()
        self.sample_marks()
        self.sample_lmbda()
        self.update_kernels()
        self.sample_g()
        self.sample_g_at_all_points()
        if self.iteration % 1 == 0:
            self.sample_kernel_params()
            print('==== sampling cov params====')
            print('BG self.iteration=', self.iteration)
            print('cov params=', self.cov_params)

        self.iteration += 1

    def sample_g_at_all_points(self):
        self.sample_unthinned_set()
        X = np.empty([len(self.S_rest_idx) + self.Pi_unthinned.shape[0] +
                      self.X_grid.shape[0], self.D])
        X[:len(self.S_rest_idx)] = self.X[self.S_rest_idx]
        X[len(self.S_rest_idx):-self.X_grid.shape[0]] = self.Pi_unthinned
        X[-self.X_grid.shape[0]:] = self.X_grid
        g_all = self.sample_from_cond_GP(X)
        self.g_X = np.empty(self.X.shape[0])
        self.g_X[self.S_0_idx] = self.g[:self.N]
        self.g_X[self.S_rest_idx] = g_all[:len(self.S_rest_idx)]
        self.g_unthinned = g_all[len(self.S_rest_idx):-self.X_grid.shape[0]]
        self.g_grid = g_all[-self.X_grid.shape[0]:]

    def set_new_branching(self, branching):
        self.S_0_idx = np.where(branching == 0)[0]
        self.S_rest_idx = np.where(branching != 0)[0]
        self.S_0 = self.X[self.S_0_idx]
        self.N = self.S_0.shape[0]
        self.X_all = np.empty([self.N + self.M, self.D])
        self.X_all[:self.N] = self.S_0
        self.X_all[self.N:] = self.Pi
        g_M = self.g[-self.M:]
        self.g_M = g_M
        self.g = np.empty(self.N + self.M)
        self.g[:self.N] = self.g_X[self.S_0_idx]
        # self.g[self.N:] = g_M
        # new: adjustment if M==0:
        if self.M > 0:
            self.g[self.N:] = g_M

    def sample_unthinned_set(self):
        if self.D == 2:
            expected_num_events = self.R * self.T * self.lmbda_star
        if self.D == 1:
            expected_num_events = self.T * self.lmbda_star
        num_events = np.random.poisson(expected_num_events, 1)[0]
        # self.Pi_unthinned = self.base_rate.sample(num_events)
        self.Pi_unthinned = np.random.rand(num_events, self.D) * self.S[None] + \
                            self.S_borders[:, 0][None]

    def cov_func(self, x, x_prime, only_diagonal=False, cov_params=None):
        """ Computes the covariance functions between x and x_prime.

        :param x: numpy.ndarray [num_points x D]
            Contains coordinates for points of x
        :param x_prime: numpy.ndarray [num_points_prime x D]
            Contains coordinates for points of x_prime
        :param only_diagonal: bool
            If true only diagonal is computed (Works only if x and x_prime
            are the same, Default=False)

        :return: numpy.ndarray [num_points x num_points_prime]
            Kernel matrix.
        """
        if cov_params is None:
            theta_1, theta_2 = self.cov_params[0], self.cov_params[1]
        else:
            theta_1, theta_2 = cov_params[0], cov_params[1]
        if only_diagonal:
            return theta_1 * np.ones(x.shape[0])

        else:
            x_theta2 = x / theta_2
            xprime_theta2 = x_prime / theta_2
            h = np.sum(x_theta2 ** 2, axis=1)[:, None] - 2. * np.dot(
                x_theta2, xprime_theta2.T) + \
                np.sum(xprime_theta2 ** 2, axis=1)[None]
            return theta_1 * np.exp(-.5 * h)

    def sample_marks(self):
        self.marks = np.empty(self.N + self.M)
        # TODO: Could be parallized?
        for imark in range(self.M + self.N):
            self.marks[imark] = self.pg.pgdraw(1., self.g[imark])

    def sample_lmbda(self):

        alpha = self.N + self.M + self.alpha0
        if self.D == 2:
            beta = self.R * self.T + self.beta0
        if self.D == 1:
            beta = self.T + self.beta0
        shape = 1. / beta
        self.lmbda_star = np.random.gamma(alpha, shape)

    def sample_g(self):
        Sigma_g_inv = np.diag(self.marks) + self.K_inv
        L_inv = np.linalg.cholesky(Sigma_g_inv + self.noise * np.eye(
            self.K.shape[0]))
        L = solve_triangular(L_inv, np.eye(self.L_inv.shape[0]),
                             lower=True, check_finite=False)
        Sigma_g = L.T.dot(L)
        u = np.empty(self.N + self.M)
        u[:self.N] = .5
        u[self.N:] = -.5
        mu_g = np.dot(Sigma_g, u)
        rand_nums = np.random.randn(self.N + self.M)
        self.g = mu_g + np.dot(L.T, rand_nums)

    def sample_latent_poisson(self):
        # Without lmbda_star
        inv_intensity = 1. / (1. + np.exp(self.g_unthinned))
        rand_nums = np.random.rand(len(self.Pi_unthinned))
        thinned_idx = np.where(inv_intensity >= rand_nums)[0]
        self.M = len(thinned_idx)
        self.Pi = self.Pi_unthinned[thinned_idx]
        g_N = self.g[:self.N]
        self.g = np.empty(self.N + self.M)
        self.g[:self.N] = g_N
        self.g[self.N:] = self.g_unthinned[thinned_idx]
        self.X_all = np.empty([self.N + self.M, self.D])
        self.X_all[:self.N] = self.S_0
        self.X_all[self.N:] = self.Pi

    def sample_from_cond_GP(self, xprime):

        k = self.cov_func(self.X_all, xprime)
        mean = k.T.dot(self.K_inv.dot(self.g))
        kprimeprime = self.cov_func(xprime, xprime)
        # var = (kprimeprime - k.T.dot(self.K_inv.dot(k))).diagonal()
        # gprime = mean + numpy.sqrt(var)*numpy.random.randn(xprime.shape[0])
        Sigma = (kprimeprime - k.T.dot(self.K_inv.dot(k)))
        L = np.linalg.cholesky(Sigma + self.noise * np.eye(Sigma.shape[0]))
        gprime = mean + np.dot(L.T, np.random.randn(xprime.shape[0]))
        return gprime

    def update_kernels(self):
        self.K = self.cov_func(self.X_all, self.X_all)
        self.L = np.linalg.cholesky(self.K + self.noise * np.eye(
            self.K.shape[0]))
        self.L_inv = solve_triangular(self.L, np.eye(self.L.shape[0]),
                                      lower=True, check_finite=False)
        self.K_inv = self.L_inv.T.dot(self.L_inv)

    def sample_kernel_params(self):
        """ Samples kernel parameters with Metropolis sampling.

        :return:
        """

        logp_old = self.compute_kernel_param_prop(self.K_inv, self.L, self.cov_params[0], self.cov_params[1])
        theta1 = np.exp(np.log(self.cov_params[0]) + self.sigma_proposal_hypers * np.random.randn(1))
        theta2 = np.exp(np.log(self.cov_params[1]) + self.sigma_proposal_hypers * np.random.randn(
            self.D))
        if self.ard is None:
            theta2[1]=theta2[0]
        K = self.cov_func(self.X_all, self.X_all, cov_params=[theta1, theta2])
        K += self.noise * np.eye(K.shape[0])
        L = np.linalg.cholesky(K)
        L_inv = solve_triangular(L, np.eye(L.shape[0]), lower=True,
                                 check_finite=False)
        K_inv = L_inv.T.dot(L_inv)
        logp_new = self.compute_kernel_param_prop(K_inv, L, theta1, theta2)
        rand_num_accept = np.random.rand(1)
        accept_p = np.exp(logp_new - logp_old)

        self.acc_hyperparams_per_iter = 0
        if rand_num_accept < accept_p:
            self.cov_params = [theta1, theta2]
            self.K = K
            self.L = L
            self.L_inv = L_inv
            self.K_inv = K_inv
            self.acc_hyperparams_per_iter = 1

    def compute_kernel_param_prop(self, K_inv, L, nu0, length_scales, mu_nu0=5, length_scale_factor=.1):
        """ Computes the log probability (plus constant) given Kernel
        parameters.

        :param K_inv: Inverse kernel matrix.
        :type K_inv: numpy.ndarray [num_of_points x num_of_points]
        :param L: Cholesky decomposition of Kernel matrix.
        :type L: numpy.ndarray [num_of_points x num_of_points]
        :return: log probability plus constant
        :rtype: float
        """
        if self.mu_nu0 is not None:
            mu_nu0 = self.mu_nu0
        if self.mu_length_scale is not None:
            mu_length_scale = self.mu_length_scale
        else:
            mu_length_scale = length_scale_factor * self.S
        logp = -.5 * self.g.T.dot(K_inv.dot(self.g)) - \
               np.sum(np.log(L.diagonal())) - nu0 / mu_nu0 - \
               np.sum(length_scales / mu_length_scale)
        self.hyper_prior_length_scale_factor = length_scale_factor
        self.hyper_prior_mu_length_scale = mu_length_scale
        self.hyper_prior_mu_nu0 = mu_nu0
        if self.iteration == 0:
            print('self.hyper_prior_length_scale_factor =', self.hyper_prior_length_scale_factor)
            print('self.hyper_prior_mu_length_scale     =', self.hyper_prior_mu_length_scale)
            print('self.hyper_prior_mu_nu0              =', self.hyper_prior_mu_nu0)
        return logp
