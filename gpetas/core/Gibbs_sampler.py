import numpy as np
import scipy as sc
from scipy import stats
import math
import time
import pickle

import gpetas


class GS_ETAS():
    """ Gibbs sampler for the Epidemic Type Aftershock Sequence (ETAS) model.
    """

    def __init__(self, data_obj, setup_obj=None, burnin=None, num_samples=None, thinning=None,
                 stat_background=False, kth_sample_obj=None, case_name=None,
                 MH_proposals_offspring=None, MH_cov_empirical_yes=None):
        """
        :param MH_cov_empirical_yes:
        :type MH_cov_empirical_yes:
        :param MH_proposals_offspring:
        :type MH_proposals_offspring:
        :param kth_sample_obj:
        :type kth_sample_obj:
        :param case_name:
        :type case_name:
        :param data: np.array [N x 4]
            Array with the observed catalogue. Rows are observations, columns
            are features (time, magnitude, x_position, y_position)
        :param T: float
            Length of time window that is considered.
        :param S_borders: np.array [2 x 2]
            Borders of the spatial window.
        :param burnin: int
            Number of samples that are used for the burn-in period. (
            Default=1000).
        :param num_samples: int
            Number of samples that are collected to get the posterior. (
            Default=2000)
        :param stat_background: bool
            If true, the model assumes a stationary background intensity.
        :param mu_x: float
            Start value of the background intensity. (Default=1.)
        """
        self.data_obj = data_obj
        self.setup_obj = setup_obj
        self.case_name = case_name
        if thinning is None:
            thinning = 1
        self.thinning = thinning
        self.sigmoid = lambda x: 1. / (1. + np.exp(-x))

        # dimension of bg: time only (1D) or space only (2D,default)
        dim = self.setup_obj.dim
        if dim is None:
            dim = 2
        self.dim = dim
        print('dim', dim)

        # preparing training data
        self.T_borders_training = data_obj.domain.T_borders_training
        self.T_borders_all = data_obj.domain.T_borders_all
        self.Tabs_training = np.diff(self.T_borders_training)
        self.T = self.Tabs_training  # old var name
        self.utm_yes = None
        self.X_borders = data_obj.domain.X_borders
        self.X_borders_NN = data_obj.domain.X_borders - np.array(
            [[data_obj.domain.X_borders[0, 0], data_obj.domain.X_borders[0, 0]],
             [data_obj.domain.X_borders[1, 0], data_obj.domain.X_borders[1, 0]]])
        self.S_borders = self.X_borders_NN  # old var name
        self.S = self.X_borders[:, 1] - self.X_borders[:, 0]
        self.Xabs = np.prod(self.S)
        self.R = self.Xabs  # old var name

        # initialize class 'data' (training) used for inference
        self._readin_data_training(data_obj)

        # setup sampler object
        if setup_obj is None:
            setup_obj = gpetas.setup_Gibbs_sampler.setup_sampler(data_obj)
        self.setup_obj = setup_obj
        cov_params = self.setup_obj.cov_params

        # utm issue
        if self.setup_obj.utm_yes is not None:
            data_obj.domain.X_borders = np.copy(data_obj.domain.X_borders_UTM_km)
            data_obj.data_all.positions = np.copy(data_obj.data_all.positions_UTM_km)
            print('Note: all positions are in UTM now!')
            self.utm_yes = self.setup_obj.utm_yes

        # setting sampler related parameters
        self.burnin = burnin
        self.num_samples = num_samples
        self.MH_proposals_offspring = MH_proposals_offspring
        self.MH_cov_empirical_proposal = MH_cov_empirical_yes
        mu0_start = self.setup_obj.mu0_start
        if mu0_start is None:
            mu0_start = (len(self.data.times) / 2. / self.R / self.T).item()
        self.mu0_start = mu0_start
        self.lmbda_star_start = mu0_start  # len(data[:,0])/self.R/self.T #2. * np.mean(mu0)
        self.stat_background = stat_background
        self.Sigma_empirical = None
        print('UPPER BOUND: lambda_star_start = ', self.lmbda_star_start)

        # offspring
        self.l_tilde = None
        self.theta_start = None
        self.acc_offspring_per_iter = None
        self.sigma_proposal_offspring_params = np.copy(self.setup_obj.sigma_proposal_offspring_params)

        # parameters offsprings
        self._initialize_parameters()

        # init BG_sampler
        mu_upper_bound = self.setup_obj.mu_upper_bound
        std_factor = self.setup_obj.std_factor
        mu_nu0 = self.setup_obj.mu_nu0
        mu_length_scale = self.setup_obj.mu_length_scale
        self.mu0_grid = self.setup_obj.mu0_grid
        if self.mu0_grid is None:
            self.mu0_grid = np.mean(mu0_start) * np.ones(self.setup_obj.X_grid.shape[0])

        if kth_sample_obj is not None:
            # offspring
            self.setup_obj.spatial_offspring = np.copy(kth_sample_obj.spatial_offspring)
            self.setup_obj.theta_start_Kcpadgqbm0[:len(kth_sample_obj.theta_phi)] = kth_sample_obj.theta_phi
            self._initialize_parameters()
            # bg
            self.mu0 = gpetas.some_fun.get_grid_data_for_a_point(grid_values_flattened=kth_sample_obj.mu_grid,
                                                                 points_xy=self.data.positions,
                                                                 X_borders=self.X_borders_NN)
            self.lmbda_star_start = np.copy(kth_sample_obj.lambda_bar)
            cov_params = setup_obj.kth_sample_obj.cov_params.copy()
            self.mu0_grid = np.copy(kth_sample_obj.mu_grid)
            self.sample_branching_structure()
            kth_sample_obj.branching = np.copy(self.branching)

        # bg only time --> dim=1
        if dim == 1:
            # cov_params 1D
            nu_lengthscale_start = gpetas.some_fun.silverman_scott_rule_d(self.data.times)
            cov_params = [np.array([5.]), np.array([nu_lengthscale_start])]
            self.cov_params = cov_params
            self.cov_params_start = self.cov_params
            self.setup_obj.cov_params = cov_params
            self.setup_obj.cov_params_start = cov_params
            print(cov_params)
            # lambda_bar start 1D
            self.lmbda_star_start = (len(self.data.times) / 2. / self.T).item()
            print(self.lmbda_star_start)
            # X_grid_NN in 1D
            S_borders_1D = self.T_borders_training.reshape([1, 2])

            self.bg_sampler = gpetas.bg_intensity_1D.BG_Intensity_Sampler(
                S_borders=S_borders_1D,
                X=self.data.times.reshape([-1, 1]),
                T=self.Tabs_training,
                cov_params=cov_params,
                lmbda_star=self.lmbda_star_start,
                X_grid=self.setup_obj.X_grid_NN,
                mu_upper_bound=mu_upper_bound,
                std_factor=std_factor,
                mu_nu0=mu_nu0,
                mu_length_scale=mu_length_scale,
                sigma_proposal_hypers=self.setup_obj.sigma_proposal_hypers,
                kth_sample_obj=kth_sample_obj)
        # bg only space --> dim=2
        if dim == 2:
            self.bg_sampler = gpetas.bg_intensity_1D.BG_Intensity_Sampler(S_borders=self.X_borders_NN,
                                                                          X=self.data.positions,
                                                                          T=self.Tabs_training, cov_params=cov_params,
                                                                          lmbda_star=self.lmbda_star_start,
                                                                          X_grid=self.setup_obj.X_grid_NN,
                                                                          mu_upper_bound=mu_upper_bound,
                                                                          std_factor=std_factor, mu_nu0=mu_nu0,
                                                                          mu_length_scale=mu_length_scale,
                                                                          sigma_proposal_hypers=
                                                                          self.setup_obj.sigma_proposal_hypers,
                                                                          kth_sample_obj=kth_sample_obj)
        # some info
        self.case_name = np.copy(self.data_obj.case_name)
        if self.case_name is None:
            self.case_name = 'case_XX'
        self.save_data_init_info()

    def sample(self):
        """ Main function for running the sampler for the ETAS model.
        """
        # Gibbs sampler
        while self.iteration < self.burnin + self.num_samples:
            self.sample_branching_structure()
            self.sample_offspring_parameters()
            self.sample_background_intensity()
            self.save_results_and_info()

        # save results of the sampler at the end
        self.save_data['track_vars'] = self.track_data_per_iter
        file = open(self.setup_obj.outdir + "/GS_save_data_%s.bin" % self.case_name, "wb")
        pickle.dump(self.save_data, file)
        file.close()

    def sample_background_intensity(self):
        """ Samples the background intensity.
        """
        print(self.stat_background)
        if self.stat_background:  # True
            # Gamma prior of static background
            # params for gamma dist are here prior_mean,prior_coefficient_of_variation
            # i.e., for each param a pair [m,c], if c=1 prior becomes an exp.distribution
            # alpha_0 = 1/c**2, beta_0 = alpha_0/prior_mean
            # often used: only in time: prior(mu) \sim Gamma(alpha=0.1,beta=0.1) --> m=1,c=1/np.sqrt(0.1)
            # samples stationary background
            # shape = self.N - len(np.nonzero(self.branching))
            # scale = 1. / (self.R * self.T)
            m, c = self.setup_obj.prior_static_bg_params
            alpha_0 = 1. / c ** 2.
            beta_0 = alpha_0 / m
            shape = self.N - len(np.nonzero(self.branching))
            scale = 1. / (self.Tabs_training + beta_0)
            mu0_new = np.random.gamma(shape, scale)
            self.mu0[:] = mu0_new/self.Xabs
        else:
            # samples the (spatially) inhomogeneous background intensity.
            self.bg_sampler.sample(self.branching)
            self.mu0[:] = self.bg_sampler.lmbda_star \
                          / (1 + np.exp(-self.bg_sampler.g_X))
            if self.bg_sampler.X_grid is not None:
                self.mu0_grid[:] = self.bg_sampler.lmbda_star \
                                   / (1 + np.exp(-self.bg_sampler.g_grid))

    def _readin_data_training(self, data_obj):
        """
        Reads data into an object structure and shifts origin of X domain to (0,0)
        :param data_obj:
        :return:
        """
        idx = np.where((data_obj.data_all.times >= self.T_borders_training[0]) & (
                data_obj.data_all.times <= self.T_borders_training[1]))
        self.data = data_structure()
        self.data.times = data_obj.data_all.times[idx]
        self.data.magnitudes = data_obj.data_all.magnitudes[idx]
        self.data.positions = data_obj.data_all.positions[np.squeeze(idx), :]
        self.data.positions[:, 0] = self.data.positions[:, 0] - self.X_borders[0, 0]
        self.data.positions[:, 1] = self.data.positions[:, 1] - self.X_borders[1, 0]
        self.N = len(np.squeeze(idx))
        self.iteration = 0
        self.save_data = {'mu0_data': [], 'mu_grid': [], 'N_S0': [], 'N_Stilde': [],
                          'l_tilde': [], 'theta_tilde': [], 'l_theta_k': [], 'l_mu_givenZ_at_S0_intensity_part': [],
                          'X': [], 'g': [], 'lambda_bar': [], 'cov_params': [], 'cov_params_theta': [],
                          'cov_params_nu1': [],
                          'cov_params_nu2': [], 'M': [], 'acc_offspring_per_iter': [],
                          'time': [], 'bm_params': [], 'gp_mu': [],
                          'int_riemann_approx': [],
                          'Campbell_1_log_I_k': [],
                          'Campbell_2_log_I_k': [],
                          'lnl_sample_overall': [],
                          'X_grid': [],
                          'data_obj': [],
                          'theta_true_Kcpadgq': [],
                          'branching': []}
        self.track_data_per_iter = {'theta_tilde': [],
                                    'acc_offspring_per_iter': [],
                                    'acc_hyperparams_per_iter': []}

    def _initialize_parameters(self):
        """ Initializes all necessary parameters of the sampler.
        Recognizes which spatial kernel is used:
        (1)  Gaussian kernel (short range decay) with stdev. param: D_gauss (DEFAULT) if given
        (2a) Power Law kernel (long range decay) with params: D_pwl, gamma_pwl, q_pwl
        (2b) Power Law kernel using an empirical relation for rupture length scaling with magnitude (wells,coppersmith,1994)
                                                 with params: D_RL_pwl, gamma_RL_pwl, q_RL_pwl
        """
        # new first part
        # parameters offsprings
        K, c, p, m_alpha, D_gauss, D_pwl, gamma_pwl, q_pwl, D_RL_pwl, gamma_RL_pwl, q_RL_pwl = np.empty(11) * np.nan
        m_beta = self.setup_obj.m_beta
        m0 = self.setup_obj.m0
        if self.setup_obj.spatial_offspring == 'G':
            K, c, p, m_alpha, D_gauss, g, q, m_beta, m0 = self.setup_obj.theta_start_Kcpadgqbm0
            D_pwl, gamma_pwl, q_pwl = None, None, None
            D_RL_pwl, gamma_RL_pwl, q_RL_pwl = None, None, None
        if self.setup_obj.spatial_offspring == 'P':
            D_gauss = None
            D_RL_pwl, gamma_RL_pwl, q_RL_pwl = None, None, None
            K, c, p, m_alpha, D_pwl, gamma_pwl, q_pwl, m_beta, m0 = self.setup_obj.theta_start_Kcpadgqbm0
        if self.setup_obj.spatial_offspring == 'R':
            D_gauss = None
            D_pwl, gamma_pwl, q_pwl = None, None, None
            K, c, p, m_alpha, D_RL_pwl, gamma_RL_pwl, q_RL_pwl, m_beta, m0 = self.setup_obj.theta_start_Kcpadgqbm0

        # old second part
        self.mu0 = self.mu0_start
        if np.isscalar(self.mu0_start):
            self.mu0 = self.mu0 * np.ones(self.N)  # background intensity (at data start)
        if self.mu0 is None:
            self.mu0 = self.N / self.R / self.T / 2. * np.ones(self.N)  # background intensity (start?)
        self.branching = np.zeros(self.N, dtype=int)  # branching structure

        self.theta = None
        if D_gauss is not None:
            self.spatial_kernel = 'Gauss with params: D_gauss'
            self.theta = np.array([K, c, p, m_alpha, D_gauss, m_beta, m0])
            self.theta_start = np.copy(self.theta)
            print('===================================================')
            print('INITIALIZATION of theta_tilde (offspring parameters)')
            print('   DEFAULT spatial kernel: ', self.spatial_kernel)
            print('[K, c, p, m_alpha, D_gauss, m_beta, m0]')
            print(self.theta)
            print('===================================================')
        else:
            if all(v is not None for v in [D_pwl, gamma_pwl, q_pwl]):
                self.spatial_kernel = 'Power Law with params: D_pwl,gamma_pwl,q_pwl'
                self.theta = np.array([K, c, p, m_alpha, D_pwl, gamma_pwl, q_pwl, m_beta, m0])
                self.theta_start = np.copy(self.theta)
                print('===================================================')
                print('INITIALIZATION of theta_tilde (offspring parameters)')
                print('     spatial kernel: ', self.spatial_kernel)
                print('[K, c, p, m_alpha, D_pwl, gamma_pwl, q_pwl, m_beta, m0]')
                print(self.theta)
                print('===================================================')
            else:
                if all(v is not None for v in [D_RL_pwl, gamma_RL_pwl, q_RL_pwl]):
                    self.spatial_kernel = 'Rupture Length Power Law with params: D_RL_pwl, gamma_RL_pwl, q_RL_pwl'
                    self.theta = np.array([K, c, p, m_alpha, D_RL_pwl, gamma_RL_pwl, q_RL_pwl, m_beta, m0])
                    self.theta_start = np.copy(self.theta)
                    print('===================================================')
                    print('INITIALIZATION of theta_tilde (offspring parameters)')
                    print('     spatial kernel: ', self.spatial_kernel)
                    print('[K, c, p, m_alpha, D_RL_pwl, gamma_RL_pwl, q_RL_pwl, m_beta, m0]')
                    print(self.theta)
                    print('===================================================')

        if self.theta is None:
            print('Error: at least one parameter for the spatial kernel is missing.')
            return None

    def sample_branching_structure(self):
        """ Samples the branching structure.

        Suggestion: Use self.branching that is an N-dimensional array,
        where n^th entry is the index, which is the event it belongs to?
        """
        Nobs = self.N
        tvec = self.data.times
        mvec = self.data.magnitudes
        xvec = self.data.positions[:, 0]
        yvec = self.data.positions[:, 1]
        zvec = np.zeros(Nobs) * np.nan
        P = np.zeros((Nobs, Nobs)) * np.nan
        P[0, 0] = 1.
        zvec[0] = 0.

        # params and SPATIAL KERNEL
        K, c, p, m_alpha, D, gamma, q, m_beta, m0 = np.empty(9) * np.nan
        if len(self.theta) == 7:
            K, c, p, m_alpha, D, m_beta, m0 = self.theta
        if len(self.theta) == 9:
            K, c, p, m_alpha, D, gamma, q, m_beta, m0 = self.theta

        i = 1
        for data_idx in range(1, Nobs):  # starts from second data point i=2 ()
            ti = tvec[data_idx]
            xi = xvec[data_idx]
            yi = yvec[data_idx]
            delta_t_ij = ti - tvec
            delta_t_ij = delta_t_ij[delta_t_ij > 0]
            self.delta_t_ij = delta_t_ij
            delta_x_ij = xi - xvec[0:i]
            delta_y_ij = yi - yvec[0:i]
            mj_i = mvec[0:len(delta_t_ij)]

            # s(x):
            # (1) GAUSS (short range)
            if len(self.theta) == 7:
                s_x_epart = (-1. / 2. * ((delta_x_ij) ** 2 + (delta_y_ij) ** 2)) / (
                        D ** 2 * np.exp(
                    m_alpha * (mj_i - m0)))
                s_x = 1. / (2 * np.pi * D ** 2 * np.exp(
                    m_alpha * (mj_i - m0))) * np.exp(
                    s_x_epart)
                # (2a) Power Law decay (long  range)
            if len(self.theta) == 9 and self.spatial_kernel[0] == 'P':
                s_x = (q - 1) / (np.pi * D ** 2 * np.exp(gamma * (mj_i - m0))) * (
                        1. + ((delta_x_ij) ** 2 + (delta_y_ij) ** 2) / (D ** 2 * np.exp(gamma * (mj_i - m0)))) ** (
                          -q)
                # (2b) Rupture Length Power Law decay (long  range)
            if len(self.theta) == 9 and self.spatial_kernel[0] == 'R':
                s_x = (q - 1) / (np.pi * D ** 2 * 10. ** (2 * gamma * (mj_i))) * (
                        1. + ((delta_x_ij) ** 2 + (delta_y_ij) ** 2) / (D ** 2 * 10. ** (2 * gamma * (mj_i)))) ** (
                          -q)

            # eq. (11)
            p_ijvec = K * np.exp(
                m_alpha * (mj_i - m0)) * 1. / (
                              delta_t_ij + c) ** p * s_x / \
                      (self.mu0[i] + np.sum(
                          K * np.exp(m_alpha * (
                                  mj_i - m0)) * 1. / (
                                  delta_t_ij + c) ** p * s_x))
            P[i, 0:i] = p_ijvec
            # eq. (10): diagonal is Pr(z_i=0) (being exogenous, immigrant)
            P[i, i] = 1. - np.sum(p_ijvec)

            # numerical issues if one gets below resolution of machine precission
            if (1. - np.sum(p_ijvec)) < 0.:
                if np.abs(1. - np.sum(p_ijvec)) < np.finfo(float).resolution:
                    print('1.-np.sum(p_ijvec) is smaller than eps', 1. - np.sum(p_ijvec))
                    P[i, i] = 0.
            self.Pii = P[i, i]
            self.Pi_line = P[i, 0:i + 1]

            # sample from multinomial with probabilities in P (each row)
            b_sample = np.random.multinomial(1, P[i, 0:i + 1])
            zvec[i] = int((np.arange(1, i + 2)[b_sample > 0]).item())
            if int((np.arange(1, i + 2)[b_sample > 0]).item()) == i + 1:
                zvec[i] = int(0)

            i = i + 1
        self.prob_background = np.diag(P)
        self.branching = zvec
        self.P = P

    def sample_offspring_parameters(self):
        """ Samples the offspring related parameters.
        """
        '''
        if self.iteration <= 50:
            Nsamples = 500
            print('Nsamples offspring=', Nsamples, 'iter=', self.iteration)
        else:
            Nsamples = np.copy(self.MH_proposals_offspring)
        '''
        '''
        Nsamples = np.copy(self.MH_proposals_offspring)
        if self.MH_proposals_offspring is None:
            if self.iteration < 4:#100:
                Nsamples = 100
            else:
                Nsamples = 10
        '''
        if self.iteration <= 50:  # 100:
            Nsamples = 50
        else:
            if self.MH_proposals_offspring is None:
                Nsamples = 10
            else:
                Nsamples = np.copy(self.MH_proposals_offspring)

        self.acc_offspring_per_iter = 0  # acceptance info
        if self.sigma_proposal_offspring_params is None:
            sigma_proposal = .0001  # it is the variance = s**2
        else:
            sigma_proposal = self.sigma_proposal_offspring_params

        # initialize markov chain
        dim_theta = len(self.theta) - 2  # last 2 params are fixed: beta_m, m0

        samples = np.empty([1, dim_theta]) * np.NaN
        samples[0, :dim_theta] = np.array(self.theta[:dim_theta])
        ll_samples = []

        if self.MH_cov_empirical_proposal is not None:
            if len(np.array(self.track_data_per_iter['theta_tilde'][::1])) >= 10:
                self.Sigma_empirical = 1.5 * np.cov(np.log(np.array(self.track_data_per_iter['theta_tilde'][::1]).T))
                print('Proposal uses Sigma_empirical')

        if sigma_proposal is not None:
            print('sigma proposal log units =', np.sqrt(sigma_proposal), 'Nproposals=', Nsamples)
        if self.Sigma_empirical is not None:
            print('empirical sigma proposal in log units =', np.sqrt(np.diag(self.Sigma_empirical)), 'Nproposals=',
                  Nsamples)

        for ii in range(Nsamples):

            # current state: theta_k
            theta_k = np.reshape(samples[ii, :], [1, dim_theta])

            # proposal state: theta* via multdim-Normal()
            ln_mean = np.squeeze(np.log(theta_k))
            ln_cov = np.eye(dim_theta) * sigma_proposal
            if self.Sigma_empirical is not None:
                ln_cov = np.copy(self.Sigma_empirical)
                idx = np.diag(ln_cov) < sigma_proposal
                if sum(idx) > 0:
                    diag = np.copy(np.diag(ln_cov))
                    diag[idx] = sigma_proposal
                    np.fill_diagonal(ln_cov, diag)

            theta_k_star = np.exp(
                np.random.multivariate_normal(ln_mean, ln_cov, 1))

            # log_like + log_prior
            l_k = self.calculate_log_likelihood(theta_k)
            log_prior_k = self.evaluate_log_prior(theta_k)
            l_k_star = self.calculate_log_likelihood(theta_k_star)
            log_prior_k_star = self.evaluate_log_prior(theta_k_star)

            # MH criterion
            # MH_alpha = np.min([1., np.exp(l_k_star - l_k)])
            MH_alpha = np.min([1., np.exp((l_k_star + log_prior_k_star) - (l_k + log_prior_k))])

            # acceptance/rejection
            u_i = np.random.uniform(0, 1, 1)
            if u_i < MH_alpha:
                samples = np.append(samples, theta_k_star, axis=0)
                ll_samples = np.append(ll_samples, l_k_star)
                self.acc_offspring_per_iter = self.acc_offspring_per_iter + 1
            else:
                samples = np.append(samples, theta_k, axis=0)
                ll_samples = np.append(ll_samples, l_k)
        self.l_tilde = ll_samples[-1]
        self.theta[:dim_theta] = samples[-1]
        self.acc_offspring_per_iter = self.acc_offspring_per_iter / Nsamples
        print('acc offspring=', self.acc_offspring_per_iter)

    def evaluate_log_prior(self, theta_k):
        log_p_prior_k = None

        # no prior is specified --> will be set to uniform
        if self.setup_obj.prior_theta_dist is None:
            self.setup_obj.prior_theta_dist = 'uniform'

        # uniform prior
        if self.setup_obj.prior_theta_dist == 'uniform':
            if self.setup_obj.prior_theta_params is None:
                self.setup_obj.prior_theta_params = np.array(
                    [[1e-7, 10], [1e-7, 10], [1e-7, 10], [1e-7, 10], [1e-7, 10], [1e-7, 10], [1., 10]])
                # self.setup_obj.prior_theta_params = np.array(
                #   [[0, 0.06], [0.001, 1.], [1., 1.3], [np.log(10) * 0.6, np.log(10)],
                #    [0.1 * 111., 1. * 111.], [1e-7, 1.], [1., 2.]])  # Hainzle and ETAS italy Lombardi ranges
            bounds = np.copy(self.setup_obj.prior_theta_params)
            if (theta_k >= bounds[:, 0]).all() and (theta_k <= bounds[:, 1]).all():
                log_p_prior_k = np.log(1. / np.prod(np.diff(bounds)))
            else:
                log_p_prior_k = -np.inf

        # gamma prior: parameterized via c_prior (coeffi of variation c=s/m) and mean_prior for each param
        if self.setup_obj.prior_theta_dist == 'gamma':
            if self.setup_obj.prior_theta_params is None:
                # params for gamma dist are here prior_mean,prior_coefficient_of_variation
                # i.e., for each param a pair [m,c], if c=1 prior becomes an exp.distribution
                # alpha_0 = 1/c**2, beta_0 = alpha_0/prior_mean
                # be aware of the following:
                # p-1 \sim Gamma()
                # q-1 \sim Gamma()
                # all other params: theta_i \sim Gamma()
                self.setup_obj.prior_theta_params = np.array(
                    [[0.1, 1.], [0.1, 1.], [0.5, 1.], [2., 1.], [0.5, 1.], [0.5, 1.], [1., 1.]])
            location_shift = [0, 0, 1., 0, 0, 0, 1.]
            alpha_vec = 1. / self.setup_obj.prior_theta_params[:, 1] ** 2.
            beta_vec = alpha_vec / self.setup_obj.prior_theta_params[:, 0]
            log_p_prior_k = np.sum(
                np.log(sc.stats.gamma.pdf(theta_k, a=alpha_vec, scale=1. / beta_vec, loc=location_shift)))

        if self.setup_obj.stable_theta_sampling is not None:
            m_beta = np.copy(self.setup_obj.m_beta)
            n_inf = gpetas.some_fun.n(m_alpha=theta_k[0, 3], m_beta=m_beta, K=theta_k[0, 0],
                                      c=theta_k[0, 1], p=theta_k[0, 2], t_start=0., t_end=np.inf)
            if not (0 <= n_inf < 1.):
                log_p_prior_k = -np.inf

        return log_p_prior_k

    def calculate_log_likelihood(self, theta_k):
        '''computes log likelihood of offspring, when branching is known/sampled'''
        # fixed parameters
        m_beta, m0 = self.theta[-2], self.theta[-1]
        # log-likelihood computation using the branching structure in Z
        kappa_m_j = lambda m_j, K, m_alpha, m0: K * np.exp(
            m_alpha * (m_j - m0))
        g_delta_t_ij = lambda delta_t_ij, c, p: 1. / (delta_t_ij + c) ** p

        # s(x_ij):
        # (1) GAUSS (short range)
        s_delta_x_ij_gauss = lambda delta_x_ij, delta_y_ij, m_j, D, m_alpha, m0: \
            1. / (2 * np.pi * D ** 2 * np.exp(m_alpha * (m_j - m0))) \
            * np.exp(
                (-1. / 2. * ((delta_x_ij) ** 2 + (delta_y_ij) ** 2)) / (
                        D ** 2 * np.exp(m_alpha * (m_j - m0))))
        # (2a) Power Law decay (long range)
        s_delta_x_ij_pwl = lambda delta_x_ij, delta_y_ij, m_j, D, gamma, q, m0: \
            (q - 1) / (np.pi * D ** 2 * np.exp(gamma * (m_j - m0))) \
            * (1. + ((delta_x_ij) ** 2 + (delta_y_ij) ** 2) / (D ** 2 * np.exp(gamma * (m_j - m0)))) ** (-q)
        # (2b) Rupture Length Power Law decay (long  range)
        s_delta_x_ij_RL_pwl = lambda delta_x_ij, delta_y_ij, m_j, D, gamma, q, m0: \
            (q - 1) / (np.pi * D ** 2 * 10. ** (2 * gamma * (m_j))) \
            * (1. + ((delta_x_ij) ** 2 + (delta_y_ij) ** 2) / (D ** 2 * 10. ** (2 * gamma * (m_j)))) ** (-q)

        # log-likelihood l(theta_k): current state of the markov chain
        l_theta_k = 0.
        offspring_idx = np.where(self.branching > 0)[0]
        self.offspring_idx = np.where(self.branching > 0)[0]
        if np.size(offspring_idx) == 0:
            l_theta_k = -np.inf
            self.l_theta_k = l_theta_k
            return -np.inf

        for o_idx in offspring_idx:
            z_i = int(self.branching[o_idx])  # z_i = j with j=1,... (starts at 1
            # not at 0)
            m_j = self.data.magnitudes[z_i - 1]
            delta_t_ij = self.data.times[o_idx] - self.data.times[z_i - 1]
            delta_x_ij = self.data.positions[o_idx, 0] - \
                         self.data.positions[z_i - 1, 0]
            delta_y_ij = self.data.positions[o_idx, 1] - \
                         self.data.positions[z_i - 1, 1]

            # spatial kernel
            K, c, p, m_alpha, D, gamma, q = np.empty(7) * np.nan
            if len(self.theta) == 7:  # (1) GAUSS (short range)
                K, c, p, m_alpha, D = np.copy(np.squeeze(theta_k))
                s_xij = s_delta_x_ij_gauss(delta_x_ij, delta_y_ij, m_j, D, m_alpha, m0)
            if len(self.theta) == 9 and self.spatial_kernel[0] == 'P':  # (2a) Power Law decay (long range)
                K, c, p, m_alpha, D, gamma, q = np.copy(np.squeeze(theta_k))
                s_xij = s_delta_x_ij_pwl(delta_x_ij, delta_y_ij, m_j, D, gamma, q, m0)
            if len(self.theta) == 9 and self.spatial_kernel[
                0] == 'R':  # (2b) Rupture Length Power Law decay (long  range)
                K, c, p, m_alpha, D, gamma, q = np.copy(np.squeeze(theta_k))
                s_xij = s_delta_x_ij_RL_pwl(delta_x_ij, delta_y_ij, m_j, D, gamma, q, m0)

            l_i = np.log(
                kappa_m_j(m_j, K, m_alpha, m0) * g_delta_t_ij(delta_t_ij, c,
                                                              p) * s_xij)
            l_theta_k = l_theta_k + l_i

        # integral part (compensator)
        time_int_k = K * np.sum(
            np.exp(m_alpha * (self.data.magnitudes - m0)) * 1. / (-p + 1.) * (
                    (c + (self.T - self.data.times[:])) ** (-p +
                                                            1.) - (c)
                    ** (
                            -p + 1.)))
        if p == 1: time_int_k = K * np.sum(
            np.exp(m_alpha * (self.data.magnitudes - m0)) * (
                    np.log(c + (self.T - self.data.times)) - np.log(c)))
        compensator_k_approx = time_int_k * 1.
        # l_k_approx = l_theta_k - compensator_k_approx

        if len(self.theta) == 7:
            int_all_k = 0.
            xmin, xmax = self.S_borders[0, :]
            ymin, ymax = self.S_borders[1, :]
            for idx in range(self.N):
                sigma_i = D * np.sqrt(
                    np.exp(m_alpha * (self.data.magnitudes[idx] - m0)))
                int_x = 1. / 2. * (math.erf(
                    (xmax - self.data.positions[idx, 0]) / (sigma_i * np.sqrt(
                        2.))) -
                                   math.erf(
                                       (xmin - self.data.positions[idx, 0]) / (
                                               sigma_i * np.sqrt(2.))))
                int_y = 1. / 2. * (math.erf(
                    (ymax - self.data.positions[idx, 1]) / (sigma_i * np.sqrt(
                        2.))) - math.erf(
                    (ymin - self.data.positions[idx, 1]) / (sigma_i * np.sqrt(
                        2.))))
                int_all_k = int_all_k + K * np.sum(
                    np.exp(m_alpha * (self.data.magnitudes[idx] - m0)) * 1. / (-p +
                                                                               1.) * (
                            (c + (self.T - self.data.times[idx])) ** (-p +
                                                                      1.) - (
                                c) ** (
                                    -p + 1.))) * int_x * int_y

            compensator_k = int_all_k
            l_k = l_theta_k - compensator_k
        else:
            # l_k = l_theta_k - compensator_k
            l_k = l_theta_k - compensator_k_approx
        # print('compensator_erfc, compensator_k_approx',compensator_k,compensator_k_approx)
        self.l_theta_k = l_theta_k
        return l_k

    def save_sample(self):
        """ Saves all current sample into some dictionary?
        """
        # pass
        self.save_data['N_S0'].append(np.copy(self.bg_sampler.N))
        self.save_data['N_Stilde'].append(np.copy(self.N - self.bg_sampler.N))
        self.save_data['theta_tilde'].append(np.copy(self.theta[:-2]))
        self.save_data['branching'].append(np.copy(self.branching))
        self.save_data['l_tilde'].append(np.copy(self.l_tilde))
        self.save_data['l_theta_k'].append(np.copy(self.l_theta_k))  # ll of intensity part
        self.save_data['l_mu_givenZ_at_S0_intensity_part'].append(
            np.copy(np.sum(np.log(self.mu0[self.bg_sampler.S_0_idx]))))
        self.save_data['M'].append(np.copy(self.bg_sampler.M))
        self.save_data['lambda_bar'].append(np.copy(self.bg_sampler.lmbda_star))
        self.save_data['mu0_data'].append(np.copy(self.mu0))
        self.save_data['mu_grid'].append(np.copy(self.mu0_grid))
        self.save_data['cov_params'].append(np.copy(np.array(self.bg_sampler.cov_params, dtype=object)))
        self.save_data['cov_params_theta'].append(np.copy(self.bg_sampler.cov_params[0]))
        if self.dim == 2:
            self.save_data['cov_params_nu1'].append(np.copy(self.bg_sampler.cov_params[1][0]))
            self.save_data['cov_params_nu2'].append(np.copy(self.bg_sampler.cov_params[1][1]))
        if self.dim == 1:
            self.save_data['cov_params_nu1'].append(np.copy(self.bg_sampler.cov_params[1]))
        self.save_data['acc_offspring_per_iter'].append(np.copy(self.acc_offspring_per_iter))
        # integral versions
        self.save_data['int_riemann_approx'].append(self.R / len(self.mu0_grid) * np.sum(self.mu0_grid) * self.T)
        self.save_data['Campbell_1_log_I_k'].append(-np.sum(self.sigmoid(self.bg_sampler.g_unthinned)))
        self.save_data['Campbell_2_log_I_k'].append(np.sum(np.log(self.sigmoid(-self.bg_sampler.g_unthinned))))
        self.save_data['time'].append(time.perf_counter())
        print('current M=', self.bg_sampler.M)

        ''' save to numpy txt each 10th iteration after burn in'''
        if self.iteration % 500 == 0:
            ''' save results of the sampler '''
            file = open(self.setup_obj.outdir + "/GS_save_data_%s.bin" % self.case_name, "wb")
            pickle.dump(self.save_data, file)
            file.close()

    def _write_info_file(self):
        """
        """
        ''' write init to file '''
        h_silverman = gpetas.some_fun.silverman_scott_rule_d(self.data_obj.data_all.positions)
        fid = open(self.setup_obj.outdir + "/init_and_setup_info_%s.txt" % self.case_name, 'w')
        fid.write("burnin                               = %i\n" % (self.burnin))
        fid.write("Number of iterations                 = %i\n" % (self.num_samples))
        fid.write("Thinning                             = %i\n" % (self.thinning))
        fid.write("K_samples after burnin               = %f\n" % ((self.num_samples - 1) / self.thinning))
        fid.write("Total number of iterations           = %i\n" % (self.burnin + self.num_samples))
        fid.write('-----------------------------------------------------\n')
        fid.write('h_silverman in degrees               = %f\n' % (h_silverman))
        fid.write('h_silverman in km                    = %f\n' % (h_silverman * 111))
        fid.write('m0                                   = %f\n' % (self.setup_obj.m0))
        fid.write('-----------------------------------------------------\n')
        fid.write(' (0) GP COV function: hyperparameters initialization:\n')
        fid.write('-----------------------------------------------------\n')
        fid.write('cov_params_init nu_0                 = %f\n' % (self.setup_obj.cov_params_start[0]))
        if self.dim == 2:
            fid.write('cov_params_init nu_1                 = %f\n' % (self.setup_obj.cov_params_start[1][0]))
            fid.write('cov_params_init nu_2                 = %f\n' % (self.setup_obj.cov_params_start[1][1]))
        if self.dim == 1:
            fid.write('cov_params_init nu_1                 = %f\n' % (self.setup_obj.cov_params_start[1]))
        fid.write('initial values for nu_1, nu_2 from Silverman rule using all the data\n')
        fid.write('-----------------------------------------------------\n')
        fid.write(' (1) GP COV function: hyper prior initialization:\n')
        fid.write('-----------------------------------------------------\n')
        fid.write('mean nu_0                            = %f\n' % (self.save_data['hyper_prior_mu_nu0']))
        fid.write('beta nu_0                            = %f\n' % (1. / self.save_data['hyper_prior_mu_nu0']))
        if self.dim == 2:
            fid.write(
                'mean nu_1                            = %f\n' % (self.save_data['hyper_prior_mu_nu12_length_scale'][0]))
            fid.write(
                'mean nu_2                            = %f\n' % (self.save_data['hyper_prior_mu_nu12_length_scale'][1]))
            fid.write('beta nu_1                            = %f\n' % (
                    1. / self.save_data['hyper_prior_mu_nu12_length_scale'][0]))
            fid.write('beta nu_2                            = %f\n' % (
                    1. / self.save_data['hyper_prior_mu_nu12_length_scale'][1]))
        fid.write('hyper_prior_length_scale_factor      = %f\n' % (self.bg_sampler.hyper_prior_length_scale_factor))
        fid.write('prior_mu_length_scale: mean nu_1 = mean nu_2 = default 0.1*dx\n')
        fid.write('default value for mu_nu12 (length scale) 0.1*dx would be in this case: %f\n' % (
                0.1 * np.diff(self.data_obj.domain.X_borders[0, :])))
        fid.write('sigma_proposal_hypers in log units   = %f\n' % (self.bg_sampler.sigma_proposal_hypers))
        fid.write('-----------------------------------------------------\n')
        fid.write(' (2) SGCP upper bound:\n')
        fid.write('-----------------------------------------------------\n')
        fid.write('lambda_bar start                     = %e\n' % (self.save_data['lambda_bar_start']))
        fid.write('hyper prior Gamma(mu,c):\n')
        fid.write('mu lambda_bar                        = %e\n' % (self.save_data['lambda_bar_hyper_prior_mu']))
        fid.write(
            'c =coeffi. of var                    = %e\n' % (self.save_data['lambda_bar_hyper_prior_c_coeffi_of_var']))
        fid.write('The choice of mu and c determines Gamma(alpha_0,beta_0) parameterization\n')
        fid.write('alpha_0                              =%f\n' % (self.bg_sampler.alpha0))
        fid.write('beta_0                               =%f\n' % (self.bg_sampler.beta0))
        fid.write('-----------------------------------------------------\n')
        fid.write(' (3) offspring:\n')
        fid.write('-----------------------------------------------------\n')
        fid.write('GS.setup_obj.spatial_offspring       =%s\n' % (self.setup_obj.spatial_offspring))
        fid.write('GS.spatial_kernel                    =%s\n' % (self.spatial_kernel))
        # if self.setup_obj.spatial_offspring == 'R':
        K, c, p, m_alpha, D, gamma, q, m_beta, m0 = self.theta_start
        fid.write('K_start                           =%f\n' % (K))
        fid.write('c_start                           =%f\n' % (c))
        fid.write('p_start                           =%f\n' % (p))
        fid.write('m_alpha_start                     =%f\n' % (m_alpha))
        fid.write('D_start                           =%f\n' % (D))
        fid.write('gamma_start                       =%f\n' % (gamma))
        fid.write('q_start                           =%f\n' % (q))
        fid.write('sigma proposal offspring log units=%f\n' % (np.sqrt(self.sigma_proposal_offspring_params)))
        fid.write('m_beta_start                      =%f\n' % (m_beta))
        fid.write('m0_start                          =%f\n' % (m0))
        # if self.setup_obj.spatial_offspring == 'P':
        #    K, c, p, m_alpha, D, gamma, q, m_beta, m0 = self.theta_start
        # if self.setup_obj.spatial_offspring == 'G':
        #    K, c, p, m_alpha, D, gamma, q, m_beta, m0 = self.theta_start
        fid.write('-----------------------------------------------------\n')
        fid.write(' (4) branching:\n')
        fid.write('-----------------------------------------------------\n')
        fid.write('N                                 =%i\n' % (self.N))
        fid.write('N0_start                          =%i\n' % (self.bg_sampler.N))
        fid.write('M_start                           =%i\n' % (self.bg_sampler.M))
        fid.close()
        return

    def save_data_init_info(self):
        # info savings
        self.save_data['time'].append(time.perf_counter())
        self.save_data['X_grid'] = self.setup_obj.X_grid
        self.save_data['X_grid_NN'] = self.bg_sampler.X_grid
        self.save_data['burnin'] = self.burnin
        self.save_data['Ksamples_after_burnin'] = self.num_samples
        self.save_data['data_obj'] = self.data_obj
        self.save_data['setup_obj'] = self.setup_obj
        self.save_data['theta_start'] = self.theta_start
        self.save_data['cov_params_start'] = self.setup_obj.cov_params_start
        self.save_data['lambda_bar_start'] = self.bg_sampler.lmbda_star_start
        self.save_data['lambda_bar_hyper_prior_mu'] = self.bg_sampler.prior_mu_upper_bound
        self.save_data['lambda_bar_hyper_prior_c_coeffi_of_var'] = self.bg_sampler.prior_std_factor_upper_bound
        self.save_data['case_name'] = self.case_name
        if hasattr(self.data_obj, 'theta_true_Kcpadgq'):
            self.save_data['theta_true_Kcpadgq'] = self.data_obj.theta_Kcpadgq

    def save_results_and_info(self):
        # save results and info of the sampler
        if self.iteration > self.burnin:
            if self.iteration % self.thinning == 0:
                self.save_sample()
        if self.iteration == self.burnin + 1:
            self._write_info_file()
        if self.iteration == 1:
            self.save_data['hyper_prior_mu_nu0'] = self.bg_sampler.hyper_prior_mu_nu0
            self.save_data['hyper_prior_mu_nu12_length_scale'] = self.bg_sampler.hyper_prior_mu_length_scale
        if self.iteration > self.burnin:
            self.track_vars_per_iter()
        self.iteration = self.iteration + 1
        print('iter= ', self.iteration, 'Current lambda_star', self.bg_sampler.lmbda_star, 'M=', self.bg_sampler.M,
              'N0=', self.bg_sampler.N)

    def track_vars_per_iter(self):
        self.track_data_per_iter['theta_tilde'].append(np.copy(self.theta[:-2]))
        self.track_data_per_iter['acc_offspring_per_iter'].append(np.copy(self.acc_offspring_per_iter))
        self.track_data_per_iter['acc_hyperparams_per_iter'].append(np.copy(self.bg_sampler.acc_hyperparams_per_iter))


def _silverman_scott_rule_d(X_data, individual_yes=None):
    ''' Silvermans rule: (4.15) page 86--87: h* = mean(sigma_ii)* N**(-1/(d+4))
     = mean_std * N^(-0.1666...)
     e.g., used for minimum bandwidth in classical kde ETAS
    '''
    # h_opt goes with the std(data)* N**()
    # ==== in case of d=2 both are the same
    # silverman  H_ij=0 if i neq j (just diagonal)
    d = X_data.shape[1]
    sigma_vec = np.std(X_data, axis=0)
    N = len(X_data[:, 0])
    silverman_hstar = 1. / d * np.sum(sigma_vec) * N ** (-1. / (d + 4))
    # ==== in case of d=2 both are the same
    if individual_yes is not None:
        silverman_hstar = sigma_vec * N ** (-1. / (d + 4))
    return silverman_hstar


# subclasses
class data_structure():
    def __init__(self):
        self.times = None
        self.magnitudes = None
        self.positions = None
