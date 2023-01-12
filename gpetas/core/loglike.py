import numpy as np
from scipy.special import logsumexp
import gpetas
from gpetas.utils.some_fun import mu_xprime_gpetas


class eval_lnl():
    def __init__(self, data_obj, mu_xi_at_all_data, integral_mu_x_unit_time,
                 theta_phi__Kcpadgq, m0=None,
                 X_borders_eval_l=None, T_borders_eval_l=None,
                 spatial_kernel=None):
        """
        :param data_obj:
        :type data_obj:
        :param mu_xi_at_all_data:
        :type mu_xi_at_all_data:
        :param integral_mu_x_unit_time:
        :type integral_mu_x_unit_time:
        :param theta_phi__Kcpadgq:
        :type theta_phi__Kcpadgq:
        :param m0:
        :type m0:
        :param X_borders_eval_l:
        :type X_borders_eval_l:
        :param T_borders_eval_l:
        :type T_borders_eval_l:
        :param spatial_kernel:
        :type spatial_kernel:
        """

        if T_borders_eval_l is None:
            T_borders_eval_l = data_obj.domain.T_borders_all
        if X_borders_eval_l is None:
            X_borders_eval_l = data_obj.domain.X_borders
        if spatial_kernel is None:
            spatial_kernel = 'R'

        self.theta_phi__Kcpadgq = theta_phi__Kcpadgq
        self.spatial_kernel = spatial_kernel
        self.m0 = m0
        self.T_borders_eval_l = T_borders_eval_l
        self.X_borders_eval_l = X_borders_eval_l
        self.Nall = len(data_obj.data_all.times)
        self.data_all = data_obj
        t_vec_all = data_obj.data_all.times.reshape(-1, 1)  # (Nall,1)
        m_vec_all = data_obj.data_all.magnitudes.reshape(-1, 1)  # (Nall,1)
        positions_all = data_obj.data_all.positions.reshape(-1, 2)  # (Nall,2)

        T1, T2 = self.T_borders_eval_l
        if T2 < T1:
            raise SystemExit('Error T_eval_end < T_eval_start. Choose T_eval_end > T_eval_start: T_borders_eval_l=',
                             self.T_borders_eval_l)

        if m0 is None:
            m0 = np.min(m_vec_all)
        if np.min(m_vec_all) < m0:
            m0 = np.min(m_vec_all)
            print('Warning: m0 input is to low; m0 is set to the minimum in the data, which is =', m0)

        # likelihood evaluation window
        self.idx = np.logical_and(T1 <= t_vec_all, t_vec_all <= T2).squeeze()
        self.N_lnl_eval = len(data_obj.data_all.times[self.idx])

        # intensity bg at data of lnl evaluation window
        if len(mu_xi_at_all_data) == len(self.idx):
            self.mu_xi_atdata = mu_xi_at_all_data[self.idx]
        elif len(mu_xi_at_all_data) == np.sum(self.idx):
            self.mu_xi_atdata = mu_xi_at_all_data
        else:
            raise ValueError("array of mu must have length either (i) N all data or (ii) N* test data.")

        # intensity offspring
        Delta_tij = np.tril(t_vec_all - t_vec_all.T, k=-1)  # take only causal part
        dx = positions_all[:, 0, np.newaxis] - positions_all[:, 0, np.newaxis].T
        dy = positions_all[:, 1, np.newaxis] - positions_all[:, 1, np.newaxis].T
        Delta_rsq_ij = np.tril(dx ** 2 + dy ** 2,
                               k=-1)  # take only causal part of the squared distance between all points
        # functions of the kernels
        kappa_mj = lambda m_vec, K, m_alpha, m0: K * np.exp(m_alpha * (m_vec - m0))
        g_delta_tij = lambda Delta_tij, c, p: np.tril(1. / (Delta_tij + c) ** p, k=-1)
        # spatial kernels
        s_delta_xij_gauss = lambda Delta_rsq_ij, D_gauss, m_vec, m_alpha, m0: \
            np.tril(1. / (2 * np.pi * D_gauss ** 2 * np.exp(m_alpha * (m_vec.squeeze() - m0))) \
                    * np.exp(-1. / (2 * D_gauss ** 2 * np.exp(m_alpha * (m_vec.squeeze() - m0))) * Delta_rsq_ij), k=-1)
        s_delta_xij_pwl = lambda Delta_rsq_ij, D, gamma, q, m_vec, m0: \
            np.tril((q - 1) / (np.pi * D ** 2 * np.exp(gamma * (m_vec.squeeze() - m0))) \
                    * (1. + 1. / (D ** 2 * np.exp(gamma * (m_vec.squeeze() - m0))) * Delta_rsq_ij) ** (-q), k=-1)
        s_delta_xij_RL_pwl = lambda Delta_rsq_ij, D, gamma, q, m_vec, m0: \
            np.tril((q - 1) / (np.pi * D ** 2 * 10 ** (2 * gamma * (m_vec.squeeze()))) \
                    * (1. + 1. / (D ** 2 * 10 ** (2 * gamma * (m_vec.squeeze()))) * Delta_rsq_ij) ** (-q), k=-1)
        # computation using s_delta_xij_RL_pwl spatial kernel
        K, c, p, m_alpha, D, gamma, q = np.empty(7) * np.nan
        s_xij = None
        if self.spatial_kernel == 'R':
            K, c, p, m_alpha, D, gamma, q = theta_phi__Kcpadgq
            s_xij = s_delta_xij_RL_pwl(Delta_rsq_ij, D, gamma, q, m_vec_all, m0)
        if self.spatial_kernel == 'P':
            K, c, p, m_alpha, D, gamma, q = theta_phi__Kcpadgq
            s_xij = s_delta_xij_pwl(Delta_rsq_ij, D, gamma, q, m_vec_all, m0)
        if self.spatial_kernel == 'G':
            K, c, p, m_alpha, D = theta_phi__Kcpadgq[0:5]
            s_xij = s_delta_xij_gauss(Delta_rsq_ij, D, m_vec_all, m_alpha, m0)

        mat = kappa_mj(m_vec_all, K, m_alpha, m0).reshape(1, self.Nall) \
              * g_delta_tij(Delta_tij, c, p) \
              * s_xij
        self.intensity_offspring = np.sum(mat, axis=1)[self.idx]

        # integral background (bg)
        self.integral_BG = integral_mu_x_unit_time * (T2 - T1)

        # integral offspring
        UB_vec = (T2 - t_vec_all).reshape(-1, 1)
        LB_vec = (T1 - t_vec_all).reshape(-1, 1)
        LB_vec[LB_vec < 0] = 0
        UB_vec[UB_vec < 0] = 0
        if p == 1:
            int_offspring = K * np.exp(m_alpha * (m_vec_all - m0)) * (np.log(UB_vec + c) - np.log(LB_vec + c))
        else:
            int_offspring = K * np.exp(m_alpha * (m_vec_all - m0)) * (
                    1. / (-p + 1) * (UB_vec + c) ** (-p + 1) - 1. / (-p + 1) * (LB_vec + c) ** (-p + 1))

        ### WARNING ALL events before t contribute to the integral of the offspring,
        # NOT only the events inside the evaluation period!!!!
        # self.integral_offspring = np.sum(int_offspring[self.idx])
        # NEW: 27.1.2022
        self.integral_offspring = np.sum(int_offspring)
        ### END WARNING ####

        # gathering all parts for the lnl evaluation window [T1,T2]
        self.sum_ln_intensity_part = np.sum(np.log(self.mu_xi_atdata + self.intensity_offspring))
        self.total_integral_part = self.integral_BG + self.integral_offspring
        self.lnl_value = self.sum_ln_intensity_part - self.total_integral_part


class lnl_sample():
    def __init__(self, save_obj_GS,
                 data_obj=None,
                 idx_samples=None,
                 T_borders_eval_l=None,
                 method_posterior_GP=None,
                 method_integral=None):
        if data_obj is None:
            data_obj = save_obj_GS['data_obj']
        if idx_samples is None:
            idx_samples = np.arange(len(save_obj_GS['lambda_bar']))
        self.idx_samples = idx_samples
        self.Ksamples = len(save_obj_GS['lambda_bar'])
        self.Ksamples_used = len(idx_samples)
        self.lnl_all = np.empty(len(idx_samples))
        self.Ntest = np.empty(len(idx_samples))
        self.Nall = np.empty(len(idx_samples))
        self.spatial_kernel = 'R'  # later part of info in save_obj_GS['setup']
        self.X_borders = data_obj.domain.X_borders
        self.absX = np.prod(np.diff(self.X_borders))
        tau_1, tau_2 = data_obj.domain.T_borders_training
        if T_borders_eval_l is None:
            T_borders_eval_l = data_obj.domain.T_borders_testing
        if method_posterior_GP is None:
            method_posterior_GP = 'nearest'  # other: interpol_linear (cubic, nearest); sparse; sparse_mean
        if method_integral is None:
            method_integral = 'Riemann_sum'  # other: Campbell_sampled
        bins = int(np.sqrt(save_obj_GS['X_grid'].shape[0]))
        self.X_grid = gpetas.some_fun.make_X_grid(self.X_borders, nbins=bins)
        arr_integral_mu_x_unit_time = np.empty(len(idx_samples))
        idx_testing = np.logical_and(data_obj.data_all.times >= T_borders_eval_l[0],
                                     data_obj.data_all.times <= T_borders_eval_l[1])
        position_testing = data_obj.data_all.positions[idx_testing, :]

        for i in range(len(idx_samples)):
            k = idx_samples[i]
            # mu_xi_at_all_data (=data_obj.data_all.positions)
            mu_xi_at_all_data = mu_xprime_gpetas(xprime=position_testing,
                                                 mu_grid=save_obj_GS['mu_grid'][k], X_grid=self.X_grid,
                                                 X_borders=self.X_borders,
                                                 method=method_posterior_GP,
                                                 lambda_bar=save_obj_GS['lambda_bar'][k],
                                                 cov_params=[save_obj_GS['cov_params_theta'][k].item(), np.array(
                                                     [save_obj_GS['cov_params_nu1'][k].item(),
                                                      save_obj_GS['cov_params_nu2'][k].item()])])

            # integral mu_x
            if method_integral == 'Riemann_sum':
                L = len(np.array(save_obj_GS['mu_grid'][0]))
                arr_integral_mu_x_unit_time[i] = self.absX / L * np.sum(np.array(save_obj_GS['mu_grid'][k]))
            if method_integral == 'Campbell_sampled':
                arr_integral_mu_x_unit_time[i] = -np.array(save_obj_GS['Campbell_1_log_I_k'][k]) / (tau_2 - tau_1)

            # lnl computation
            lnl_i = eval_lnl(data_obj, mu_xi_at_all_data,
                             integral_mu_x_unit_time=arr_integral_mu_x_unit_time[i],
                             theta_phi__Kcpadgq=save_obj_GS['theta_tilde'][k],
                             m0=data_obj.domain.m0,
                             X_borders_eval_l=self.X_borders,
                             T_borders_eval_l=T_borders_eval_l,
                             spatial_kernel=self.spatial_kernel)
            self.lnl_all[i] = lnl_i.lnl_value
            self.Ntest[i] = lnl_i.N_lnl_eval
            self.Nall[i] = lnl_i.Nall
            self.lnl_i = lnl_i


class test_likelihood_GS():
    def __init__(self, save_obj_GS,
                 testing_periods=None,
                 data_obj=None,
                 idx_samples=None,
                 method_posterior_GP=None,
                 method_integral=None):
        '''

        :param save_obj_GS:
        :type save_obj_GS:
        :param testing_periods:
        :type testing_periods:
        :param data_obj:
        :type data_obj:
        :param idx_samples:
        :type idx_samples:
        :param T_borders_eval_l:
        :type T_borders_eval_l:
        :param method_posterior_GP:
        :type method_posterior_GP:
        :param method_integral:
        :type method_integral:
        '''

        if data_obj is None:
            data_obj = save_obj_GS['data_obj']
        if testing_periods is None:
            testing_periods = np.copy(data_obj.domain.T_borders_testing).reshape([1, -1])
        if idx_samples is None:
            Ksamples = len(save_obj_GS['lambda_bar'])
            idx_samples = np.arange(0, Ksamples, 100)
            print('Number of employed posterior samples:', len(idx_samples))

        # info
        print('testing periods')
        print(testing_periods, 'days.')
        print('Number of employed posterior samples:', len(idx_samples))

        # output arrays
        l_test_GPetas_log_E_L = np.empty(len(testing_periods[:, 0]))
        Ntest_arr = np.empty(len(testing_periods[:, 0]))
        lnl_samples_mat = np.zeros([len(testing_periods[:, 0]), len(idx_samples)]) * np.nan

        # loop over testing periods
        for i in range(len(testing_periods[:, 0])):
            T_borders_testing = testing_periods[i, :]
            print('Current T_star_testing =', T_borders_testing)

            # gpetas: log_E_L_gpetas
            lnl_samples_i = gpetas.loglike.lnl_sample(save_obj_GS=save_obj_GS,
                                                      data_obj=data_obj,
                                                      idx_samples=idx_samples,
                                                      T_borders_eval_l=T_borders_testing,
                                                      method_posterior_GP=method_posterior_GP,
                                                      method_integral=method_integral)
            lnl_samples_mat[i, :] = np.copy(lnl_samples_i.lnl_all)
            l_test_GPetas_log_E_L[i] = logsumexp(lnl_samples_i.lnl_all) - np.log(len(lnl_samples_i.lnl_all))
            Ntest_arr[i] = lnl_samples_i.Ntest[0]
            print('Employed number of posterior samples:', len(lnl_samples_i.idx_samples))

        self.Ntest_arr = np.copy(Ntest_arr)
        self.l_test_GPetas_log_E_L = np.copy(l_test_GPetas_log_E_L)
        self.lnl_samples_mat = np.copy(lnl_samples_mat)


class test_likelihood_mle():
    def __init__(self,
                 mle_obj,
                 data_obj=None,
                 testing_periods=None, method_mu=None):
        '''

        :param mle_obj:
        :type mle_obj:
        :param data_obj:
        :type data_obj:
        :param idx_samples:
        :type idx_samples:
        :param T_borders_eval_l:
        :type T_borders_eval_l:
        :param method_posterior_GP:
        :type method_posterior_GP:
        :param method_integral:
        :type method_integral:
        '''
        if data_obj is None:
            data_obj = mle_obj.data_obj
        if testing_periods is None:
            testing_periods = np.copy(data_obj.domain.T_borders_testing).reshape([1, -1])

        # info
        print('testing periods')
        print(testing_periods, 'days.')

        # output arrays
        l_test_kde_default = np.empty(len(testing_periods[:, 0]))
        Ntest_arr = np.empty(len(testing_periods[:, 0]))

        # loop over testing periods
        for i in range(len(testing_periods[:, 0])):
            T_borders_eval_l = testing_periods[i, :]
            print('Current T_star_testing =', T_borders_eval_l)

            # testing_i data
            idx_testing = np.logical_and(data_obj.data_all.times >= T_borders_eval_l[0],
                                         data_obj.data_all.times <= T_borders_eval_l[1])
            position_testing = data_obj.data_all.positions[idx_testing, :]
            Ntest_arr[i] = len(position_testing[:, 0])

            # Evaluation of mu_xi_at_all_data
            if method_mu is None:
                method_mu = 'kde'
            if method_mu is not None:
                if method_mu == 'kde':
                    mu_xprime = mle_obj.eval_kde_xprime(position_testing)
                else:
                    xprime = np.copy(position_testing)
                    X_borders = mle_obj.data_obj.domain.X_borders
                    mu_xprime = gpetas.some_fun.get_grid_data_for_a_point(mle_obj.mu_grid.flatten(),
                                                                          xprime, X_borders=X_borders,
                                                                          method=method_mu)

            # lnl mle: kde DEFAULT of the realization
            lnl_KDE_default = gpetas.loglike.eval_lnl(data_obj=data_obj,
                                                      mu_xi_at_all_data=mu_xprime,
                                                      integral_mu_x_unit_time=np.sum(
                                                          mle_obj.p_i_vec) / mle_obj.absT_training,
                                                      theta_phi__Kcpadgq=mle_obj.theta_mle_Kcpadgq,
                                                      m0=mle_obj.setup_obj.m0,
                                                      X_borders_eval_l=mle_obj.data_obj.domain.X_borders,
                                                      T_borders_eval_l=T_borders_eval_l,
                                                      spatial_kernel=mle_obj.setup_obj.spatial_offspring)
            l_test_kde_default[i] = lnl_KDE_default.lnl_value
        # write to object
        self.Ntest_arr = np.copy(Ntest_arr)
        self.l_test_kde_default = np.copy(l_test_kde_default)
