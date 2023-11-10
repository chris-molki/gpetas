import os
import sys
import warnings
# warnings.filterwarnings("ignore")
import pickle

import numpy as np
import scipy as sc
from scipy.special import logsumexp

'''matplotlib backend issue'''
import matplotlib
# matplotlib.use('Agg')#matplotlib.use('PS') # agg, which is a non-GUI backend
import matplotlib.pyplot as plt


class generate_synthetic_data:
    """
    class: generates synthetic data according to three synthetic cases shown in the paper
    """

    def __init__(self, m0=3.36, m_beta=np.log(10),
                 theta_true_Kcpadgq=np.array([0.018, 0.006, 1.2, 1.69, 0.015, 0.2, 2.]), spatial_offspring='R',
                 X_borders=np.array([[0., 5.], [0., 5.]]), T_borders_all=np.array([0., 5000.]),
                 T_borders_training=np.array([0., 5000.]), T_borders_testing=np.array([0., 5000.]),
                 grid_points_n_per_dim=50, seed=None):
        """
        class: generates synthetic data according to three synthetic cases shown in the paper

        :param m0: float:
            Lower cutoff magnitude of f_M(m), aka (magnitude of completeness).
        :param m_beta: float:
        :type m_beta:
        :param theta_true_Kcpadgq:
        :type theta_true_Kcpadgq:
        :param spatial_offspring:
        :type spatial_offspring:
        :param X_borders:
        :type X_borders:
        :param T_borders_all:
        :type T_borders_all:
        :param T_borders_training:
        :type T_borders_training:
        :param T_borders_testing:
        :type T_borders_testing:
        :param grid_points_n_per_dim:
        :type grid_points_n_per_dim:
        :param seed:
        :type seed:
        """

        # init structures
        self.domain = dom()
        self.data_all = obs()
        self.case_name = None
        self.lambda_bar_true = None

        # setup experiment
        self.domain.X_borders = X_borders
        self.domain.T_borders_all = T_borders_all
        self.domain.T_borders_training = T_borders_training
        self.domain.T_borders_testing = T_borders_testing
        self.domain.m0 = m0
        self.N0 = None
        self.N = None
        self.Noff = None
        self.S0_events = None
        self.integral_mu_x_unit_time = None
        self.idx_training = None

        # attributes and info
        self.mu_i_at_data = None
        self.nbins = 50
        self.X_grid = make_X_grid(self.domain.X_borders, self.nbins)
        self.mu_grid = None
        self.n_stability = None
        self.n_stability_inf = None
        self.case01 = None
        self.case02 = None
        self.case03 = None

        # self.m0 = m0
        self.m_beta = m_beta
        self.theta_true_Kcpadgq = theta_true_Kcpadgq
        self.spatial_offspring = spatial_offspring
        self.grid_points_n_per_dim = grid_points_n_per_dim
        self.seed = seed

    def sim_case_01(self,
                    mu_true_3zones_ua_ut=None,
                    sep_fac=None,
                    seed=4, save_yes=None,
                    m0=None,
                    T_borders_all=None, T_borders_training=None,T_borders_testing=None
                    ):
        """
        Generates data according to case1 of the paper.

        :param mu_true_3zones_ua_ut: array (3,1):
            True intensities of the three zones.
        :param sep_fac: array (2,1):
            Determines how domain X is subdivided into three zones X = [X_1 \cup X_2 \cup X_3].
        :param seed: int or None:
            Initial random seed for the simulations.
        :param save_yes: not None or None:
            Option if simulated data is saved (pickled).
        :return: gm_obj: object:
            Contains the simulated data and additional info about the simulation.
        """
        self.case_name = 'case_01'
        # case individual time domain calT=[tau0,tau1]
        if T_borders_all is not None:
            self.domain.T_borders_all = T_borders_all
        if T_borders_training is not None:
            self.domain.T_borders_training = T_borders_training
        if T_borders_testing is not None:
            self.domain.T_borders_testing = T_borders_testing
        # case individual m0
        if m0 is not None:
            self.domain.m0 = m0
        if m0 is None:
            m0 = self.domain.m0
        m_beta = self.m_beta
        if sep_fac is None:
            sep_fac = ([0.6, 0.3])
        if mu_true_3zones_ua_ut is None:
            mu_true_3zones_ua_ut = ([0.005, 0.001, 0.0005])
        self.case01 = case01_info()
        self.case01.seed = seed
        self.case01.sep_fac = sep_fac
        self.case01.lambda_bar_true = np.max(mu_true_3zones_ua_ut)
        self.lambda_bar_true = np.max(mu_true_3zones_ua_ut)
        self.case01.mu_true_3zones_ua_ut = mu_true_3zones_ua_ut

        abs_T = np.diff(self.domain.T_borders_all)
        X_borders = np.copy(self.domain.X_borders)

        # ====================================================
        # (1) generates bg events in domain (X x T)
        #   with three spatial subregions with constant rate
        #   preparation
        sep_x_fac = sep_fac[0]
        sep_y_fac = sep_fac[1]
        X_borders_sub_1 = np.array(
            [[0, sep_x_fac * X_borders[0, 1]], [sep_y_fac * X_borders[1, 1], X_borders[1, 1]]])
        X_borders_sub_2 = np.array([[sep_x_fac * X_borders[0, 1], X_borders[0, 1]],
                                    [sep_y_fac * X_borders[1, 1], X_borders[1, 1]]])
        X_borders_sub_3 = np.array([[0, X_borders[0, 1]], [0, sep_y_fac * X_borders[1, 1]]])
        three_rates_info_Rsubvec = np.zeros(3) * np.nan
        three_rates_info_Nsubvec = np.zeros(3) * np.nan

        # generation of bg data
        if seed is not None:
            np.random.seed(seed)
        S0_events_tmxyb = None  # bg events with t_ime, m_agnitude, x_pos, y_pos, b_ranching
        subregion_borders = None
        absX_sub = None

        # (1.1) bg in space
        for ii in range(len(mu_true_3zones_ua_ut)):
            mu_i = mu_true_3zones_ua_ut[ii]
            if ii + 1 == 1:
                subregion_borders = np.array(X_borders_sub_1)
                absX_sub = np.prod(np.diff(X_borders_sub_1))  # area of subregion1
            if ii + 1 == 2:
                subregion_borders = np.array(X_borders_sub_2)
                absX_sub = np.prod(np.diff(X_borders_sub_2))  # area of subregion2
            if ii + 1 == 3:
                subregion_borders = np.array(X_borders_sub_3)
                absX_sub = np.prod(np.diff(X_borders_sub_3))  # area of subregion3
            N = int(np.random.poisson(mu_i * abs_T * absX_sub))
            S0part = np.zeros([N, 5])
            S0part[:, 1] = np.random.exponential(1. / m_beta, N) + m0  # m_i: marks
            S0part[:, 2] = np.random.uniform(subregion_borders[0, 0], subregion_borders[0, 1], N)  # x_coord
            S0part[:, 3] = np.random.uniform(subregion_borders[1, 0], subregion_borders[1, 1], N)  # y_coord
            if ii == 0:
                S0_events_tmxyb = np.copy(S0part)
            else:
                S0_events_tmxyb = np.concatenate((S0_events_tmxyb, S0part))
            three_rates_info_Rsubvec[ii] = absX_sub
            three_rates_info_Nsubvec[ii] = N
        # info
        self.case01.three_rates_info_mu_true_3zones_ua_ut = mu_true_3zones_ua_ut
        self.case01.sep_x_fac = sep_x_fac
        self.case01.sep_y_fac = sep_y_fac
        self.case01.X_borders_sub_1 = X_borders_sub_1
        self.case01.X_borders_sub_2 = X_borders_sub_2
        self.case01.X_borders_sub_3 = X_borders_sub_3
        self.case01.three_rates_info_Rsubvec = three_rates_info_Rsubvec
        self.case01.three_rates_info_Nsubvec = three_rates_info_Nsubvec

        # (1.2) bg in time (Homogeneous Poisson -> uniform in time given N)
        S0_events_tmxyb[:, 0] = np.random.uniform(0, abs_T, len(S0_events_tmxyb))  # t_i: event times
        sort_idx = np.argsort(S0_events_tmxyb[:, 0])
        S0_events_tmxyb = S0_events_tmxyb[sort_idx, :]  # ordering in time
        S0_events_tmxyb[:, 0] = S0_events_tmxyb[:, 0] - np.min(S0_events_tmxyb[:, 0])  # set start at 0

        # ====================================================
        # (2) add offspring events (aftershocks)
        S_tmxyb, complete_branching_structure_tmxyzcg = self.sim_Xall_tmxyz_branching(X0_events=S0_events_tmxyb)

        # generate mu_grid
        X_grid = np.copy(self.X_grid)
        mu_grid = np.empty([X_grid.shape[0]]) * np.nan
        for ii in range(X_grid.shape[0]):
            if (X_grid[ii, 0] >= X_borders_sub_1[0][0]) & (X_grid[ii, 0] < X_borders_sub_1[0][1]):
                if (X_grid[ii, 1] >= X_borders_sub_1[1][0]) & (X_grid[ii, 1] <= X_borders_sub_1[1][1]):
                    mu_grid[ii] = mu_true_3zones_ua_ut[0]
            if (X_grid[ii, 0] >= X_borders_sub_2[0][0]) & (X_grid[ii, 0] <= X_borders_sub_2[0][1]):
                if (X_grid[ii, 1] >= X_borders_sub_2[1][0]) & (X_grid[ii, 1] <= X_borders_sub_2[1][1]):
                    mu_grid[ii] = mu_true_3zones_ua_ut[1]
            if (X_grid[ii, 0] >= X_borders_sub_3[0][0]) & (X_grid[ii, 0] <= X_borders_sub_3[0][1]):
                if (X_grid[ii, 1] >= X_borders_sub_3[1][0]) & (X_grid[ii, 1] <= X_borders_sub_3[1][1]):
                    mu_grid[ii] = mu_true_3zones_ua_ut[2]
        self.mu_grid = np.copy(mu_grid)

        # get mu_at_data
        mu_at_data = np.empty([S_tmxyb.shape[0]]) * np.NaN
        xy_data = S_tmxyb[:, 2:4]
        for ii in range(mu_at_data.shape[0]):
            if (xy_data[ii, 0] >= X_borders_sub_1[0][0]) & (xy_data[ii, 0] < X_borders_sub_1[0][1]):
                if (xy_data[ii, 1] >= X_borders_sub_1[1][0]) & (xy_data[ii, 1] <= X_borders_sub_1[1][1]):
                    mu_at_data[ii] = mu_true_3zones_ua_ut[0]
            if (xy_data[ii, 0] >= X_borders_sub_2[0][0]) & (xy_data[ii, 0] <= X_borders_sub_2[0][1]):
                if (xy_data[ii, 1] >= X_borders_sub_2[1][0]) & (xy_data[ii, 1] <= X_borders_sub_2[1][1]):
                    mu_at_data[ii] = mu_true_3zones_ua_ut[1]
            if (xy_data[ii, 0] >= X_borders_sub_3[0][0]) & (xy_data[ii, 0] <= X_borders_sub_3[0][1]):
                if (xy_data[ii, 1] >= X_borders_sub_3[1][0]) & (xy_data[ii, 1] <= X_borders_sub_3[1][1]):
                    mu_at_data[ii] = mu_true_3zones_ua_ut[2]
        self.mu_i_at_data = np.copy(mu_at_data)
        # info
        self.case01.three_rates_info_ll_intensity_part = np.sum(
            three_rates_info_Nsubvec * np.log(mu_true_3zones_ua_ut))
        self.case01.three_rates_info_ll_integral_part = abs_T * np.sum(
            three_rates_info_Rsubvec * mu_true_3zones_ua_ut)
        self.case01.three_rates_info_ll_total = self.case01.three_rates_info_ll_intensity_part - self.case01.three_rates_info_ll_integral_part
        self.S0_events = np.array(S0_events_tmxyb)

        # write to structures
        Xall_events_tmxyzcg = np.array(complete_branching_structure_tmxyzcg)
        self.data_all.times = Xall_events_tmxyzcg[:, 0]
        self.data_all.magnitudes = Xall_events_tmxyzcg[:, 1]
        self.data_all.positions = Xall_events_tmxyzcg[:, 2:4]
        self.data_all.branching = Xall_events_tmxyzcg[:, 4]
        self.data_all.cluster = Xall_events_tmxyzcg[:, 5]
        self.data_all.generation = Xall_events_tmxyzcg[:, 6]
        self.N0 = len(self.S0_events[:, 0])
        self.N = len(Xall_events_tmxyzcg[:, 0])
        self.Noff = self.N - self.N0
        self.integral_mu_x_unit_time = np.copy(self.case01.three_rates_info_ll_integral_part) / abs_T
        self.idx_training = np.where((self.data_all.times >= self.domain.T_borders_training[0]) & (
                self.data_all.times <= self.domain.T_borders_training[1]))

        # write to file
        if save_yes is not None:
            self.write_to_file()
        return

    def sim_case_02(self, m0=3., seed=42, save_yes=None, T_borders_all=None, T_borders_training=None,
                    T_borders_testing=None):
        """
        Generates data according to case1 of the paper.

        :param T_borders_testing:
        :type T_borders_testing:
        :param T_borders_training:
        :type T_borders_training:
        :param T_borders_all:
        :type T_borders_all:
        :param m0: float:
            Lower cutoff magnitude of the magnitude density f_M(m), aka magnitude of completeness
        :param seed: int or None:
            Initial random seed for the simulations. In the paper seed = 42.
        :param save_yes: not None or None:
            Option if simulated data is saved (pickled).
        :return: gm_obj: object
            Contains the simulated data and additional info about the data.
        """
        self.case_name = 'case_02'
        # case individual time domain calT=[tau0,tau1]
        if T_borders_all is not None:
            self.domain.T_borders_all = T_borders_all
        if T_borders_training is not None:
            self.domain.T_borders_training = T_borders_training
        if T_borders_testing is not None:
            self.domain.T_borders_testing = T_borders_testing
        if m0 is None:
            m0 = self.domain.m0
        if m0 is not None:
            self.domain.m0 = m0
        mu_vec_Fi__ua_ut = np.array([0.07, 0.07, 0.035])
        A0_yes = 1
        mu_A0 = 0.00035
        nbins = np.copy(self.nbins)
        m_beta = self.m_beta
        self.case02 = case02_info()
        self.case02.m0 = m0
        self.case02.lambda_bar_true = np.max(mu_vec_Fi__ua_ut)
        self.lambda_bar_true = np.max(mu_vec_Fi__ua_ut)
        self.case02.seed = seed
        self.case02.mu_vec_Fi__ua_ut = np.copy(mu_vec_Fi__ua_ut)
        self.case02.A0_yes = A0_yes
        self.case02.mu_A0 = mu_A0
        self.case02.three_rates_info_mu_A0 = mu_A0

        # some variables
        abs_T = np.diff(self.domain.T_borders_all)
        X_borders = np.copy(self.domain.X_borders)
        abs_X = np.prod(np.diff(self.domain.X_borders))
        S = self.domain.X_borders[:, 1] - self.domain.X_borders[:, 0]

        # (1) bg events: generates events in area R, with 3 different constant background rates
        #   in 3 fault regions, and a low overall intensity elsewhere of mu_A0 boundary effects are neglected.
        three_rates_info_mu_vec_Fi__ua_ut = mu_vec_Fi__ua_ut

        # BG HPP
        if seed is not None:
            np.random.seed(seed)
        S0_events_tmxyb = None

        # 3 Faults
        #   x \in [x1,x2]
        #   y \in [y -+ 0.5 faultwidth,y]
        width_grid_line = S[0] / nbins
        F_1 = np.array([[1, 1.500], [3, 1.500]]) * S[0] / 5. - width_grid_line / 2.
        F_2 = np.array([[1, 2.5], [4, 2.5]]) * S[0] / 5. - width_grid_line / 2.
        F_3 = np.array([[2, 4], [3, 4]]) * S[0] / 5. - width_grid_line / 2.
        F_width = 0.5 * S[0] / nbins - 1e-12  # HERE it is HALF-Width of the Fault
        F_array = (F_1, F_2, F_3)

        # generate mu_grid
        X_grid = np.copy(self.X_grid)
        mu_grid = np.zeros([X_grid.shape[0]]) #* np.nan
        if A0_yes is not None:
            mu_grid = mu_grid + mu_A0
        grid_ij = np.reshape(mu_grid, [nbins, nbins])  # [z,y,x]

        three_Faults_X_borders_F1 = np.zeros([2, 2]) * np.nan
        three_Faults_X_borders_F2 = np.zeros([2, 2]) * np.nan
        three_Faults_X_borders_F3 = np.zeros([2, 2]) * np.nan
        for i in range(len(mu_vec_Fi__ua_ut)):
            mu_i = mu_vec_Fi__ua_ut[i]

            y_l = np.array(F_array[i])[0, 1] - F_width
            y_u = np.array(F_array[i])[0, 1] + F_width
            iy_l = (np.ceil(y_l * nbins / np.sqrt(abs_X)) - 1).astype(int)
            iy_u = (np.ceil(y_u * nbins / np.sqrt(abs_X)) - 1).astype(int)

            x_l = np.array(F_array[i])[0, 0] + width_grid_line / 2.  # +0.001
            x_u = np.array(F_array[i])[1, 0] + width_grid_line / 2.  # -0.001
            jx_l = (np.ceil(x_l * nbins / np.sqrt(abs_X)) - 1).astype(int)
            jx_u = (np.ceil(x_u * nbins / np.sqrt(abs_X)) - 1).astype(int)

            if iy_l == iy_u:
                grid_ij[iy_l, jx_l + 1:jx_u + 1] = mu_vec_Fi__ua_ut[i]
                if A0_yes is not None:
                    grid_ij[iy_l, jx_l + 1:jx_u + 1] = mu_vec_Fi__ua_ut[i] + mu_A0
            else:
                grid_ij[iy_l:iy_u, jx_l + 1:jx_u + 1] = mu_vec_Fi__ua_ut[i]
                if A0_yes is not None:
                    grid_ij[iy_l:iy_u, jx_l + 1:jx_u + 1] = mu_vec_Fi__ua_ut[i] + mu_A0

            if i == 0:
                three_Faults_X_borders_F1[0, :] = np.array([x_l, x_u])
                three_Faults_X_borders_F1[1, :] = np.array([y_l, y_u])
            if i == 1:
                three_Faults_X_borders_F2[0, :] = np.array([x_l, x_u])
                three_Faults_X_borders_F2[1, :] = np.array([y_l, y_u])
            if i == 2:
                three_Faults_X_borders_F3[0, :] = np.array([x_l, x_u])
                three_Faults_X_borders_F3[1, :] = np.array([y_l, y_u])

            R_sub = (x_u - x_l) * (y_u - y_l)
            if A0_yes is not None:
                N = np.random.poisson((mu_i + mu_A0) * abs_T * R_sub)
            else:
                N = np.random.poisson(mu_i * abs_T * R_sub)
            new_events = np.zeros([int(N), 5])
            new_events[:, 1] = np.random.exponential(1. / m_beta, N) + m0  # m_i: marks
            new_events[:, 2] = np.random.uniform(x_l, x_u, N)  # x_coord
            new_events[:, 3] = np.random.uniform(y_l, y_u, N)  # y_coord

            if i == 0:
                S0_events_tmxyb = np.copy(new_events)
            else:
                S0_events_tmxyb = np.concatenate((S0_events_tmxyb, new_events))

        # info
        self.case02.three_Faults_X_borders_F1 = three_Faults_X_borders_F1
        self.case02.three_Faults_X_borders_F2 = three_Faults_X_borders_F2
        self.case02.three_Faults_X_borders_F3 = three_Faults_X_borders_F3

        # reshaping mu_grid
        mu_grid = np.reshape(grid_ij, [nbins * nbins])  # [z,y,x]

        # total area intensity
        if A0_yes is not None:
            N = np.random.poisson(mu_A0 * abs_T * abs_X)
            new_events = np.zeros([int(N), 5])
            new_events[:, 1] = np.random.exponential(1. / m_beta, N) + m0  # m_i: marks
            new_events[:, 2] = np.random.uniform(X_borders[0, 0], X_borders[0, 1], N)  # x_coord
            new_events[:, 3] = np.random.uniform(X_borders[1, 0], X_borders[1, 1], N)  # y_coord
            A0_events_tmxyb = np.copy(new_events)
            S0_events_tmxyb = np.concatenate((S0_events_tmxyb, A0_events_tmxyb))

        # event times as HPP in abs_T
        S0_events_tmxyb[:, 0] = np.random.uniform(0, abs_T, len(S0_events_tmxyb))  # t_i: event times

        # ordering in time
        sort_idx = np.argsort(S0_events_tmxyb[:, 0])
        S0_events_tmxyb = S0_events_tmxyb[sort_idx, :]
        S0_events_tmxyb[:, 0] = S0_events_tmxyb[:, 0] - np.min(S0_events_tmxyb[:, 0])  # set start at 0
        self.S0_events = np.array(S0_events_tmxyb)

        # (2) Adding offspring events, branching
        S_tmxyb, complete_branching_structure_tmxyzcg = self.sim_Xall_tmxyz_branching(X0_events=S0_events_tmxyb)

        # get mu at ALL data
        mu_at_data = np.zeros([S_tmxyb.shape[0]])
        if A0_yes is not None:
            mu_at_data = np.ones([S_tmxyb.shape[0]]) * mu_A0

        xy_data = S_tmxyb[:, 2:4]
        for ii in range(mu_at_data.shape[0]):
            if (xy_data[ii, 0] >= three_Faults_X_borders_F1[0][0]) & (
                    xy_data[ii, 0] < three_Faults_X_borders_F1[0][1]):
                if (xy_data[ii, 1] >= three_Faults_X_borders_F1[1][0]) & (
                        xy_data[ii, 1] <= three_Faults_X_borders_F1[1][1]):
                    mu_at_data[ii] = mu_vec_Fi__ua_ut[0]
                    if A0_yes is not None:
                        mu_at_data[ii] = mu_vec_Fi__ua_ut[0] + mu_A0
            if (xy_data[ii, 0] >= three_Faults_X_borders_F2[0][0]) & (
                    xy_data[ii, 0] <= three_Faults_X_borders_F2[0][1]):
                if (xy_data[ii, 1] >= three_Faults_X_borders_F2[1][0]) & (
                        xy_data[ii, 1] <= three_Faults_X_borders_F2[1][1]):
                    mu_at_data[ii] = mu_vec_Fi__ua_ut[1]
                    if A0_yes is not None:
                        mu_at_data[ii] = mu_vec_Fi__ua_ut[1] + mu_A0
            if (xy_data[ii, 0] >= three_Faults_X_borders_F3[0][0]) & (
                    xy_data[ii, 0] <= three_Faults_X_borders_F3[0][1]):
                if (xy_data[ii, 1] >= three_Faults_X_borders_F3[1][0]) & (
                        xy_data[ii, 1] <= three_Faults_X_borders_F3[1][1]):
                    mu_at_data[ii] = mu_vec_Fi__ua_ut[2]
                    if A0_yes is not None:
                        mu_at_data[ii] = mu_vec_Fi__ua_ut[2] + mu_A0

        # write data to the class
        self.mu_i_at_data = np.copy(mu_at_data)
        self.S0_events = np.array(S0_events_tmxyb)
        self.mu_grid = np.array(mu_grid)

        # analytical integral part PAPER: 3 faults with 0.1 width and
        A_f1 = 0.1 * 2.
        A_f2 = 0.1 * 3.
        A_f3 = 0.1 * 1.
        A_vec = np.array([A_f1, A_f2, A_f3])
        self.case02.integral_BG_analytical = abs_X * mu_A0 + np.sum(A_vec * three_rates_info_mu_vec_Fi__ua_ut)
        self.case02.A_vec = np.copy(A_vec)

        # write to structures
        Xall_events_tmxyzcg = np.array(complete_branching_structure_tmxyzcg)
        self.data_all.times = Xall_events_tmxyzcg[:, 0]
        self.data_all.magnitudes = Xall_events_tmxyzcg[:, 1]
        self.data_all.positions = Xall_events_tmxyzcg[:, 2:4]
        self.data_all.branching = Xall_events_tmxyzcg[:, 4]
        self.data_all.cluster = Xall_events_tmxyzcg[:, 5]
        self.data_all.generation = Xall_events_tmxyzcg[:, 6]
        self.N0 = len(self.S0_events[:, 0])
        self.N = len(Xall_events_tmxyzcg[:, 0])
        self.Noff = self.N - self.N0
        self.idx_training = np.where((self.data_all.times >= self.domain.T_borders_training[0]) & (
                self.data_all.times <= self.domain.T_borders_training[1]))
        self.integral_mu_x_unit_time = np.copy(self.case02.integral_BG_analytical)

        # write to file
        if save_yes is not None:
            self.write_to_file()
        return

    def sim_case_03_gaborlike(self, seed=40, m0=None, save_yes=None,
                              T_borders_all=None, T_borders_training=None, T_borders_testing=None):
        """
        Generates data according to case1 of the paper. (an altered Gabor function for the background intensity.)
        :param seed: int or None:
             Initial random seed for the simulations.
        :param m0: float:
            Lower cutoff magnitude of the magnitude density f_M(m), aka magnitude of completeness.
        :param save_yes: not None or None:
             Option if simulated data is saved (pickled).
        :return: gm_obj: object:
             Contains the simulated data and additional info about the simulation.
         """

        self.case_name = 'case_03'
        # case individual time domain calT=[tau0,tau1]
        if T_borders_all is not None:
            self.domain.T_borders_all = T_borders_all
        if T_borders_training is not None:
            self.domain.T_borders_training = T_borders_training
        if T_borders_testing is not None:
            self.domain.T_borders_testing = T_borders_testing
        # case individual m0
        if m0 is not None:
            self.domain.m0 = m0
        if m0 is None:
            m0 = self.domain.m0
        m_beta = self.m_beta
        self.case03 = case03_info()
        self.case03.seed = seed
        abs_T = np.diff(self.domain.T_borders_all)
        abs_X = np.prod(np.diff(self.domain.X_borders))
        X_borders = np.copy(self.domain.X_borders)

        # bg events
        if seed is not None:
            np.random.seed(seed)

        # generate background events from a gabor like function
        # generate candidates
        mu_max = 0.1
        N = int(np.random.poisson(mu_max * abs_T * abs_X))
        positions = np.random.rand(N, 2) * np.diff(X_borders).reshape(1, 2) + X_borders[:, 0]

        # thinning
        Ui = np.random.rand(N) * mu_max  # np.random.uniform(0,mu_max,N)
        mu_i = self.gabor_like_intensity(positions)
        idx_thinned = (Ui <= mu_i)
        self.N0 = np.sum(idx_thinned)

        # construction marked background events
        S0_events_tmxyb = np.zeros([self.N0, 5]) * np.nan
        S0_events_tmxyb[:, 0] = np.random.uniform(0, abs_T, self.N0)  # t_i: event times uniform
        S0_events_tmxyb[:, 1] = np.random.exponential(1. / m_beta,
                                                      self.N0) + m0  # m_i: marks from GR exponential distribution
        S0_events_tmxyb[:, 2] = np.copy(positions[idx_thinned, 0])
        S0_events_tmxyb[:, 3] = np.copy(positions[idx_thinned, 1])
        S0_events_tmxyb[:, 4] = np.zeros(self.N0, dtype=int)
        # ordering in time
        sort_idx = np.argsort(S0_events_tmxyb[:, 0])
        S0_events_tmxyb = S0_events_tmxyb[sort_idx, :]
        S0_events_tmxyb[:, 0] = S0_events_tmxyb[:, 0] - np.min(S0_events_tmxyb[:, 0])  # set start at 0
        self.S0_events = np.array(S0_events_tmxyb)

        # generate offspring events given background events from above: branching
        S_tmxyb, complete_branching_structure_tmxyzcg = self.sim_Xall_tmxyz_branching(X0_events=S0_events_tmxyb)

        # write to structures
        Xall_events_tmxyzcg = np.array(complete_branching_structure_tmxyzcg)
        self.data_all.times = Xall_events_tmxyzcg[:, 0]
        self.data_all.magnitudes = Xall_events_tmxyzcg[:, 1]
        self.data_all.positions = Xall_events_tmxyzcg[:, 2:4]
        self.data_all.branching = Xall_events_tmxyzcg[:, 4]
        self.data_all.cluster = Xall_events_tmxyzcg[:, 5]
        self.data_all.generation = Xall_events_tmxyzcg[:, 6]
        self.N = len(Xall_events_tmxyzcg[:, 0])
        self.Noff = self.N - self.N0
        self.idx_training = np.where((self.data_all.times >= self.domain.T_borders_training[0]) & (
                self.data_all.times <= self.domain.T_borders_training[1]))

        # get mu_grid for integral approx = |X|/L*sum(mu_l), l=1,...,NperDim^2
        self.mu_grid = self.gabor_like_intensity(self.X_grid)
        self.integral_mu_x_unit_time = self.integral_ut_gabor_like_intensity(res=1000)

        # get mu_i at data positions for product of rates at data points for the likehood computation
        self.mu_i_at_data = self.gabor_like_intensity(self.data_all.positions)

        # lambda bar true
        self.case03.lambda_bar_true = self.gabor_like_intensity(
            np.array([self.gabor_mean_shift, self.gabor_mean_shift]).reshape(1, 2))  # (1.+0.0001)/100
        self.lambda_bar_true = np.copy(self.case03.lambda_bar_true)

        ''' write to file '''
        if save_yes is not None:
            self.write_to_file()
        return

    def sim_Xall_tmxyz_branching(self, X0_events):
        """
        Adds offspring events (aftershocks) to a list of N0 bg events.
        :param X0_events: array (N0,5)
            Background events (N0:number of bg events), cols: time,mag,x,y,branching
        """

        abs_T = np.diff(self.domain.T_borders_all)
        X_borders = np.copy(self.domain.X_borders)
        m_beta = self.m_beta
        m0 = self.domain.m0
        K, c, p, m_alpha, D, gamma, q = np.zeros(7) * np.nan

        # params and SPATIAL OFFSPRING
        if self.spatial_offspring == 'G':
            K, c, p, m_alpha, D = self.theta_true_Kcpadgq[:5]
            print(K, c, p, m_alpha, D, gamma, q, m_beta, m0, self.spatial_offspring)
        if self.spatial_offspring == 'P':
            K, c, p, m_alpha, D, gamma, q = self.theta_true_Kcpadgq
            print(K, c, p, m_alpha, D, gamma, q, m_beta, m0, self.spatial_offspring)
        if self.spatial_offspring == 'R':
            K, c, p, m_alpha, D, gamma, q = self.theta_true_Kcpadgq
            print(K, c, p, m_alpha, D, gamma, q, m_beta, m0, self.spatial_offspring)

        # stability info, branching ratio
        self.n_stability = self.n(m_alpha, m_beta, K, c, p, t_start=0., t_end=abs_T)
        self.n_stability_inf = self.n(m_alpha, m_beta, K, c, p, t_start=0., t_end=np.inf)
        #print('n', self.n_stability, self.n_stability_inf)

        x0 = X0_events[:, 0:4]
        idx = 0
        z_parent = np.zeros((np.size(x0[:, 0]), 1))
        z_cluster = np.array(np.arange(np.size(x0[:, 0])) + 1)  # numbered clusters from 1,...,len(x)
        z_generation = np.zeros((np.size(x0[:, 0]), 1))  # zeros
        # intensity function of the NHPP in time with arg t = t-t_i
        lam_MOL = lambda t, m_i, K, c, p, m_alpha, m0: K * np.exp(m_alpha * (m_i - m0)) * 1. / (c + t) ** p

        x = x0
        while idx < np.size(x[:, 0]):
            # if np.mod(idx, 1000) == 0: print('current event=', idx + 1)
            t_i = x[idx, 0]
            m_i = x[idx, 1]
            x_i = x[idx, 2]
            y_i = x[idx, 3]
            fparams = np.array([m_i, K, c, p, m_alpha, m0])
            max_lambda = lam_MOL(0., m_i, K, c, p, m_alpha, m0)
            # get new children
            xnew = self.sim_NHPP(0., abs_T - t_i, lam_MOL, fparams, max_lambda) + t_i
            coord_xy_new = None

            if xnew.size != 0:
                Nnew = np.size(xnew)
                id_cluster = z_cluster[idx]
                id_generation = int(z_generation[idx])
                xnew = np.vstack((xnew, np.random.exponential(1. / m_beta, np.size(xnew)) + m0)).transpose()

                # s(x-x_i)
                # (1) GAUSS (short range)
                # if len(self.theta) == 7:
                if self.spatial_offspring == 'G':
                    coord_xy_new = self.sample_s_xy_short_range(mean=np.array([x_i, y_i]),
                                                                cov=np.eye(2) * D ** 2. * np.exp(m_alpha * (m_i - m0)),
                                                                N=Nnew)
                    # (2a) Power Law decay (long  range)
                # if len(self.theta) == 9 and self.spatial_kernel[0] == 'P':
                if self.spatial_offspring == 'P':
                    coord_xy_new = self.sample_long_range_decay_pwl(N=Nnew, x_center=x_i, y_center=y_i, m_center=m_i,
                                                                    q=q, D=D, gamma_m=gamma, m0=m0)

                    # (2b) Rupture Length Power Law decay (long  range)
                # if len(self.theta) == 9 and self.spatial_kernel[0] == 'R':
                if self.spatial_offspring == 'R':
                    coord_xy_new = self.sample_long_range_decay_RL_pwl(N=Nnew, x_center=x_i, y_center=y_i, m_center=m_i,
                                                                       q=q, D=D, gamma_m=gamma)

                    # check if xnew is inside domain S
                xx = np.copy(coord_xy_new)
                xmin, xmax = X_borders[0, :]
                ymin, ymax = X_borders[1, :]
                idx_in = np.logical_and(xx[:, 0] > xmin, xx[:, 0] < xmax) * np.logical_and(xx[:, 1] > ymin,
                                                                                           xx[:, 1] < ymax)
                xnew = np.hstack((xnew, coord_xy_new))
                xnew = xnew[idx_in, :]
                Nnew = np.size(xnew[:, 0])

                # update branching structure: z variables
                z_parent = np.append(z_parent, idx * np.ones((Nnew, 1)) + 1)
                z_cluster = np.append(z_cluster, id_cluster * np.ones((Nnew, 1)))
                z_generation = np.append(z_generation, np.zeros((Nnew, 1)) + id_generation + 1.)

                x = np.vstack((x, xnew))

            idx = idx + 1

        # sorting wrt time=x[:,0]
        parents_sorted_t = np.zeros([x[:, 0].size, 1])
        x_sorted_t = np.empty([x[:, 0].size, 4])

        id_unsorted = np.arange(1, np.size(x[:, 0]) + 1)
        idx = np.argsort(x[:, 0])
        x_sorted_t[:, :] = x[idx, :]
        z_parent_sorted_t = z_parent[idx]
        z_cluster_sorted_t = z_cluster[idx]
        z_generation_sorted_t = z_generation[idx]
        # re-ordering from id_unsorted in time to sorted in time
        for ii in z_parent_sorted_t[z_parent_sorted_t > 0]:
            idz_pa = z_parent_sorted_t == ii
            idd = id_unsorted[id_unsorted[idx] == ii]
            parents_sorted_t[idz_pa] = idd

        # generate data tmxyz
        data_tmxyz = np.zeros((np.size(x_sorted_t[:, 0]), 5)) * np.nan
        data_tmxyz[:, 0:4] = x_sorted_t
        data_tmxyz[:, 4] = np.squeeze(parents_sorted_t).astype(int)
        complete_branching_structure_tmxyzcg = np.zeros((np.size(x_sorted_t[:, 0]), 8)) * np.nan
        complete_branching_structure_tmxyzcg[:, 0:5] = data_tmxyz[:, 0:5]
        complete_branching_structure_tmxyzcg[:, 5] = z_cluster_sorted_t
        complete_branching_structure_tmxyzcg[:, 6] = z_generation_sorted_t.reshape([-1, ])
        complete_branching_structure_tmxyzcg[:, 7] = np.array(np.arange(np.size(x_sorted_t[:, 0])) + 1)
        return data_tmxyz, complete_branching_structure_tmxyzcg

    # stability functions
    def PHI_t_omori(self, c, p, t_start=0., t_end=np.inf):
        PHI_t = np.nan
        if (p > 1. and t_end == np.inf):
            PHI_t = 1. / (p - 1) * (c + t_start) ** (1. - p)
        if (p > 1. and t_end != np.inf):
            PHI_t = 1. / (-p + 1) * ((c + t_end) ** (1. - p) - (c + t_start) ** (1. - p))
        if (p == 1 and t_end != np.inf):
            PHI_t = np.log(c + t_end) - np.log(c + t_start)
        return (PHI_t)

    def n(self, m_alpha, m_beta, K, c, p, t_start=0., t_end=np.inf):
        n_t = self.PHI_t_omori(c, p, t_start, t_end)
        n = n_t * K / (1. - m_alpha / m_beta)
        return (n)

    # simulation inhom. Poisson process
    def sim_NHPP(self, t1, t2, lambda_t, fargs, max_lambda):
        """
        Simulates a 1D non-homogenous Poisson process (NHPP) via thinning.

        :param t1: float:
            Starting time of the simulation.
        :param t2: float:
            Ending time of the simulation.
        :param lambda_t: str, name of a function:
            Definition of the 1D intensity function. Here Mod.Omori law.
        :param fargs:
        :type fargs:
        :param max_lambda: float:
            Maximum value of the intensity function in [t1,t2] of the NHPP.
        :return: array (N,1):
            Time series of events of the NHPP in [t1,t2].
        """
        ti_event = np.array([])
        ti_cand = np.array([0.])
        while np.max(ti_cand) <= t2:
            tau_i = np.random.exponential(scale=1. / max_lambda, size=1)
            ti_cand = np.append(ti_cand, np.max(ti_cand) + tau_i)
            # thinning
            if (np.random.rand(1) <= lambda_t(ti_cand[-1], *fargs) / max_lambda):
                ti_event = np.append(ti_event, np.max(ti_cand))
            # lambda(t) is monotone decreasing all the time
            max_lambda = lambda_t(ti_cand[-1], *fargs)
        if ti_event.size != 0:
            ti_event = ti_event[(ti_event > t1) & (ti_event < t2)]
        return (ti_event)

    def sample_s_xy_short_range(self, mean, cov, N):
        xy_offspring = np.random.multivariate_normal(mean, cov, N)
        return (xy_offspring)

    def sample_long_range_decay_pwl(self, N, x_center, y_center, m_center, q, D, gamma_m, m0):
        sigma_mi = D ** 2 * np.exp(gamma_m * (m_center - m0))
        # sample theta: angle
        theta_j = np.random.uniform(0, 2. * np.pi, N)
        # sample r: radius from center
        u_j = np.random.uniform(0, 1, N)
        r = np.sqrt((1. / (1. - u_j) ** (-1. / (1. - q)) - 1.)) * np.sqrt(sigma_mi)  # returns non-negative sqrt
        x_vec = x_center + r * np.cos(theta_j)
        y_vec = y_center + r * np.sin(theta_j)
        xy_array = np.vstack((x_vec, y_vec)).T
        return xy_array

    def sample_long_range_decay_RL_pwl(self, N, x_center, y_center, m_center, q, D, gamma_m):
        sigma_mi = D ** 2 * 10 ** (2 * gamma_m * m_center)
        # sample theta: angle
        theta_j = np.random.uniform(0, 2. * np.pi, N)
        # sample r: radius from center
        u_j = np.random.uniform(0, 1, N)
        r = np.sqrt((1. / (1. - u_j) ** (-1. / (1. - q)) - 1.)) * np.sqrt(sigma_mi)  # returns non-negative sqrt
        x_vec = x_center + r * np.cos(theta_j)
        y_vec = y_center + r * np.sin(theta_j)
        xy_array = np.vstack((x_vec, y_vec)).T
        return xy_array

    def gabor_like_intensity(self, positions):
        """ my intensity function case04"""
        self.gabor_mean_shift = np.diff(self.domain.X_borders[0, :]) / 2.
        z = (self.pos_gabor(positions - self.gabor_mean_shift, theta=2. * (np.pi / 4.), freq=1 / 3, sigma=1.5) + 0.0001) \
            / 100.
        return z

    def pos_gabor(self, positions, freq=1., theta=0., psi=0., sigma=1., gamma=1.):
        ''' gabor function type '''
        x1 = positions[:, 0]
        x2 = positions[:, 1]

        # rotation
        x1_prime = x1 * np.cos(theta) + x2 * np.sin(theta)
        x2_prime = -x1 * np.sin(theta) + x2 * np.cos(theta)

        # function eval
        lmbda = 1. / freq  # wavelength
        z = np.exp(-1 / (2 * sigma ** 2.) * (x1_prime ** 2. + gamma ** 2. * x2_prime ** 2.)) * np.cos(
            2 * np.pi / lmbda * x1_prime + psi) ** 2.

        return z

    def integral_ut_gabor_like_intensity(self, res=50):
        absX = np.prod(np.diff(self.domain.X_borders))
        L = res ** 2.
        pos = make_X_grid(self.domain.X_borders, nbins=res)
        mu_grid = self.gabor_like_intensity(pos)
        integral_mu_x_unit_time = absX / L * np.sum(mu_grid)
        return integral_mu_x_unit_time

    def write_to_file(self):
        """
        Saves simulated data into two files, and generates plots of the data.
        1: catalog in a numpy.savetxt file with format (idx xlon ylat magnitude time)
        2: save_obj_gm
        """
        # create subdirectory for output
        out_dir = "synthetic_data_examples"
        if os.path.isdir(out_dir):
            print(out_dir + ' subdirectory exists')
        else:
            os.mkdir(out_dir)
            print(out_dir + ' subdirectory has been created.')

        # write to file: catalog
        write_data = np.zeros((len(self.data_all.times), 5))
        write_data[:, 0] = np.arange(len(self.data_all.times)) + 1
        write_data[:, 1] = self.data_all.positions[:, 0]
        write_data[:, 2] = self.data_all.positions[:, 1]
        write_data[:, 3] = self.data_all.magnitudes
        write_data[:, 4] = self.data_all.times
        fname = "./" + out_dir + "/%s_data_sim_3rates.cat" % (self.case_name)
        np.savetxt(fname, X=write_data, delimiter='\t', fmt='%.0f\t%.4f\t%.4f\t%.2f\t%.6f')

        # write to file save_obj_gm
        file = open("./" + out_dir + "/%s_gm.all" % (self.case_name), "wb")  # remember to open the file in binary mode
        pickle.dump(self, file)
        file.close()

        # generates plots of the data (Figure 2 in the paper)
        hf1, hf2 = self.plot_paper_figure02_data()
        hf1.savefig(out_dir + '/%s_data_01.pdf' % self.case_name, bbox_inches='tight')
        hf2.savefig(out_dir + '/%s_data_02.pdf' % self.case_name, bbox_inches='tight')

    def plot_paper_figure02_data(self):
        # plot settings
        pSIZE = 30
        plt.rc('font', size=pSIZE)
        plt.rc('axes', titlesize=pSIZE)

        # idx BG events
        idx_BG = (self.data_all.branching == 0)

        ''' as: dot clouds '''
        hf1 = plt.figure(figsize=(15, 7))
        grid = plt.GridSpec(1, 2)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.01)
        ax = plt.subplot(1, 2, 1)
        plt.plot(self.data_all.positions[idx_BG, 0], self.data_all.positions[idx_BG, 1], 'k.')
        # plt.scatter(xx[:,2],xx[:,3],marker='o',linewidths=1,s=0.001*np.exp(1.75*xx[:,1].astype(float))*xx[:,1].astype(float),facecolors='none', edgecolors='k')
        plt.xlabel('x')
        plt.ylabel('y')
        ticks = np.linspace(0, np.max(self.domain.X_borders), 6)
        plt.xticks(ticks)
        plt.yticks(ticks)
        ax.set_yticklabels([int(ticks[0]), '', int(ticks[2]), '', int(ticks[4]), ''])
        ax.set_xticklabels([int(ticks[0]), '', int(ticks[2]), '', int(ticks[4]), ''])
        ax.tick_params(direction='out', left=True, right=True, top=True, bottom=True)
        plt.axis('square')
        ax = plt.subplot(1, 2, 2)
        plt.plot(self.data_all.positions[:, 0], self.data_all.positions[:, 1], 'k.')
        plt.xlabel('x')
        ticks = np.linspace(0, np.max(self.domain.X_borders), 6)
        plt.xticks(ticks)
        plt.yticks(ticks)
        ax.set_yticklabels([int(ticks[0]), '', int(ticks[2]), '', int(ticks[4]), ''])
        ax.set_xticklabels([int(ticks[0]), '', int(ticks[2]), '', int(ticks[4]), ''])
        ax.set_yticklabels([])
        ax.tick_params(direction='out', left=True, right=True, top=True, bottom=True)
        plt.axis('square')
        # plt.show(block=False)

        ''' as: earthquake sequence '''
        hf2 = plt.figure(figsize=(15, 7))
        plt.subplots_adjust(hspace=0.1)
        ax = plt.subplot(2, 1, 1)
        hl01 = plt.plot(self.data_all.times, self.data_all.magnitudes, '.k', markersize=5)
        plt.ylim((np.min(self.data_all.magnitudes) - 0.5, np.max(self.data_all.magnitudes) + 0.5))
        plt.xlim((np.min(self.data_all.times), np.max(self.data_all.times)))
        plt.ylabel('magnitude');
        ax = plt.gca()
        ax.yaxis.set_label_position("right")
        plt.xticks(ticks=([0, 1000, 2000, 3000, 4000]))
        ax.tick_params(direction='out', left=True, right=True, top=True, bottom=True, labelright=True,
                       labelleft=False)
        ax.set_xticklabels([])

        ax = plt.subplot(2, 1, 2)
        h1 = plt.step(np.concatenate([[0.], self.data_all.times]),
                      np.arange(0, self.data_all.times.shape[0] + 1, 1),
                      'k', linewidth=3)
        plt.xlim((np.min(self.data_all.times), np.max(self.data_all.times)))
        plt.ylabel('counts')
        ax = plt.gca()
        ax.yaxis.set_label_position("right")
        plt.xlabel('time, days')
        plt.xticks(ticks=([0, 1000, 2000, 3000, 4000]))
        plt.text(np.min(self.data_all.times) + 50, len(self.data_all.magnitudes) - 0,
                 '$N_{D}$ = %s, $m\in$[%.2f,%.2f]\n$N_{D_0}$= %i'
                 % (
                 len(self.data_all.times), np.min(self.data_all.magnitudes), np.max(self.data_all.magnitudes),
                 sum(idx_BG)),
                 horizontalalignment='left', verticalalignment='top', fontsize=30)
        ax.tick_params(direction='out', left=True, right=True, top=True, bottom=True, labelright=True,
                       labelleft=False)
        # plt.show(block=False)
        return hf1, hf2


def make_X_grid(X_borders, nbins=50, D=2):
    """
    Generates positions of a regular (nbins x nbins) grid.

    :param X_borders: array (D,2)
        Domain of the grid as np.array([[d1_min, d1_max],... ,[D_min, D_max]]).
    :param nbins: int
        Number of bins in each dimension.
    :param D: int
        Dimension.
    :return: array (nbins**2,D)
        X_grid (meshgrid) positions.
    """
    S = X_borders[:, 1] - X_borders[:, 0]
    grid_points = nbins
    X_grid = np.empty([grid_points, D])  # * np.nan
    for di in range(D):
        X_grid[:, di] = np.linspace(0, S[di], grid_points)
    X_mesh = np.meshgrid(*X_grid.T.tolist())
    X_mesh = np.array(X_mesh).reshape([D, -1]).T  # (N,2 array: 2D points)
    X_grid = X_mesh + X_borders[:, 0][np.newaxis]
    return X_grid


# subclasses
class dom():
    def __init__(self):
        self.T_borders_all = None
        self.T_borders_training = None
        self.T_borders_testing = None
        self.X_borders = None
        self.X_borders_UTM_km = None
        self.X_borders_original = None
        self.m0 = None


class obs():
    def __init__(self):
        self.times = None
        self.magnitudes = None
        self.positions = None
        self.positions_UTM_km = None
        self.branching = None
        self.cluster = None
        self.generation = None


class case01_info():
    def __init__(self):
        self.seed = None
        self.mu_true_3zones_ua_ut = None
        self.sep_fac = None
        self.lambda_bar_true = None
        self.sep_x_fac = None
        self.sep_y_fac = None
        self.X_borders_sub_1 = None
        self.X_borders_sub_2 = None
        self.X_borders_sub_3 = None
        self.three_rates_info_Rsubvec = None
        self.three_rates_info_Nsubvec = None
        self.three_rates_info_mu_true_3zones_ua_ut = None
        self.three_rates_info_ll_total = None
        self.three_rates_info_ll_intensity_part = None
        self.three_rates_info_ll_integral_part = None


class case02_info():
    def __init__(self):
        self.seed = None
        self.m0 = None
        self.mu_vec_Fi__ua_ut = None
        self.lambda_bar_true = None
        self.mu_A0 = None
        self.A0_yes = None
        self.three_rates_info_Rsubvec = None
        self.three_rates_info_Nsubvec = None
        self.three_rates_info_mu_vec_Fi__ua_ut = None
        self.three_rates_info_mu_A0 = None
        self.three_rates_info_ll_total = None
        self.three_rates_info_ll_intensity_part = None
        self.three_rates_info_ll_integral_part = None
        self.three_Faults_X_borders_F1 = None
        self.three_Faults_X_borders_F2 = None
        self.three_Faults_X_borders_F3 = None
        self.A_vec = None
        self.integral_BG_analytical = None


class case03_info():
    def __init__(self):
        self.seed = None
