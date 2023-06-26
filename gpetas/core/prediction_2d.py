import numpy as np
from scipy.linalg import solve_triangular
import scipy as sc
from scipy.stats import nbinom
from scipy import stats
import time
from gpetas.utils.some_fun import get_grid_data_for_a_point
import gpetas
import os
import sys
import pickle
import datetime
import matplotlib.pyplot as plt

# some globals
time_format = "%Y-%m-%d %H:%M:%S.%f"
output_dir = "output_pred"
output_dir_tables = "output_pred/tables"
output_dir_figures = "output_pred/figures"
output_dir_data = "output_pred/data"


# bock/enable print()
# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def sample_from_truncated_exponential_rv(beta, a, b=None, sample_size=1, seed=None):
    if seed is not None:
        np.random.seed(seed)
    if b is None:
        b = np.inf
    X = stats.truncexpon(b=beta * (b - a), loc=a, scale=1. / beta)
    return X.rvs(sample_size)


class setup_pred():
    def __init__(self, save_obj_GS=None, tau1=None, tau2=None, tau0_Ht=None,
                 Ksim=None, sample_idx_vec=None,
                 mle_obj=None, mle_obj_silverman=None, epsilon_after_mainshock=1e-6,
                 m_max=None):
        """
        Generates setup_obj_pred for T*=[tau1,tau2] based on inference results saved in corresponding objects
        :param save_obj_GS:
        :type save_obj_GS: Python class
        :param tau1: start time of prediction
        :type tau1: float
        :param tau2: end time of prediction
        :type tau2: float
        :param tau0_Ht: time before tau1 for which Ht is considered, e.g. tau1-100. in days
        :type tau0_Ht: float
        :param Ksim: number of simulations
        :type Ksim: int
        :param mle_obj:
        :type mle_obj: Python class
        :param mle_obj_silverman:
        :type mle_obj_silverman: Python class
        :param epsilon_after_mainshock: small value of time after tau1, e.g. 1e-4
        :type epsilon_after_mainshock: float
        """
        init_outdir()
        if tau1 is None:
            tau1 = 0.
        if tau2 is None:
            tau2 = tau1 + 30.
        if tau0_Ht is None:
            tau0_Ht = 0.
        if Ksim is None:
            Ksim = 100

        self.save_obj_GS = save_obj_GS
        self.mle_obj = mle_obj
        self.mle_obj_silverman = mle_obj_silverman
        self.tau1 = tau1
        self.tau2 = tau2
        self.tau0_Ht = tau0_Ht
        self.Ksim = Ksim
        if sample_idx_vec is None:
            Ksamples = len(save_obj_GS['lambda_bar'])
            sample_idx_vec = np.arange(0, Ksamples, 1)
        self.sample_idx_vec = sample_idx_vec
        self.epsilon_after_mainshock = epsilon_after_mainshock
        self.case_name = save_obj_GS['setup_obj'].case_name
        self.output_dir = output_dir

        # write to file
        fname_setup_obj = output_dir + "/setup_obj_pred_%s.all" % (self.case_name)
        file = open(fname_setup_obj, "wb")  # remember to open the file in binary mode
        pickle.dump(self, file)
        file.close()
        print('setup_obj has been created and saved:', fname_setup_obj)


class setup_sequential_pred():
    def __init__(self, save_obj_GS,
                 tau1_forecast, tau2_forecast,
                 dt_update=None,
                 tau0_Ht=None,
                 Ksim=None,
                 sample_idx_vec=None,
                 mle_obj=None, m0_plot=None,
                 m_star=None,
                 epsilon_after_mainshock=1e-6,
                 m_max=None):

        init_outdir()
        self.output_dir = output_dir
        self.m0_plot = m0_plot
        self.save_obj_GS = save_obj_GS
        if 'case_name' in save_obj_GS:
            self.case_name = str(save_obj_GS['case_name'])
        else:
            self.case_name = 'R0xx'
        self.mle_obj = mle_obj
        if Ksim is None:
            Ksim = 100
        self.Ksim = Ksim
        if sample_idx_vec is None:
            sample_idx_vec = np.arange(0, len(save_obj_GS['lambda_bar']), 1)  # all samples
        self.sample_idx_vec = sample_idx_vec
        dt = dt_update
        if dt is None:
            dt = int((tau2_forecast - tau1_forecast) / 10)  # in days
            if dt == 0:
                dt = (tau2_forecast - tau1_forecast) / 10.
        if tau1_forecast + dt > tau2_forecast:
            dt = tau2_forecast - tau1_forecast
            print('Warning: dt is too large, dt is set to:', dt, ' days.')
        self.dt_update = dt
        self.tau1_forecast = tau1_forecast
        self.tau2_forecast = tau2_forecast
        self.tau1_vec, self.tau2_vec = generate_tau1_tau2_vec_seq_forecast(tau1_forecast, tau2_forecast, save_obj_GS,
                                                                           dt=dt,
                                                                           m_star=m_star, eps=None)

        # history
        if tau0_Ht is None:
            tau0_Ht = tau1_forecast - 100
        self.tau0_Ht = tau0_Ht

        print('info: dt_update = ', self.dt_update)
        print('info: tau2_vec  = ', self.tau0_Ht)

        # update forecast window after a big shock with m_star
        self.m_star = m_star

        # maximum magnitude used in simulations if None m_max=np.inf
        self.m_max = m_max

        # write to file
        fname_setup_obj = output_dir + "/setup_obj_sequential_pred_%s.all" % (self.case_name)
        file = open(fname_setup_obj, "wb")  # remember to open the file in binary mode
        pickle.dump(self, file)
        file.close()
        print('setup_obj has been created and saved:', fname_setup_obj)


def init_outdir():
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    if not os.path.isdir(output_dir_tables):
        os.mkdir(output_dir_tables)
    if not os.path.isdir(output_dir_figures):
        os.mkdir(output_dir_figures)
    if not os.path.isdir(output_dir_data):
        os.mkdir(output_dir_data)


# stability issues
def PHI_t_omori(c, p, t_start=0., t_end=np.inf):
    PHI_t = np.nan
    if (p > 1. and t_end == np.inf):
        PHI_t = 1. / (p - 1) * (c + t_start) ** (1. - p)
    if (p > 1. and t_end != np.inf):
        PHI_t = 1. / (-p + 1) * ((c + t_end) ** (1. - p) - (c + t_start) ** (1. - p))
    if (p == 1 and t_end != np.inf):
        PHI_t = np.log(c + t_end) - np.log(c + t_start)
    return (PHI_t)


def n(m_alpha, m_beta, K, c, p, t_start=0., t_end=np.inf):
    n_t = PHI_t_omori(c, p, t_start, t_end)
    n = n_t * K / (1. - m_alpha / m_beta)
    return (n)


def NHPPsim(tau1, tau2, lambda_fun, fargs, max_lambda, dim=1):
    '''a function which simulates a 1D NHPP in [tau1,tau2)'''
    Nc = np.random.poisson(lam=(tau2 - tau1) * max_lambda)
    X_unthinned = np.random.uniform(tau1, tau2, size=Nc)
    idx = (np.random.uniform(0., max_lambda, size=Nc) <= lambda_fun(X_unthinned, *fargs))
    return np.sort(X_unthinned[idx])


def init_save_dictionary(obj):
    obj.save_pred = {'pred_offspring_Ht': [],
                     'pred_bgnew_and_offspring': [],
                     'pred_bgnew_and_offspring_with_Ht_offspring': [],
                     'M_max_all': [],
                     'k_sample': [],
                     'theta_Kcpadgq_k': [],
                     'N495_offspring_Ht': [],
                     'N495_all_with_Ht': [],
                     'N495_all_without_Ht': [],
                     'N495_bgnew': [],
                     'H_N495_all_with_Ht': [],
                     'Nm0_offspring_Ht': [],
                     'Nm0_all_with_Ht': [],
                     'Nm0_all_without_Ht': [],
                     'Nm0_bgnew': [],
                     'H_N_all_with_Ht': [],
                     'n_tau1_tau2': [],
                     'n_inf': [],
                     'N495_true': [],
                     'seed_orig': [],
                     'seed': [],
                     'tau_vec': []}
    obj.save_pred['tau_vec'].append(obj.tau_vec)


def sim_add_offspring(obj, X0_events):
    """
    simulates offspring: with branching
    :param X0_events:
    :return:
    """
    tau0, tau1, tau2 = obj.tau_vec
    T = tau2 - tau1
    m_beta = obj.m_beta
    m0 = obj.m0
    m_max = obj.m_max
    K, c, p, m_alpha, D, gamma, q = np.zeros(7) * np.nan

    # params and SPATIAL OFFSPRING
    if obj.spatial_offspring == 'G':
        K, c, p, m_alpha, D = obj.theta_Kcpadgq[:5]
        # print(K, c, p, m_alpha, D, gamma, q, m_beta, m0,obj.spatial_offspring)
    if obj.spatial_offspring == 'P':
        K, c, p, m_alpha, D, gamma, q = obj.theta_Kcpadgq
        # print(K, c, p, m_alpha, D, gamma, q, m_beta, m0,obj.spatial_offspring)
    if obj.spatial_offspring == 'R':
        K, c, p, m_alpha, D, gamma, q = obj.theta_Kcpadgq
        # print(K, c, p, m_alpha, D, gamma, q, m_beta, m0, obj.spatial_offspring)

    x0 = X0_events[:, 0:4]
    idx = 0
    z_parent = np.zeros((np.size(x0[:, 0]), 1))
    z_cluster = np.array(np.arange(np.size(x0[:, 0])) + 1)  # numbered clusters from 1,...,len(x)
    z_generation = np.zeros((np.size(x0[:, 0]), 1))  # zeros
    # lam_MOL = lambda t, m_i, K, c, p, m_alpha, m0: K * np.exp(m_alpha * (m_i - m0)) * 1. / (c + t) ** p
    MOL_fun = lambda t, t_i, m_i, K, c, p, m_alpha, m0: K * np.exp(m_alpha * (m_i - m0)) * 1. / (c + t - t_i) ** p

    x = x0
    while idx < np.size(x[:, 0]):
        # if np.mod(idx, 1000) == 0:  # info every 10th event
        #    print('current event=', idx + 1)
        t_i = x[idx, 0]
        m_i = x[idx, 1]
        x_i = x[idx, 2]
        y_i = x[idx, 3]
        fparams = np.array([t_i, m_i, K, c, p, m_alpha, m0])
        max_lambda = MOL_fun(t_i, t_i, m_i, K, c, p, m_alpha, m0)
        xnew = NHPPsim(t_i, tau2 - tau1, lambda_fun=MOL_fun, fargs=fparams, max_lambda=max_lambda, dim=1)
        coord_xy_new = None

        if xnew.size != 0:
            Nnew = np.size(xnew)
            id_cluster = z_cluster[idx]
            id_generation = int(z_generation[idx])
            if obj.Bayesian_m_beta is not None:
                m_samples = sc.stats.lomax.rvs(size=np.size(xnew), c=obj.alpha_posterior_mb, loc=m0,
                                               scale=obj.beta_posterior_mb)
                xnew = np.vstack((xnew, m_samples)).transpose()
            else:
                #xnew = np.vstack((xnew, np.random.exponential(1. / m_beta, np.size(xnew)) + m0)).transpose()
                m_sample = sample_from_truncated_exponential_rv(m_beta, m0, b=m_max,sample_size=np.size(xnew))
                xnew = np.vstack((xnew, m_sample)).transpose()

            # s(x-x_i)
            # (1) GAUSS (short range)
            if obj.spatial_offspring == 'G':
                coord_xy_new = sample_s_xy_short_range(mean=np.array([x_i, y_i]),
                                                       cov=np.eye(2) * D ** 2. * np.exp(m_alpha * (m_i - m0)),
                                                       N=Nnew)
                # (2a) Power Law decay (long  range)
            if obj.spatial_offspring == 'P':
                coord_xy_new = sample_long_range_decay_pwl(N=Nnew, x_center=x_i, y_center=y_i, m_center=m_i,
                                                           q=q, D=D, gamma_m=gamma, m0=m0)

                # (2b) Rupture Length Power Law decay (long  range)
            if obj.spatial_offspring == 'R':
                coord_xy_new = sample_long_range_decay_RL_pwl(N=Nnew, x_center=x_i, y_center=y_i, m_center=m_i,
                                                              q=q, D=D, gamma_m=gamma)

                # check if xnew is inside domain S
            xx = np.copy(coord_xy_new)
            xmin, xmax = obj.X_borders[0, :]
            ymin, ymax = obj.X_borders[1, :]
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
    z_parent_sorted_t = np.empty([x[:, 0].size, 1])
    z_cluster_sorted_t = np.empty([x[:, 0].size, 1])
    z_generation_sorted_t = np.empty([x[:, 0].size, 1])

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

    print('in add_offspring K:', K)
    return (data_tmxyz, complete_branching_structure_tmxyzcg)


def sim_offspring_from_Ht(obj, theta_Kcpadgq, spatial_offspring, Ht=None, all_pts_yes=None, print_info=None):
    """

    :param theta_Kcpadgq:
    :type theta_Kcpadgq:
    :param spatial_offspring:
    :type spatial_offspring:
    :param Ht:
    :type Ht:
    :return:
    :rtype:
    """

    tau0, tau1, tau2 = np.copy(obj.tau_vec)
    if Ht is None:
        idx = np.where((obj.data_obj.data_all.times >= tau0) & (obj.data_obj.data_all.times <= tau1))
        Ht = np.empty([len(obj.data_obj.data_all.times[idx]), 4])
        Ht[:, 0] = np.copy(obj.data_obj.data_all.times[idx])
        Ht[:, 1] = np.copy(obj.data_obj.data_all.magnitudes[idx])
        Ht[:, 2] = np.copy(obj.data_obj.data_all.positions[idx, 0])
        Ht[:, 3] = np.copy(obj.data_obj.data_all.positions[idx, 1])
    X0_events = np.copy(Ht)
    obj.Ht = np.copy(Ht)
    m_beta = np.copy(obj.m_beta)
    m0 = np.copy(obj.m0)
    m_max = np.copy(obj.m_max)
    K, c, p, m_alpha, D, gamma, q = np.zeros(7) * np.nan
    obj.theta_true_Kcpadgq = np.copy(theta_Kcpadgq)
    obj.spatial_offspring = spatial_offspring

    # params and SPATIAL OFFSPRING
    if obj.spatial_offspring == 'G':
        K, c, p, m_alpha, D = obj.theta_true_Kcpadgq[:5]
        # print(K, c, p, m_alpha, D, gamma, q, m_beta, m0, obj.spatial_offspring)
    if obj.spatial_offspring == 'P':
        K, c, p, m_alpha, D, gamma, q = obj.theta_true_Kcpadgq
        # print(K, c, p, m_alpha, D, gamma, q, m_beta, m0, obj.spatial_offspring)
    if obj.spatial_offspring == 'R':
        K, c, p, m_alpha, D, gamma, q = obj.theta_true_Kcpadgq
        # print(K, c, p, m_alpha, D, gamma, q, m_beta, m0, obj.spatial_offspring)

    # stability info, branching ratio
    obj.n_stability = n(m_alpha, m_beta, K, c, p, t_start=tau1, t_end=tau2)
    obj.n_stability_inf = n(m_alpha, m_beta, K, c, p, t_start=0., t_end=np.inf)

    x0 = X0_events[:, 0:4]
    if print_info is not None:
        print('tau_vec              =', obj.tau_vec)
        print('n(tau0,tau2)         =', obj.n_stability)
        print('n(tau0,+inf)         =', obj.n_stability_inf)
        print('shape x0', x0.shape)
    idx = 0
    z_parent = np.zeros((np.size(x0[:, 0]), 1))
    z_cluster = np.array(np.arange(np.size(x0[:, 0])) + 1)  # numbered clusters from 1,...,len(x)
    # z_cluster = np.flipud(-np.array(np.arange(np.size(x0[:, 0])) + 1))  # numbered clusters from -N_Ht, ... -2, -1
    z_generation = np.zeros((np.size(x0[:, 0]), 1))  # zeros
    # lam_MOL = lambda t, m_i, K, c, p, m_alpha, m0: K * np.exp(m_alpha * (m_i - m0)) * 1. / (c + t) ** p
    MOL_fun = lambda t, t_i, m_i, K, c, p, m_alpha, m0: K * np.exp(m_alpha * (m_i - m0)) * 1. / (c + t - t_i) ** p

    x = x0
    xnew = 0
    while idx < np.size(x[:, 0]):
        # if np.mod(idx, 1000) == 0:  # info every 10th event
        #    print('current event=', idx + 1)
        t_i = x[idx, 0]
        m_i = x[idx, 1]
        x_i = x[idx, 2]
        y_i = x[idx, 3]
        fparams = np.array([t_i, m_i, K, c, p, m_alpha, m0])
        max_lambda = None
        if t_i < tau1:
            max_lambda = MOL_fun(tau1, t_i, m_i, K, c, p, m_alpha, m0)
        if t_i >= tau1:
            max_lambda = MOL_fun(t_i, t_i, m_i, K, c, p, m_alpha, m0)
        # get new children
        # xnew = obj.simNHHP(0., T - t_i, lam_MOL, fparams, max_lambda) + t_i
        if t_i < tau1:
            xnew = NHPPsim(tau1, tau2, lambda_fun=MOL_fun, fargs=fparams, max_lambda=max_lambda, dim=1)
        if t_i >= tau1:
            xnew = NHPPsim(t_i, tau2, lambda_fun=MOL_fun, fargs=fparams, max_lambda=max_lambda, dim=1)
        coord_xy_new = None

        if xnew.size != 0:
            Nnew = np.size(xnew)
            id_cluster = z_cluster[idx]
            id_generation = int(z_generation[idx])
            if obj.Bayesian_m_beta is not None:
                m_samples = sc.stats.lomax.rvs(size=np.size(xnew), c=obj.alpha_posterior_mb, loc=m0,
                                               scale=obj.beta_posterior_mb)
                xnew = np.vstack((xnew, m_samples)).transpose()
            else:
                #xnew = np.vstack((xnew, np.random.exponential(1. / m_beta, np.size(xnew)) + m0)).transpose()
                m_sample = sample_from_truncated_exponential_rv(m_beta, m0, b=m_max,sample_size=np.size(xnew))
                xnew = np.vstack((xnew, m_sample)).transpose()

            # s(x-x_i)
            # (1) GAUSS (short range)
            # if len(obj.theta) == 7:
            if obj.spatial_offspring == 'G':
                coord_xy_new = sample_s_xy_short_range(mean=np.array([x_i, y_i]), \
                                                       cov=np.eye(2) * D ** 2. * np.exp(m_alpha * (m_i - m0)), \
                                                       N=Nnew)
                # (2a) Power Law decay (long  range)
            # if len(obj.theta) == 9 and obj.spatial_kernel[0] == 'P':
            if obj.spatial_offspring == 'P':
                coord_xy_new = sample_long_range_decay_pwl(N=Nnew, x_center=x_i, y_center=y_i, m_center=m_i,
                                                           q=q, D=D, gamma_m=gamma, m0=m0)

                # (2b) Rupture Length Power Law decay (long  range)
            # if len(obj.theta) == 9 and obj.spatial_kernel[0] == 'R':
            if obj.spatial_offspring == 'R':
                coord_xy_new = sample_long_range_decay_RL_pwl(N=Nnew, x_center=x_i, y_center=y_i, m_center=m_i,
                                                              q=q, D=D, gamma_m=gamma)

                # check if xnew is inside domain S
            xx = np.copy(coord_xy_new)
            xmin, xmax = obj.data_obj.domain.X_borders[0, :]
            ymin, ymax = obj.data_obj.domain.X_borders[1, :]
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
    z_parent_sorted_t = np.empty([x[:, 0].size, 1])
    z_cluster_sorted_t = np.empty([x[:, 0].size, 1])
    z_generation_sorted_t = np.empty([x[:, 0].size, 1])

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
    shift_Ht = len(Ht)
    data_tmxyz = np.zeros((np.size(x_sorted_t[:, 0]), 5)) * np.nan
    data_tmxyz[:, 0:4] = x_sorted_t
    data_tmxyz[:, 4] = np.squeeze(parents_sorted_t - shift_Ht).astype(int)
    complete_branching_structure_tmxyzcg = np.zeros((np.size(x_sorted_t[:, 0]), 8)) * np.nan
    complete_branching_structure_tmxyzcg[:, 0:5] = data_tmxyz[:, 0:5]
    complete_branching_structure_tmxyzcg[:, 5] = z_cluster_sorted_t - shift_Ht
    complete_branching_structure_tmxyzcg[:, 6] = z_generation_sorted_t.reshape([-1, ])
    complete_branching_structure_tmxyzcg[:, 7] = -np.ones(
        np.size(x_sorted_t[:, 0]))  # np.array(np.arange(np.size(x_sorted_t[:, 0])) + 1)
    obj.Ht_offspring = np.copy(
        complete_branching_structure_tmxyzcg[complete_branching_structure_tmxyzcg[:, 0] > tau1, :])
    if all_pts_yes is not None:
        obj.Ht_offspring = np.copy(complete_branching_structure_tmxyzcg)

    print('in sim offspring from Ht K:', K)
    return (data_tmxyz[complete_branching_structure_tmxyzcg[:, 0] > tau1, :],
            complete_branching_structure_tmxyzcg[complete_branching_structure_tmxyzcg[:, 0] > tau1, :])


class predictions_mle():
    def __init__(self, mle_obj, tau1, tau2, tau0_Ht=0., Ksim=None, data_obj=None, seed=None,
                 Bayesian_m_beta=None, m_max=None):
        """

        :param mle_obj:
        :type mle_obj:
        :param Ksim:
        :type Ksim:
        :param tau1:
        :type tau1:
        :param tau2:
        :type tau2:
        :param tau0_Ht:
        :type tau0_Ht:
        :param seed:
        :type seed:
        :param Bayesian_m_beta:
        :type Bayesian_m_beta:
        """
        if Ksim is None:
            Ksim = 1
        dim = 2
        self.tau0_Ht = tau0_Ht
        self.tau1 = tau1
        self.tau2 = tau2
        self.tau_vec = np.array([tau0_Ht, tau1, tau2])
        self.data_obj = mle_obj.data_obj
        self.mle_obj = mle_obj
        self.save_pred = None
        self.N495_true = np.sum(self.data_obj.data_all.magnitudes[
                                    np.logical_and(self.data_obj.data_all.times >= tau1,
                                                   self.data_obj.data_all.times <= tau2)] >= 4.95)
        self.Nm0_true = np.sum(self.data_obj.data_all.magnitudes[
                                   np.logical_and(self.data_obj.data_all.times >= tau1,
                                                  self.data_obj.data_all.times <= tau2)] >= self.data_obj.domain.m0)

        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            self.seed = seed
        self.tic = time.perf_counter()

        # domain
        T_abs = tau2 - tau1
        X_abs = np.prod(np.diff(self.data_obj.domain.X_borders))
        self.X_borders = np.copy(self.data_obj.domain.X_borders)
        dim = self.data_obj.domain.X_borders.shape[0]
        self.spatial_offspring = mle_obj.setup_obj.spatial_offspring

        # offspring params
        self.theta_Kcpadgq = np.copy(mle_obj.theta_mle_Kcpadgq)

        # fixed params
        self.m0 = self.data_obj.domain.m0
        self.m_max = m_max
        self.m_beta = mle_obj.m_beta_lower_T2
        if data_obj is None:
            data_obj = mle_obj.data_obj

        init_save_dictionary(self)
        self.save_pred['N495_true'] = np.copy(self.N495_true)
        self.save_pred['Nm0_true'] = np.copy(self.Nm0_true)
        self.save_pred['seed_orig'] = np.copy(seed)
        self.save_pred['seed'] = np.copy(self.seed)
        self.save_pred['data_obj'] = data_obj
        self.save_pred['mle_obj'] = self.mle_obj
        self.save_pred['m0'] = np.copy(self.data_obj.domain.m0)

        # some params
        mu_max_bg_mle = np.max(mle_obj.mu_grid)
        abs_T = tau2 - tau1
        abs_X = np.prod(np.diff(data_obj.domain.X_borders))
        self.tic = time.perf_counter()
        self.Bayesian_m_beta = Bayesian_m_beta

        for k in range(Ksim):

            # generate bg via thinning
            Nc = np.random.poisson(mu_max_bg_mle * abs_T * abs_X)
            X_unthinned = np.random.rand(Nc, dim) * np.diff(data_obj.domain.X_borders).T + \
                          data_obj.domain.X_borders[:, 0][np.newaxis]
            fast = 'yes'
            if fast is not None:
                X_borders = np.copy(self.data_obj.domain.X_borders)
                mu_unthinned = get_grid_data_for_a_point(mle_obj.mu_grid,
                                                         points_xy=X_unthinned,
                                                         X_borders=X_borders,
                                                         method='nearest')
            else:
                mu_unthinned = mle_obj.eval_kde_xprime(X_unthinned)

            if Nc > 0:
                if np.max(mu_unthinned) > mu_max_bg_mle:
                    raise ValueError("Upper bound mu_max is too low.")
            # thinning
            thinned_idx = np.where(mu_max_bg_mle * np.random.rand(len(X_unthinned)) <= mu_unthinned)[0]
            X_thinned = X_unthinned[thinned_idx]

            # writing to a matrix
            bg_events = np.zeros([int(len(X_thinned[:, 0])), 5])
            bg_events[:, 0] = np.random.rand(len(X_thinned[:, 0])) * (tau2 - tau1)
            #bg_events[:, 1] = np.random.exponential(1. / self.m_beta,len(X_thinned[:, 0])) + self.m0  # m_i: marks
            bg_events[:, 1] = sample_from_truncated_exponential_rv(self.m_beta, self.m0, b=self.m_max,sample_size=len(X_thinned[:, 0]))
            bg_events[:, 2] = X_thinned[:, 0]  # x_coord
            bg_events[:, 3] = X_thinned[:, 1]  # y_coord
            bg_events[:, 4] = np.zeros(len(X_thinned[:, 0]))  # branching z=0
            sort_idx = np.argsort(bg_events[:, 0])
            bg_events = bg_events[sort_idx, :]

            # offspring (aftershocks) using: mle theta_phi
            pred, pred_aug = sim_add_offspring(self, bg_events)
            bgnew_and_offspring = np.copy(pred_aug)
            bgnew_and_offspring[:, 0] += tau1

            # offspring from Ht
            Ht_pred, Ht_pred_aug = sim_offspring_from_Ht(self, theta_Kcpadgq=self.theta_Kcpadgq,
                                                         spatial_offspring=self.spatial_offspring)

            # adding together: offspring from Ht + new background + offspring of background
            added = np.vstack([Ht_pred_aug, bgnew_and_offspring])
            events_all_with_Ht_offspring = added[np.argsort(added[:, 0])]

            # info
            self.bgnew = np.copy(bg_events)
            self.bgnew[:, 0] += tau1
            self.bgnew_and_offspring = np.copy(bgnew_and_offspring)
            self.Ht_offspring = np.copy(Ht_pred_aug)
            self.events_all_with_Ht_offspring = np.copy(events_all_with_Ht_offspring)
            self.X_unthinned = np.copy(X_unthinned) + self.data_obj.domain.X_borders[:, 0][np.newaxis]
            self.mu_unthinned = np.copy(mu_unthinned)
            self.thinned_idx = thinned_idx
            self.X_thinned = X_thinned
            self.mu_thinned = mu_unthinned[thinned_idx]
            sim_k = self.events_all_with_Ht_offspring
            H_N, xed, yed = np.histogram2d(sim_k[:, 2], sim_k[:, 3], range=self.data_obj.domain.X_borders, bins=20)
            self.H_N = H_N
            idx_N495 = sim_k[:, 1] >= 4.95
            H_N495, xed, yed = np.histogram2d(sim_k[idx_N495, 2], sim_k[idx_N495, 3],
                                              range=self.data_obj.domain.X_borders,
                                              bins=20)
            self.H_N495 = H_N495

            # save results
            self.save_pred['pred_offspring_Ht'].append(self.Ht_offspring)
            self.save_pred['pred_bgnew_and_offspring'].append(self.bgnew_and_offspring)
            self.save_pred['pred_bgnew_and_offspring_with_Ht_offspring'].append(
                self.events_all_with_Ht_offspring)
            if len(self.events_all_with_Ht_offspring[:, 1]) > 0:
                self.save_pred['M_max_all'].append(np.max(self.events_all_with_Ht_offspring[:, 1]))
            self.save_pred['k_sample'].append(k)
            self.save_pred['theta_Kcpadgq_k'].append(self.theta_true_Kcpadgq)
            self.save_pred['N495_offspring_Ht'].append(np.sum(self.Ht_offspring[:, 1] >= 4.95))
            self.save_pred['N495_all_with_Ht'].append(np.sum(self.events_all_with_Ht_offspring[:, 1] >= 4.95))
            self.save_pred['N495_all_without_Ht'].append(np.sum(self.bgnew_and_offspring[:, 1] >= 4.95))
            self.save_pred['N495_bgnew'].append(np.sum(self.bgnew[:, 1] >= 4.95))
            self.save_pred['H_N495_all_with_Ht'].append(self.H_N495)
            self.save_pred['Nm0_offspring_Ht'].append(np.sum(self.Ht_offspring[:, 1] >= self.m0))
            self.save_pred['Nm0_all_with_Ht'].append(np.sum(self.events_all_with_Ht_offspring[:, 1] >= self.m0))
            self.save_pred['Nm0_all_without_Ht'].append(np.sum(self.bgnew_and_offspring[:, 1] >= self.m0))
            self.save_pred['Nm0_bgnew'].append(np.sum(self.bgnew[:, 1] >= self.m0))
            self.save_pred['H_N_all_with_Ht'].append(self.H_N)
            K, c, p, m_alpha = self.theta_true_Kcpadgq[:4]
            m_beta = np.copy(self.m_beta)
            self.n_stability = n(m_alpha, m_beta, K, c, p, t_start=tau1, t_end=tau2)
            self.n_stability_inf = n(m_alpha, m_beta, K, c, p, t_start=0., t_end=np.inf)
            self.save_pred['n_tau1_tau2'].append(self.n_stability)
            self.save_pred['n_inf'].append(self.n_stability_inf)

            if np.mod(k, 1) == 0:  # info every 10th event
                tictoc = time.perf_counter() - self.tic
                print('current simulation is k =', k + 1, 'of K =', Ksim,
                      '%.2f sec. elapsed time.' % tictoc,
                      ' approx. done %.1f percent.' % (100. * (k / float(Ksim))))

            # some postprocessing
            self.save_pred['cumsum'] = cumsum_events_pred(save_obj_pred=self.save_pred,
                                                          tau1=self.tau1, tau2=self.tau2, m0=self.m0,
                                                          which_events='all')
        return


class predictions_gpetas():
    def __init__(self, save_obj_GS, tau1, tau2, tau0_Ht=0., sample_idx_vec=None, seed=None, approx=None, Ksim=None,
                 randomized_samples='yes', Bayesian_m_beta=None, m_max=None):
        '''
        Simulates data from the predictive distribution (using posterior samples)
        :param Bayesian_m_beta:
        :type Bayesian_m_beta:
        :param randomized_samples: if None no random set of the samples is taken
        :type randomized_samples: arbitrary
        :param Ksim:
        :type Ksim:
        :param save_obj_GS:
        :type save_obj_GS:
        :param tau1:
        :type tau1:
        :param tau2:
        :type tau2:
        :param tau0_Ht:
        :type tau0_Ht:
        :param sample_idx_vec:
        :type sample_idx_vec:
        :param seed:
        :type seed:
        :param approx:
        :type approx:
        '''
        self.tau0_Ht = tau0_Ht
        self.tau1 = tau1
        self.tau2 = tau2
        self.tau_vec = np.array([tau0_Ht, tau1, tau2])
        self.data_obj = save_obj_GS['data_obj']
        self.save_obj_GS = save_obj_GS  # new ... maybe delete all individual sub attributes
        self.Ksamples = len(save_obj_GS['lambda_bar'])
        self.save_pred = None
        self.N495_true = np.sum(self.data_obj.data_all.magnitudes[
                                    np.logical_and(self.data_obj.data_all.times >= tau1,
                                                   self.data_obj.data_all.times <= tau2)] >= 4.95)
        self.Nm0_true = np.sum(self.data_obj.data_all.magnitudes[
                                   np.logical_and(self.data_obj.data_all.times >= tau1,
                                                  self.data_obj.data_all.times <= tau2)] >= self.data_obj.domain.m0)

        # simple m_beta Bayesian
        self.Bayesian_m_beta = Bayesian_m_beta
        if Bayesian_m_beta is not None:
            mu_prior__m_beta = np.log(10)
            c_prior__m_beta = 0.1
            alpha_prior_mb = 1. / c_prior__m_beta ** 2.
            beta_prior_mb = 1. / (c_prior__m_beta ** 2 * mu_prior__m_beta)
            alpha_posterior_mb = alpha_prior_mb + save_obj_GS['setup_obj'].N_training
            sum_mi_minus_m0 = save_obj_GS['setup_obj'].N_training / save_obj_GS['setup_obj'].m_beta
            beta_posterior_mb = beta_prior_mb + sum_mi_minus_m0
            self.alpha_posterior_mb = np.copy(alpha_posterior_mb)
            self.beta_posterior_mb = np.copy(beta_posterior_mb)
            print('Uses default prior on m_beta with mu=', mu_prior__m_beta, ' c=', c_prior__m_beta)
            print('m beta posterior: alpha,beta posterior=', alpha_posterior_mb, beta_posterior_mb)

        if sample_idx_vec is None:
            sample_idx_vec = [0]
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            self.seed = seed
        self.noise = 1e-4
        if approx is None:
            self.approx = 'sampling_from_cond_sparse_GP'
        else:
            self.approx = 'using_sparse_GP_mean'

        # domain
        T_abs = tau2 - tau1
        X_abs = np.prod(np.diff(self.data_obj.domain.X_borders))
        self.X_borders = np.copy(self.data_obj.domain.X_borders)
        dim = self.data_obj.domain.X_borders.shape[0]
        self.spatial_offspring = save_obj_GS['setup_obj'].spatial_offspring

        # offspring params
        self.theta_Kcpadgq = None

        # fixed params
        self.m0 = self.data_obj.domain.m0
        self.m_beta = save_obj_GS['setup_obj'].m_beta

        self.init_save_dictionary()
        self.save_pred['N495_true'] = np.copy(self.N495_true)
        self.save_pred['Nm0_true'] = np.copy(self.Nm0_true)
        self.save_pred['seed_orig'] = np.copy(seed)
        self.save_pred['seed'] = np.copy(self.seed)
        self.save_pred['data_obj'] = save_obj_GS['data_obj']
        self.save_pred['m0'] = np.copy(self.data_obj.domain.m0)
        self.save_pred['data_obj'] = self.data_obj
        self.save_pred['save_obj_GS'] = self.save_obj_GS
        self.tic = time.perf_counter()

        if Ksim is not None:
            print('1', len(sample_idx_vec))
            # if len(sample_idx_vec)==1:
            #    sample_idx_vec = np.arange(0,self.Ksamples,1)
            if randomized_samples is not None:
                randomized_samples = np.random.choice(sample_idx_vec, Ksim)
                sample_idx_vec = np.copy(randomized_samples)
            if randomized_samples is None:
                if len(sample_idx_vec) < Ksim:
                    sample_idx_vec = np.repeat(sample_idx_vec, int(Ksim / len(sample_idx_vec)))
                if len(sample_idx_vec) > Ksim:
                    sample_idx_vec = np.ceil(np.linspace(0, len(sample_idx_vec) - 1, Ksim)).astype(int)
                if len(sample_idx_vec) < Ksim:
                    sample_idx_vec = np.append(sample_idx_vec, sample_idx_vec[:(Ksim - len(sample_idx_vec))])
                print('Fixed samples:randomized_samples is None.')
                print('len(sample_idx_vec)=', len(sample_idx_vec), 'Ksim=', Ksim)
            print(Ksim, len(sample_idx_vec))
            print(sample_idx_vec)
        self.save_pred['sample_idx_vec'] = sample_idx_vec

        for i in range(len(sample_idx_vec)):
            if np.mod(i, 1) == 0:  # info every 10th event
                Ksim = len(sample_idx_vec)
                tictoc = time.perf_counter() - self.tic
                print('current simulation is k =', i + 1, 'of K =', Ksim,
                      '%.2f sec. elapsed time.' % tictoc,
                      ' approx. done %.1f percent.' % (100. * (i / float(Ksim))))
            k = sample_idx_vec[i]
            # background using: lambda_bar, f_grid, hyper_nus, X_grid
            lmbda_bar = save_obj_GS['lambda_bar'][k].item()
            mu_grid = save_obj_GS['mu_grid'][k]
            self.f = -np.log(lmbda_bar / mu_grid - 1.)
            self.cov_params = [save_obj_GS['cov_params_theta'][k],
                               np.array([save_obj_GS['cov_params_nu1'][k], save_obj_GS['cov_params_nu2'][k]])]
            self.X = save_obj_GS['X_grid_NN']
            Nc = np.random.poisson(lmbda_bar * X_abs * T_abs)
            X_unthinned = np.random.rand(Nc, dim) * np.diff(self.data_obj.domain.X_borders).T
            if approx is None:
                # f_unthinned = self.sample_from_cond_GP(xprime=X_unthinned)
                # mu_unthinned = lmbda_bar * 1. / (1 + np.exp(-f_unthinned))

                ## new attemps of accelerating predictions: now on a grid
                X_borders = np.copy(self.data_obj.domain.X_borders)
                X_borders_NN = X_borders - np.array(
                    [[X_borders[0, 0], X_borders[0, 0]], [X_borders[1, 0], X_borders[1, 0]]])
                mu_unthinned = get_grid_data_for_a_point(mu_grid,
                                                         points_xy=X_unthinned,
                                                         X_borders=X_borders_NN,
                                                         method='nearest')
            else:
                f_unthinned = self.mean_from_cond_GP(xprime=X_unthinned)
                mu_unthinned = lmbda_bar * 1. / (1 + np.exp(-f_unthinned))
            thinned_idx = np.where(lmbda_bar * np.random.rand(len(X_unthinned)) <= mu_unthinned)[0]
            X_thinned = X_unthinned[thinned_idx] + self.data_obj.domain.X_borders[:, 0][np.newaxis]
            bg_events = np.zeros([int(len(X_thinned[:, 0])), 5])
            bg_events[:, 0] = np.random.rand(len(X_thinned[:, 0])) * (tau2 - tau1)
            if Bayesian_m_beta is not None:
                bg_events[:, 1] = sc.stats.lomax.rvs(size=len(X_thinned[:, 0]), c=alpha_posterior_mb, loc=self.m0,
                                                     scale=beta_posterior_mb)
            else:
                #bg_events[:, 1] = np.random.exponential(1. / self.m_beta, len(X_thinned[:, 0])) + self.m0  # m_i: marks
                bg_events[:, 1] = sample_from_truncated_exponential_rv(self.m_beta, self.m0, b=self.m_max, sample_size=len(X_thinned[:, 0]))
            bg_events[:, 2] = X_thinned[:, 0]  # x_coord
            bg_events[:, 3] = X_thinned[:, 1]  # y_coord
            bg_events[:, 4] = np.zeros(len(X_thinned[:, 0]))  # branching z=0
            sort_idx = np.argsort(bg_events[:, 0])
            bg_events = bg_events[sort_idx, :]
            self.bg_events = np.copy(bg_events)

            # offspring (aftershocks) using: kth theta_phi
            theta = None
            self.theta_Kcpadgq = None
            theta = np.copy(save_obj_GS['theta_tilde'][k])  # *np.random.uniform(0.95,1.1)
            self.theta_Kcpadgq = np.copy(theta)  # gives it to the subroutines

            pred, pred_aug = sim_add_offspring(self, bg_events)
            bgnew_and_offspring = np.copy(pred_aug)
            bgnew_and_offspring[:, 0] += tau1

            # offspring from Ht
            Ht_pred, Ht_pred_aug = sim_offspring_from_Ht(self, theta_Kcpadgq=self.theta_Kcpadgq,
                                                         spatial_offspring=self.spatial_offspring)
            # adding together: offspring from Ht + new background + offspring of background
            added = np.vstack([Ht_pred_aug, bgnew_and_offspring])
            events_all_with_Ht_offspring = added[np.argsort(added[:, 0])]

            # info
            self.bgnew = np.copy(bg_events)
            self.bgnew[:, 0] += tau1
            self.bgnew_and_offspring = np.copy(bgnew_and_offspring)
            self.Ht_offspring = np.copy(Ht_pred_aug)
            self.events_all_with_Ht_offspring = np.copy(events_all_with_Ht_offspring)
            self.mu_grid = mu_grid
            self.X_unthinned = np.copy(X_unthinned) + self.data_obj.domain.X_borders[:, 0][np.newaxis]
            # self.f_unthinned = np.copy(f_unthinned)
            self.mu_unthinned = np.copy(mu_unthinned)
            self.thinned_idx = thinned_idx
            self.X_thinned = X_thinned
            self.mu_thinned = mu_unthinned[thinned_idx]
            sim_k = self.events_all_with_Ht_offspring
            H_N, xed, yed = np.histogram2d(sim_k[:, 2], sim_k[:, 3], range=self.data_obj.domain.X_borders, bins=20)
            self.H_N = H_N
            idx_N495 = sim_k[:, 1] >= 4.95
            H_N495, xed, yed = np.histogram2d(sim_k[idx_N495, 2], sim_k[idx_N495, 3],
                                              range=self.data_obj.domain.X_borders,
                                              bins=20)
            self.H_N495 = H_N495

            # save results
            self.save_pred['pred_offspring_Ht'].append(self.Ht_offspring)
            self.save_pred['pred_bgnew_and_offspring'].append(self.bgnew_and_offspring)
            self.save_pred['pred_bgnew_and_offspring_with_Ht_offspring'].append(
                self.events_all_with_Ht_offspring)
            if len(self.events_all_with_Ht_offspring[:, 1]) > 0:
                self.save_pred['M_max_all'].append(np.max(self.events_all_with_Ht_offspring[:, 1]))
            self.save_pred['k_sample'].append(k)
            self.save_pred['theta_Kcpadgq_k'].append(self.theta_true_Kcpadgq)
            self.save_pred['N495_offspring_Ht'].append(np.sum(self.Ht_offspring[:, 1] >= 4.95))
            self.save_pred['N495_all_with_Ht'].append(np.sum(self.events_all_with_Ht_offspring[:, 1] >= 4.95))
            self.save_pred['N495_all_without_Ht'].append(np.sum(self.bgnew_and_offspring[:, 1] >= 4.95))
            self.save_pred['N495_bgnew'].append(np.sum(self.bgnew[:, 1] >= 4.95))
            self.save_pred['H_N495_all_with_Ht'].append(self.H_N495)
            self.save_pred['Nm0_offspring_Ht'].append(np.sum(self.Ht_offspring[:, 1] >= self.m0))
            self.save_pred['Nm0_all_with_Ht'].append(np.sum(self.events_all_with_Ht_offspring[:, 1] >= self.m0))
            self.save_pred['Nm0_all_without_Ht'].append(np.sum(self.bgnew_and_offspring[:, 1] >= self.m0))
            self.save_pred['Nm0_bgnew'].append(np.sum(self.bgnew[:, 1] >= self.m0))
            self.save_pred['H_N_all_with_Ht'].append(self.H_N)
            K, c, p, m_alpha = self.theta_true_Kcpadgq[:4]
            m_beta = np.copy(self.m_beta)
            self.n_stability = n(m_alpha, m_beta, K, c, p, t_start=tau1, t_end=tau2)
            self.n_stability_inf = n(m_alpha, m_beta, K, c, p, t_start=0., t_end=np.inf)
            self.save_pred['n_tau1_tau2'].append(self.n_stability)
            self.save_pred['n_inf'].append(self.n_stability_inf)
            # some postprocessing
            self.save_pred['cumsum'] = cumsum_events_pred(save_obj_pred=self.save_pred,
                                                          tau1=self.tau1, tau2=self.tau2, m0=self.m0,
                                                          which_events='all')
        return

    def NHPPsim(self, tau1, tau2, lambda_fun, fargs, max_lambda, dim=1):
        '''a function which simulates a 1D NHPP in [t1,t2)'''
        Nc = np.random.poisson(lam=(tau2 - tau1) * max_lambda)
        X_unthinned = np.random.uniform(tau1, tau2, size=Nc)
        idx = (np.random.uniform(0., max_lambda, size=Nc) <= lambda_fun(X_unthinned, *fargs))
        return np.sort(X_unthinned[idx])

    def init_save_dictionary(self):
        self.save_pred = {'pred_offspring_Ht': [],
                          'pred_bgnew_and_offspring': [],
                          'pred_bgnew_and_offspring_with_Ht_offspring': [],
                          'M_max_all': [],
                          'k_sample': [],
                          'theta_Kcpadgq_k': [],
                          'N495_offspring_Ht': [],
                          'N495_all_with_Ht': [],
                          'N495_all_without_Ht': [],
                          'N495_bgnew': [],
                          'H_N495_all_with_Ht': [],
                          'Nm0_offspring_Ht': [],
                          'Nm0_all_with_Ht': [],
                          'Nm0_all_without_Ht': [],
                          'Nm0_bgnew': [],
                          'H_N_all_with_Ht': [],
                          'n_tau1_tau2': [],
                          'n_inf': [],
                          'N495_true': [],
                          'seed_orig': [],
                          'seed': [],
                          'tau_vec': []}
        self.save_pred['tau_vec'].append(self.tau_vec)

    def sample_from_cond_GP(self, xprime):

        self.K = self.cov_func(self.X, self.X)
        self.L = np.linalg.cholesky(self.K + self.noise * np.eye(self.K.shape[0]))
        self.L_inv = solve_triangular(self.L, np.eye(self.L.shape[0]), lower=True, check_finite=False)
        self.K_inv = self.L_inv.T.dot(self.L_inv)
        k = self.cov_func(self.X, xprime)
        mean = k.T.dot(self.K_inv.dot(self.f))
        kprimeprime = self.cov_func(xprime, xprime)
        # var = (kprimeprime - k.T.dot(self.K_inv.dot(k))).diagonal()
        # gprime = mean + numpy.sqrt(var)*numpy.random.randn(xprime.shape[0])
        Sigma = (kprimeprime - k.T.dot(self.K_inv.dot(k)))
        L = np.linalg.cholesky(Sigma + self.noise * np.eye(Sigma.shape[0]))
        gprime = mean + np.dot(L.T, np.random.randn(xprime.shape[0]))
        return gprime

    def mean_from_cond_GP(self, xprime):

        self.K = self.cov_func(self.X, self.X)
        self.L = np.linalg.cholesky(self.K + self.noise * np.eye(self.K.shape[0]))
        self.L_inv = solve_triangular(self.L, np.eye(self.L.shape[0]), lower=True, check_finite=False)
        self.K_inv = self.L_inv.T.dot(self.L_inv)
        k = self.cov_func(self.X, xprime)
        mean = k.T.dot(self.K_inv.dot(self.f))
        gprime = np.copy(mean)
        return gprime

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


def sample_s_xy_short_range(mean, cov, N):
    xy_offspring = np.random.multivariate_normal(mean, cov, N)
    return xy_offspring


def sample_long_range_decay_pwl(N, x_center, y_center, m_center, q, D, gamma_m, m0):
    sigma_mi = D ** 2 * np.exp(gamma_m * (m_center - m0))
    # sample theta: angle
    theta_j = np.random.uniform(0, 2. * np.pi, N)
    # sample r: radius from center
    u_j = np.random.uniform(0, 1, N)
    r = np.sqrt(((1. - u_j) ** (1. / (1. - q)) - 1.)) * np.sqrt(sigma_mi)  # returns non-negative sqrt
    x_vec = x_center + r * np.cos(theta_j)
    y_vec = y_center + r * np.sin(theta_j)
    xy_array = np.vstack((x_vec, y_vec)).T
    return xy_array


def sample_long_range_decay_RL_pwl(N, x_center, y_center, m_center, q, D, gamma_m):
    sigma_mi = D ** 2 * 10 ** (2 * gamma_m * m_center)
    # sample theta: angle
    theta_j = np.random.uniform(0, 2. * np.pi, N)
    # sample r: radius from center
    u_j = np.random.uniform(0, 1, N)
    r = np.sqrt(((1. - u_j) ** (1. / (1. - q)) - 1.)) * np.sqrt(sigma_mi)  # returns non-negative sqrt
    x_vec = x_center + r * np.cos(theta_j)
    y_vec = y_center + r * np.sin(theta_j)
    xy_array = np.vstack((x_vec, y_vec)).T
    return xy_array


# post processing
def cumsum_events_pred(save_obj_pred, tau1, tau2, m0=None, which_events=None):
    """
    Extracts
    :param save_obj_pred:contains the forecast/prediction
    :type save_obj_pred:dict
    :param tau1:
    :type tau1: float
    :param tau2:
    :type tau2: float
    :param m0:
    :type m0:float
    :param which_events:
    :type which_events:string
    :return:
    :rtype:dictionary
    """
    data_obj = save_obj_pred['data_obj']
    out = {'y_obs': [],
           'x_obs': [],
           'y': [],
           'x': [],
           'y_bg': [],
           'x_bg': [],
           'y_off': [],
           'x_off': [],
           'y_bg_off': [],
           'x_bg_off': [],
           'y_Htoff': [],
           'x_Htoff': [],
           'm0': [],
           'Ksim': []}

    # observations
    N_tau1 = np.sum(data_obj.data_all.times <= tau1)
    N_in_tau1_tau2 = np.sum(np.logical_and(data_obj.data_all.times > tau1, data_obj.data_all.times <= tau2))
    idx_test = np.logical_and(data_obj.data_all.times > tau1, data_obj.data_all.times <= tau2)
    if m0 is not None:
        N_tau1 = np.sum(np.logical_and(data_obj.data_all.times <= tau1, data_obj.data_all.magnitudes >= m0))
        N_in_tau1_tau2 = np.sum(
            np.logical_and(np.logical_and(data_obj.data_all.times > tau1, data_obj.data_all.times <= tau2),
                           data_obj.data_all.magnitudes >= m0))
        idx_test = np.logical_and(np.logical_and(data_obj.data_all.times > tau1, data_obj.data_all.times <= tau2),
                                  data_obj.data_all.magnitudes >= m0)
    out['y_obs'] = np.hstack([N_tau1, np.cumsum(np.ones(N_in_tau1_tau2)) + N_tau1]) - N_tau1
    out['x_obs'] = np.hstack([tau1, data_obj.data_all.times[idx_test]]) - tau1

    # simulations
    Ksim = len(save_obj_pred['pred_bgnew_and_offspring_with_Ht_offspring'])
    out['Ksim'] = Ksim
    pred_data = []
    for i in range(Ksim):
        pred_data = save_obj_pred['pred_bgnew_and_offspring_with_Ht_offspring'][i]
        N_in_tau1_tau2 = np.sum(np.logical_and(pred_data[:, 0] > tau1, pred_data[:, 0] <= tau2))
        idx_test = np.logical_and(pred_data[:, 0] > tau1, pred_data[:, 0] <= tau2)
        if m0 is not None:
            N_in_tau1_tau2 = np.sum(np.logical_and(
                np.logical_and(pred_data[:, 0] > tau1, pred_data[:, 0] <= tau2), pred_data[:, 1] >= m0))
            idx_test = np.logical_and(np.logical_and(
                pred_data[:, 0] > tau1, pred_data[:, 0] <= tau2), pred_data[:, 1] >= m0)
        y = np.hstack([N_tau1, np.cumsum(np.ones(N_in_tau1_tau2)) + N_tau1]) - N_tau1
        x = np.hstack([tau1, pred_data[idx_test, 0]]) - tau1
        out['y'].append(y)
        out['x'].append(x)

    if which_events == 'bg':
        pred_data = []
        # bg only
        for i in range(Ksim):
            idx_bg = save_obj_pred['pred_bgnew_and_offspring'][i][:, 4] == 0
            pred_data = save_obj_pred['pred_bgnew_and_offspring'][i][idx_bg, :]
            N_in_tau1_tau2 = np.sum(np.logical_and(pred_data[:, 0] > tau1, pred_data[:, 0] <= tau2))
            idx_test = np.logical_and(pred_data[:, 0] > tau1, pred_data[:, 0] <= tau2)
            if m0 is not None:
                N_in_tau1_tau2 = np.sum(np.logical_and(
                    np.logical_and(pred_data[:, 0] > tau1, pred_data[:, 0] <= tau2), pred_data[:, 1] >= m0))
                idx_test = np.logical_and(np.logical_and(
                    pred_data[:, 0] > tau1, pred_data[:, 0] <= tau2), pred_data[:, 1] >= m0)
            y = np.hstack([N_tau1, np.cumsum(np.ones(N_in_tau1_tau2)) + N_tau1]) - N_tau1
            x = np.hstack([tau1, pred_data[idx_test, 0]]) - tau1
            out['y_bg'].append(y)
            out['x_bg'].append(x)

    if which_events == 'off':
        pred_data = []
        # offspring (from bg) only
        for i in range(Ksim):
            idx_off = save_obj_pred['pred_bgnew_and_offspring'][i][:, 4] > 0
            pred_data = save_obj_pred['pred_bgnew_and_offspring'][i][idx_off, :]
            N_in_tau1_tau2 = np.sum(np.logical_and(pred_data[:, 0] > tau1, pred_data[:, 0] <= tau2))
            idx_test = np.logical_and(pred_data[:, 0] > tau1, pred_data[:, 0] <= tau2)
            if m0 is not None:
                N_in_tau1_tau2 = np.sum(np.logical_and(
                    np.logical_and(pred_data[:, 0] > tau1, pred_data[:, 0] <= tau2), pred_data[:, 1] >= m0))
                idx_test = np.logical_and(np.logical_and(
                    pred_data[:, 0] > tau1, pred_data[:, 0] <= tau2), pred_data[:, 1] >= m0)
            y = np.hstack([N_tau1, np.cumsum(np.ones(N_in_tau1_tau2)) + N_tau1]) - N_tau1
            x = np.hstack([tau1, pred_data[idx_test, 0]]) - tau1
            out['y_off'].append(y)
            out['x_off'].append(x)

    if which_events == 'bg_off':
        pred_data = []
        # bg + offspring
        for i in range(Ksim):
            pred_data = save_obj_pred['pred_bgnew_and_offspring'][i]
            N_in_tau1_tau2 = np.sum(np.logical_and(pred_data[:, 0] > tau1, pred_data[:, 0] <= tau2))
            idx_test = np.logical_and(pred_data[:, 0] > tau1, pred_data[:, 0] <= tau2)
            if m0 is not None:
                N_in_tau1_tau2 = np.sum(np.logical_and(
                    np.logical_and(pred_data[:, 0] > tau1, pred_data[:, 0] <= tau2), pred_data[:, 1] >= m0))
                idx_test = np.logical_and(np.logical_and(
                    pred_data[:, 0] > tau1, pred_data[:, 0] <= tau2), pred_data[:, 1] >= m0)
            y = np.hstack([N_tau1, np.cumsum(np.ones(N_in_tau1_tau2)) + N_tau1]) - N_tau1
            x = np.hstack([tau1, pred_data[idx_test, 0]]) - tau1
            out['y_bg_off'].append(y)
            out['x_bg_off'].append(x)

    if which_events == 'Htoff':
        pred_data = []
        # Htoff only
        for i in range(Ksim):
            pred_data = save_obj_pred['pred_offspring_Ht'][i]
            N_in_tau1_tau2 = np.sum(np.logical_and(pred_data[:, 0] > tau1, pred_data[:, 0] <= tau2))
            idx_test = np.logical_and(pred_data[:, 0] > tau1, pred_data[:, 0] <= tau2)
            if m0 is not None:
                N_in_tau1_tau2 = np.sum(np.logical_and(
                    np.logical_and(pred_data[:, 0] > tau1, pred_data[:, 0] <= tau2), pred_data[:, 1] >= m0))
                idx_test = np.logical_and(np.logical_and(
                    pred_data[:, 0] > tau1, pred_data[:, 0] <= tau2), pred_data[:, 1] >= m0)
            y = np.hstack([N_tau1, np.cumsum(np.ones(N_in_tau1_tau2)) + N_tau1]) - N_tau1
            x = np.hstack([tau1, pred_data[idx_test, 0]]) - tau1
            out['y_Htoff'].append(y)
            out['x_Htoff'].append(x)

    out['m0'] = m0

    return out


def get_marginal_Nt_pred(t=None, save_obj_pred=None, m0_plot=None, which_events=None):
    """
    Extracts cummulative number of events at time t with m>=m0_plot
    :param t:time in days from the start point t=0 of the forecast
    :type t:float
    :param save_obj_pred:contains the forecast/prediction
    :type save_obj_pred:dict
    :param m0_plot:lower bound of m
    :type m0_plot:float
    :param which_events: determines which events, 'bg','off','bg_off','Htoff'
    :type which_events: str
    :return:N_t,Nobs_t
    :rtype:int,int
    """
    t0Ht, tau1, tau2 = save_obj_pred['tau_vec'][0]
    if m0_plot is None:
        m0_plot = save_obj_pred['m0']
    cumsum = cumsum_events_pred(save_obj_pred, tau1=tau1, tau2=tau1 + t, m0=m0_plot, which_events=which_events)
    if m0_plot is None:
        m0_plot = save_obj_pred['m0']
    if m0_plot < save_obj_pred['m0']:
        m0_plot = save_obj_pred['m0']
    Ksim = cumsum['Ksim']
    N_t = np.zeros(Ksim)
    for i in range(cumsum['Ksim']):
        if which_events is None:
            N_t[i] = cumsum['y'][i][cumsum['x'][i] <= t][-1]
        else:
            if len(which_events) == 0:
                N_t[i] = cumsum['y'][i][cumsum['x'][i] <= t][-1]
            else:
                N_t[i] = cumsum['y_' + which_events][i][cumsum['x_' + which_events][i] <= t][-1]
    Nobs_t = cumsum['y_obs'][cumsum['x_obs'] <= t][-1]
    return N_t, Nobs_t


# fast plotting routines

def plot_pred_hist_cumsum_Nt_at_t(t, save_obj_pred=None, save_obj_pred_mle=None, save_obj_pred_mle_silverman=None,
                                  m0_plot=None, scale=None, xlim=None):
    """
    Plots a histogram of the forecasted number of events of a region (integrated over space).
    :param t:time of the histogram
    :type t:float
    :param save_obj_pred:
    :type save_obj_pred:
    :param save_obj_pred_mle:
    :type save_obj_pred_mle:
    :param save_obj_pred_mle_silverman:
    :type save_obj_pred_mle_silverman:
    :param m0_plot:
    :type m0_plot:float
    :param scale: either 'linear' or 'log10'
    :type scale: str
    :return: hf
    :rtype: figure handle
    """
    # plot definitions
    pSIZE = 20
    plt.rc('font', size=pSIZE)
    plt.rc('axes', titlesize=pSIZE)
    if scale is None:
        scale = 'linear'

    # input
    t_slice = t  # in days
    N_t = None
    Nobs_t = None
    if save_obj_pred is not None:
        if m0_plot is None:
            m0_plot = save_obj_pred['m0']
        if m0_plot < save_obj_pred['m0']:
            m0_plot = save_obj_pred['m0']
            print('Warning: lowest predicted magnitde is:', save_obj_pred['m0'])
        tau0Htm, tau1, tau2 = save_obj_pred['tau_vec'][0]
        if t_slice > tau2 - tau1:
            t_slice = tau2 - tau1
        N_t, Nobs_t = get_marginal_Nt_pred(t=t_slice, save_obj_pred=save_obj_pred, m0_plot=m0_plot)
        Ksim = save_obj_pred['cumsum']['Ksim']
        if scale == 'log10':
            N_t = np.log10(N_t)
            Nobs_t = np.log10(Nobs_t)
    N_t_mle = None
    if save_obj_pred_mle is not None:
        if m0_plot is None:
            m0_plot = save_obj_pred_mle['m0']
        if m0_plot < save_obj_pred_mle['m0']:
            m0_plot = save_obj_pred_mle['m0']
            print('Warning: lowest predicted magnitde is:', save_obj_pred_mle['m0'])
        tau0Htm, tau1, tau2 = save_obj_pred_mle['tau_vec'][0]
        if t_slice > tau2 - tau1:
            t_slice = tau2 - tau1
        N_t_mle, Nobs_t = get_marginal_Nt_pred(t=t_slice, save_obj_pred=save_obj_pred_mle,
                                               m0_plot=m0_plot)
        Ksim = save_obj_pred_mle['cumsum']['Ksim']
        if scale == 'log10':
            N_t_mle = np.log10(N_t_mle)
            Nobs_t = np.log10(Nobs_t)
    N_t_mle_silverman = None
    if save_obj_pred_mle_silverman is not None:
        if m0_plot is None:
            m0_plot = save_obj_pred_mle_silverman['m0']
        if m0_plot < save_obj_pred_mle_silverman['m0']:
            m0_plot = save_obj_pred_mle_silverman['m0']
            print('Warning: lowest predicted magnitde is:', save_obj_pred_mle_silverman['m0'])
        tau0Htm, tau1, tau2 = save_obj_pred_mle_silverman['tau_vec'][0]
        if t_slice > tau2 - tau1: t_slice = tau2 - tau1
        N_t_mle_silverman, Nobs_t = get_marginal_Nt_pred(t=t_slice, save_obj_pred=save_obj_pred_mle_silverman,
                                                         m0_plot=m0_plot)
        Ksim = save_obj_pred_mle_silverman['cumsum']['Ksim']
        if scale == 'log10':
            N_t_mle_silverman = np.log10(N_t_mle_silverman)
            Nobs_t = np.log10(Nobs_t)

    # histograms
    bins = None
    if N_t is None and N_t_mle is not None and N_t_mle_silverman is not None:
        z = np.hstack((N_t_mle, N_t_mle_silverman))
        if scale == 'log10':
            z = z[z != -np.inf]
        bins = np.histogram(z, bins=int(np.sqrt(Ksim)))[1]  # get the bin edges
    if N_t is None and N_t_mle is not None and N_t_mle_silverman is None:
        z = N_t_mle
        if scale == 'log10':
            z = z[z != -np.inf]
        bins = np.histogram(z, bins=int(np.sqrt(Ksim)))[1]  # get the bin edges
    if N_t is None and N_t_mle is None and N_t_mle_silverman is not None:
        z = N_t_mle_silverman
        if scale == 'log10':
            z = z[z != -np.inf]
        bins = np.histogram(z, bins=int(np.sqrt(Ksim)))[1]  # get the bin edges
    if N_t is not None and N_t_mle is not None and N_t_mle_silverman is None:
        z = np.hstack((N_t, N_t_mle))
        if scale == 'log10':
            z = z[z != -np.inf]
        bins = np.histogram(z, bins=int(np.sqrt(Ksim)))[1]  # get the bin edges
    if N_t is not None and N_t_mle is None and N_t_mle_silverman is not None:
        z = np.hstack((N_t, N_t_mle_silverman))
        if scale == 'log10':
            z = z[z != -np.inf]
        bins = np.histogram(z, bins=int(np.sqrt(Ksim)))[1]  # get the bin edges
    if N_t is not None and N_t_mle is not None and N_t_mle_silverman is not None:
        z = np.hstack((N_t, N_t_mle, N_t_mle_silverman))
        if scale == 'log10':
            z = z[z != -np.inf]
        bins = np.histogram(z, bins=int(np.sqrt(Ksim)))[
            1]  # get the bin edges
    if N_t is not None and N_t_mle is None and N_t_mle_silverman is None:
        z = N_t
        if scale == 'log10':
            z = z[z != -np.inf]
        bins = np.histogram(z, bins=int(np.sqrt(Ksim)))[1]  # get the bin edges

    hf = plt.figure()
    if save_obj_pred is not None:
        plt.hist(N_t, bins=bins, density=True, facecolor='k', alpha=0.5, label='GP-E')
    if save_obj_pred_mle is not None:
        plt.hist(N_t_mle, bins=bins, density=True, facecolor='b', alpha=0.5, label='E')
    if save_obj_pred_mle_silverman is not None:
        plt.hist(N_t_mle_silverman, bins=bins, density=True, facecolor='g', alpha=0.5, label='E-S')
    plt.axvline(x=Nobs_t, color='r')
    plt.xlabel('# events')
    plt.ylabel('density')
    if scale == 'log10':
        plt.text(0.925, 0.225,
                 '$t^*$=%.1f days\n$log_{10} N_{obs}$=%.2f\n$m\geq$%.2f\n$\\tau_1$=%.1f.\n$K_{\\rm sim}$=%i' % (
                     t_slice, Nobs_t, m0_plot, tau1, Ksim),
                 transform=plt.gcf().transFigure, horizontalalignment='left')
        if xlim is not None:
            if xlim[0] <= 0:
                xlim[0] = 1
            plt.gca().set_xlim(np.log10(xlim))
    else:
        plt.text(0.925, 0.225,
                 '$t^*$=%.1f days\n$N_{obs}$=%i\n$m\geq$%.2f\n$\\tau_1$=%.1f.\n$K_{\\rm sim}$=%i' % (
                     t_slice, Nobs_t, m0_plot, tau1, Ksim),
                 transform=plt.gcf().transFigure, horizontalalignment='left')
        if xlim is not None:
            plt.gca().set_xlim(xlim)
    plt.legend(bbox_to_anchor=(1.04, 1.), loc='upper left')
    # plt.show()
    return hf


def plot_pred_cumsum_Nt_path(save_obj_pred=None, m0_plot=None, save_obj_pred_mle=None,
                             save_obj_pred_mle_silverman=None, scale='logy', which_events=None):
    # plot definitions
    pSIZE = 20
    plt.rc('font', size=pSIZE)
    plt.rc('axes', titlesize=pSIZE)
    xlim = 1e-04

    # input
    cumsum = None
    cumsum_mle = None
    cumsum_mle_silverman = None
    if save_obj_pred is not None:
        t0Ht, tau1, tau2 = save_obj_pred['tau_vec'][0]
        cumsum = cumsum_events_pred(save_obj_pred, tau1, tau2, m0=m0_plot,
                                    which_events=which_events)
        if m0_plot is None: m0_plot = save_obj_pred['m0']
        if m0_plot < save_obj_pred['m0']: m0_plot = save_obj_pred['m0']
    if save_obj_pred_mle is not None:
        t0Ht, tau1, tau2 = save_obj_pred_mle['tau_vec'][0]
        cumsum_mle = cumsum_events_pred(save_obj_pred_mle, tau1, tau2, m0=m0_plot,
                                        which_events=which_events)
        if m0_plot is None: m0_plot = save_obj_pred_mle['m0']
        if m0_plot < save_obj_pred_mle['m0']: m0_plot = save_obj_pred_mle['m0']
    if save_obj_pred_mle_silverman is not None:
        t0Ht, tau1, tau2 = save_obj_pred_mle_silverman['tau_vec'][0]
        cumsum_mle_silverman = cumsum_events_pred(save_obj_pred_mle_silverman, tau1, tau2,
                                                  m0=m0_plot, which_events=which_events)
        if m0_plot is None: m0_plot = save_obj_pred_mle_silverman['m0']
        if m0_plot < save_obj_pred_mle_silverman['m0']: m0_plot = save_obj_pred_mle_silverman['m0']

    if which_events is None: which_events = ''
    if len(which_events) == 0: key_str = ''
    if len(which_events) > 0: key_str = '_' + which_events

    hf = plt.figure(figsize=(14, 7))
    if cumsum is not None:
        x_obs = cumsum['x_obs']
        y_obs = cumsum['y_obs']
        Ksim = cumsum['Ksim']
        for i in range(cumsum['Ksim']):
            x = cumsum['x' + key_str][i]
            y = cumsum['y' + key_str][i]
            plt.step(np.append(x, tau2 - tau1), np.append(y, y[-1]), '#333333', linewidth=0.1)  # gray
    if cumsum_mle is not None:
        x_obs = cumsum_mle['x_obs']
        y_obs = cumsum_mle['y_obs']
        Ksim = cumsum_mle['Ksim']
        for i in range(cumsum_mle['Ksim']):
            x = cumsum_mle['x' + key_str][i]
            y = cumsum_mle['y' + key_str][i]
            plt.step(np.append(x, tau2 - tau1), np.append(y, y[-1]), '#3776ab', linewidth=0.1)  # some blue
    if cumsum_mle_silverman is not None:
        x_obs = cumsum_mle_silverman['x_obs']
        y_obs = cumsum_mle_silverman['y_obs']
        Ksim = cumsum_mle_silverman['Ksim']
        for i in range(cumsum_mle_silverman['Ksim']):
            x = cumsum_mle_silverman['x' + key_str][i]
            y = cumsum_mle_silverman['y' + key_str][i]
            plt.step(np.append(x, tau2 - tau1), np.append(y, y[-1]), 'g', linewidth=0.1, alpha=0.9)

    print(tau1, tau2, m0_plot)

    plt.step(np.append(x_obs, tau2 - tau1), np.append(y_obs, y_obs[-1]), 'm', linewidth=3, where='post', label='Obs.')
    plt.text(0.3, 0.775,
             '$T^*=$[%.1f %.1f] days. $m\\geq$%.2f. $|T^*|$=%.1f days.\n$N_{\\rm obs}=$%i. $K_{\\rm sim}$=%i' % (
                 tau1, tau2, m0_plot, tau2 - tau1, max(y_obs), Ksim), transform=plt.gcf().transFigure)
    plt.ylabel('counts')
    plt.xlabel('time, days')
    if scale == 'loglog':
        plt.yscale('log')
        plt.xscale('log')
        ylim = plt.gca().get_ylim()
        plt.gca().set_ylim([0.9, ylim[1]])
    if scale == 'logy':
        plt.yscale('log')
        ylim = plt.gca().get_ylim()
        plt.gca().set_ylim([0.9, ylim[1]])
    if scale == 'logx':
        plt.xscale('log')
    plt.xlim([xlim, tau2 - tau1])
    # ax = plt.gca()
    # ax.set_rasterized(True)
    plt.legend()
    # plt.show()
    return hf


def plot_pred_boxplot(t, save_obj_pred=None, save_obj_pred_mle=None, save_obj_pred_mle_silverman=None,
                      m0_plot=None, scale=None, xlim=None):
    if scale is None:
        scale = 'linear'
    t_slice = t

    N_t = None
    Nobs_t = None
    if save_obj_pred is not None:
        if m0_plot is None:
            m0_plot = save_obj_pred['m0']
        if m0_plot < save_obj_pred['m0']:
            m0_plot = save_obj_pred['m0']
            print('Warning: lowest predicted magnitde is:', save_obj_pred['m0'])
        tau0Htm, tau1, tau2 = save_obj_pred['tau_vec'][0]
        if t_slice > tau2 - tau1:
            t_slice = tau2 - tau1
        N_t, Nobs_t = get_marginal_Nt_pred(t=t_slice, save_obj_pred=save_obj_pred, m0_plot=m0_plot)
        Ksim = save_obj_pred['cumsum']['Ksim']
        if scale == 'log10':
            N_t = np.log10(N_t)
            Nobs_t = np.log10(Nobs_t)

    N_t_mle = None
    if save_obj_pred_mle is not None:
        if m0_plot is None:
            m0_plot = save_obj_pred_mle['m0']
        if m0_plot < save_obj_pred_mle['m0']:
            m0_plot = save_obj_pred_mle['m0']
            print('Warning: lowest predicted magnitde is:', save_obj_pred_mle['m0'])
        tau0Htm, tau1, tau2 = save_obj_pred_mle['tau_vec'][0]
        if t_slice > tau2 - tau1:
            t_slice = tau2 - tau1
        N_t_mle, Nobs_t = get_marginal_Nt_pred(t=t_slice, save_obj_pred=save_obj_pred_mle,
                                               m0_plot=m0_plot)
        Ksim = save_obj_pred_mle['cumsum']['Ksim']
        if scale == 'log10':
            N_t_mle = np.log10(N_t_mle)
            Nobs_t = np.log10(Nobs_t)

    N_t_mle_silverman = None
    if save_obj_pred_mle_silverman is not None:
        if m0_plot is None:
            m0_plot = save_obj_pred_mle_silverman['m0']
        if m0_plot < save_obj_pred_mle_silverman['m0']:
            m0_plot = save_obj_pred_mle_silverman['m0']
            print('Warning: lowest predicted magnitde is:', save_obj_pred_mle_silverman['m0'])
        tau0Htm, tau1, tau2 = save_obj_pred_mle_silverman['tau_vec'][0]
        if t_slice > tau2 - tau1: t_slice = tau2 - tau1
        N_t_mle_silverman, Nobs_t = get_marginal_Nt_pred(t=t_slice,
                                                         save_obj_pred=save_obj_pred_mle_silverman,
                                                         m0_plot=m0_plot)
        Ksim = save_obj_pred_mle_silverman['cumsum']['Ksim']
        if scale == 'log10':
            N_t_mle_silverman = np.log10(N_t_mle_silverman)
            Nobs_t = np.log10(Nobs_t)

    hf = plt.figure()
    if Nobs_t is not None:
        plt.axhline(y=Nobs_t, color='m', linestyle='--')
    if N_t is not None:
        plt.boxplot(N_t, positions=range(1, 2))
        plt.xticks([1], ['GP-E'])
    if N_t is not None and N_t_mle is not None:
        plt.boxplot(N_t_mle, positions=range(2, 3))
        plt.xticks([1, 2], ['GP-E', 'E'])
    if N_t is not None and N_t_mle_silverman is not None:
        plt.boxplot(N_t_mle_silverman, positions=range(2, 3))
        plt.xticks([1, 2], ['GP-E', 'E-S'])
    if N_t is not None and N_t_mle is not None and N_t_mle_silverman is not None:
        plt.boxplot(N_t_mle, positions=range(2, 3))
        plt.boxplot(N_t_mle_silverman, positions=range(3, 4))
        plt.xticks([1, 2, 3], ['GP-E', 'E', 'E-S'])
    if scale == 'log10':
        plt.ylabel('$log_{10}$ number of events')
    if scale == 'linear':
        plt.ylabel('number of events')
    # plt.show()
    return hf


def plot_pred_quantile(t, save_obj_pred=None, save_obj_pred_mle=None, save_obj_pred_mle_silverman=None,
                       m0_plot=None, scale=None, xlim=None, quantile=0.05):
    if scale is None:
        scale = 'linear'
    t_slice = t
    ms = 10

    N_t = None
    Nobs_t = None
    if save_obj_pred is not None:
        if m0_plot is None:
            m0_plot = save_obj_pred['m0']
        if m0_plot < save_obj_pred['m0']:
            m0_plot = save_obj_pred['m0']
            print('Warning: lowest predicted magnitde is:', save_obj_pred['m0'])
        tau0Htm, tau1, tau2 = save_obj_pred['tau_vec'][0]
        if t_slice > tau2 - tau1:
            t_slice = tau2 - tau1
        N_t, Nobs_t = get_marginal_Nt_pred(t=t_slice, save_obj_pred=save_obj_pred, m0_plot=m0_plot)
        Ksim = save_obj_pred['cumsum']['Ksim']
        if scale == 'log10':
            N_t = np.log10(N_t)
            Nobs_t = np.log10(Nobs_t)

    N_t_mle = None
    if save_obj_pred_mle is not None:
        if m0_plot is None:
            m0_plot = save_obj_pred_mle['m0']
        if m0_plot < save_obj_pred_mle['m0']:
            m0_plot = save_obj_pred_mle['m0']
            print('Warning: lowest predicted magnitde is:', save_obj_pred_mle['m0'])
        tau0Htm, tau1, tau2 = save_obj_pred_mle['tau_vec'][0]
        if t_slice > tau2 - tau1:
            t_slice = tau2 - tau1
        N_t_mle, Nobs_t = get_marginal_Nt_pred(t=t_slice, save_obj_pred=save_obj_pred_mle,
                                               m0_plot=m0_plot)
        Ksim = save_obj_pred_mle['cumsum']['Ksim']
        if scale == 'log10':
            N_t_mle = np.log10(N_t_mle)
            Nobs_t = np.log10(Nobs_t)

    N_t_mle_silverman = None
    if save_obj_pred_mle_silverman is not None:
        if m0_plot is None:
            m0_plot = save_obj_pred_mle_silverman['m0']
        if m0_plot < save_obj_pred_mle_silverman['m0']:
            m0_plot = save_obj_pred_mle_silverman['m0']
            print('Warning: lowest predicted magnitde is:', save_obj_pred_mle_silverman['m0'])
        tau0Htm, tau1, tau2 = save_obj_pred_mle_silverman['tau_vec'][0]
        if t_slice > tau2 - tau1: t_slice = tau2 - tau1
        N_t_mle_silverman, Nobs_t = get_marginal_Nt_pred(t=t_slice,
                                                         save_obj_pred=save_obj_pred_mle_silverman,
                                                         m0_plot=m0_plot)
        Ksim = save_obj_pred_mle_silverman['cumsum']['Ksim']
        if scale == 'log10':
            N_t_mle_silverman = np.log10(N_t_mle_silverman)
            Nobs_t = np.log10(Nobs_t)

    hf = plt.figure()
    if Nobs_t is not None:
        if scale == 'log10':
            plt.axhline(y=Nobs_t, color='m', linestyle='--', label='$\\log10 N_{obs}$=%.2f' % Nobs_t)
        else:
            plt.axhline(y=Nobs_t, color='m', linestyle='--', label='$N_{obs}$=%i' % Nobs_t)

    if N_t is not None and N_t_mle is None and N_t_mle_silverman is None:
        plt.plot(1, np.median(N_t), 'ko', markersize=ms)
        plt.plot([1, 1], [np.quantile(a=N_t, q=quantile), np.quantile(a=N_t, q=1 - quantile)], '-k')
        plt.xticks([1], ['GP-E'])
    if N_t is not None and N_t_mle is not None and N_t_mle_silverman is None:
        plt.plot(1, np.median(N_t), 'ko', markersize=ms)
        plt.plot([1, 1], [np.quantile(a=N_t, q=quantile), np.quantile(a=N_t, q=1 - quantile)], '-k')
        plt.plot(2, np.median(N_t_mle), 'ob', markersize=ms)
        plt.plot([2, 2], [np.quantile(a=N_t_mle, q=quantile), np.quantile(a=N_t_mle, q=1 - quantile)], '-b')
        plt.xticks([1, 2], ['GP-E', 'E'])
        plt.xlim([0.5, 2.5])
    if N_t is not None and N_t_mle is None and N_t_mle_silverman is not None:
        plt.plot(1, np.median(N_t), 'ko', markersize=ms)
        plt.plot([1, 1], [np.quantile(a=N_t, q=quantile), np.quantile(a=N_t, q=1 - quantile)], '-k')
        plt.plot(2, np.median(N_t_mle_silverman), 'og', markersize=ms)
        plt.plot([2, 2],
                 [np.quantile(a=N_t_mle_silverman, q=quantile), np.quantile(a=N_t_mle_silverman, q=1 - quantile)], '-g')
        plt.xticks([1, 2], ['GP-E', 'E-S'])
        plt.xlim([0.5, 2.5])
    if N_t is not None and N_t_mle is not None and N_t_mle_silverman is not None:
        plt.plot(1, np.median(N_t), 'ko', markersize=ms)
        plt.plot([1, 1], [np.quantile(a=N_t, q=quantile), np.quantile(a=N_t, q=1 - quantile)], '-k')
        plt.plot(2, np.median(N_t_mle), 'ob', markersize=ms)
        plt.plot([2, 2], [np.quantile(a=N_t_mle, q=quantile), np.quantile(a=N_t_mle, q=1 - quantile)], '-b')
        plt.plot(3, np.median(N_t_mle_silverman), 'og', markersize=ms)
        plt.plot([3, 3],
                 [np.quantile(a=N_t_mle_silverman, q=quantile), np.quantile(a=N_t_mle_silverman, q=1 - quantile)], '-g')
        plt.xticks([1, 2, 3], ['GP-E', 'E', 'E-S'])
        plt.xlim([0.5, 3.5])

    if scale == 'log10':
        plt.ylabel('$log_{10}$ number of events')
    if scale == 'linear':
        plt.ylabel('number of events')
    plt.legend(fontsize=12)
    plt.title('$\\tau_2$=%.1f days. q=%.3f' % (t_slice, quantile))

    return hf


def plot_pred_histkernel_Nt_at_t(t, save_obj_pred=None,
                                 save_obj_pred_mle=None,
                                 save_obj_pred_mle_silverman=None,
                                 m0_plot=None,
                                 scale=None,
                                 xlim=None,
                                 bw_method='silverman',
                                 nbins=None,
                                 hist=None):
    if scale is None:
        scale = 'linear'
    t_slice = t

    N_t = None
    Nobs_t = None
    if save_obj_pred is not None:
        if m0_plot is None:
            m0_plot = save_obj_pred['m0']
        if m0_plot < save_obj_pred['m0']:
            m0_plot = save_obj_pred['m0']
            print('Warning: lowest predicted magnitde is:', save_obj_pred['m0'])
        tau0Htm, tau1, tau2 = save_obj_pred['tau_vec'][0]
        if t_slice > tau2 - tau1:
            t_slice = tau2 - tau1
        N_t, Nobs_t = get_marginal_Nt_pred(t=t_slice, save_obj_pred=save_obj_pred, m0_plot=m0_plot)
        Ksim = save_obj_pred['cumsum']['Ksim']
        if scale == 'log10':
            N_t = np.log10(N_t)
            N_t = N_t[N_t != -np.inf]
            Nobs_t = np.log10(Nobs_t)

    N_t_mle = None
    if save_obj_pred_mle is not None:
        if m0_plot is None:
            m0_plot = save_obj_pred_mle['m0']
        if m0_plot < save_obj_pred_mle['m0']:
            m0_plot = save_obj_pred_mle['m0']
            print('Warning: lowest predicted magnitde is:', save_obj_pred_mle['m0'])
        tau0Htm, tau1, tau2 = save_obj_pred_mle['tau_vec'][0]
        if t_slice > tau2 - tau1:
            t_slice = tau2 - tau1
        N_t_mle, Nobs_t = get_marginal_Nt_pred(t=t_slice, save_obj_pred=save_obj_pred_mle,
                                               m0_plot=m0_plot)
        Ksim = save_obj_pred_mle['cumsum']['Ksim']
        if scale == 'log10':
            N_t_mle = np.log10(N_t_mle)
            N_t_mle = N_t_mle[N_t_mle != -np.inf]
            Nobs_t = np.log10(Nobs_t)

    N_t_mle_silverman = None
    if save_obj_pred_mle_silverman is not None:
        if m0_plot is None:
            m0_plot = save_obj_pred_mle_silverman['m0']
        if m0_plot < save_obj_pred_mle_silverman['m0']:
            m0_plot = save_obj_pred_mle_silverman['m0']
            print('Warning: lowest predicted magnitde is:', save_obj_pred_mle_silverman['m0'])
        tau0Htm, tau1, tau2 = save_obj_pred_mle_silverman['tau_vec'][0]
        if t_slice > tau2 - tau1: t_slice = tau2 - tau1
        N_t_mle_silverman, Nobs_t = get_marginal_Nt_pred(t=t_slice,
                                                         save_obj_pred=save_obj_pred_mle_silverman,
                                                         m0_plot=m0_plot)
        Ksim = save_obj_pred_mle_silverman['cumsum']['Ksim']
        if scale == 'log10':
            N_t_mle_silverman = np.log10(N_t_mle_silverman)
            N_t_mle_silverman = N_t_mle_silverman[N_t_mle_silverman != -np.inf]
            Nobs_t = np.log10(Nobs_t)

    if N_t is not None:
        kde_N_t = sc.stats.gaussian_kde(dataset=N_t, bw_method=bw_method, weights=None)
        print('kde bw = ', kde_N_t.factor)
    if N_t_mle is not None:
        kde_N_t_mle = sc.stats.gaussian_kde(dataset=N_t_mle, bw_method=bw_method, weights=None)
        print('kde bw = ', kde_N_t_mle.factor)
    if N_t_mle_silverman is not None:
        kde_N_t_mle_silverman = sc.stats.gaussian_kde(dataset=N_t_mle_silverman, bw_method=bw_method, weights=None)
        print('kde bw = ', kde_N_t_mle_silverman.factor)

    # all same bins
    if nbins is None:
        nbins = int(np.sqrt(Ksim))
    if N_t is not None and N_t_mle is None and N_t_mle_silverman is None:
        bins = np.histogram(np.hstack(N_t), bins=nbins)[1]  # get the bin edges
    if N_t is not None and N_t_mle is not None and N_t_mle_silverman is None:
        bins = np.histogram(np.hstack((N_t, N_t_mle)), bins=nbins)[1]  # get the bin edges
    if N_t is not None and N_t_mle is None and N_t_mle_silverman is not None:
        bins = np.histogram(np.hstack((N_t, N_t_mle_silverman)), bins=nbins)[1]  # get the bin edges
    if N_t is not None and N_t_mle is not None and N_t_mle_silverman is not None:
        bins = np.histogram(np.hstack((N_t, N_t_mle, N_t_mle_silverman)), bins=nbins)[1]  # get the bin edges

    hf = plt.figure()
    x = np.linspace(0.1, 1.1 * np.max(bins), 1000)
    if N_t is not None:
        if hist is not None:
            plt.hist(N_t, bins=bins, density=True, facecolor='k', alpha=0.5)
        plt.plot(x, kde_N_t.pdf(x), 'k', label='GP-E')
    if N_t_mle is not None:
        if hist is not None:
            plt.hist(N_t_mle, bins=bins, density=True, facecolor='b', alpha=0.5)
        plt.plot(x, kde_N_t_mle.pdf(x), 'b', label='E')
    if N_t_mle_silverman is not None:
        if hist is not None:
            plt.hist(N_t_mle_silverman, bins=bins, density=True, facecolor='g', alpha=0.5)
        plt.plot(x, kde_N_t_mle_silverman.pdf(x), 'g', label='E-S')
    plt.ylabel('density')
    if scale == 'log10':
        plt.xlabel('$log_{10}$ number of events')
        plt.axvline(x=Nobs_t, color='m', linestyle='--', label='$\\log_{10} N_{\\rm obs}$')
        if xlim is not None:
            plt.xlim(np.log10(xlim))
    if scale == 'linear':
        plt.xlabel('number of events')
        plt.axvline(x=Nobs_t, color='m', linestyle='--', label='$N_{\\rm obs}$')
    plt.title('$\\tau_2$=%.1f days. $m\\geq%.2f$' % (t_slice, m0_plot))
    plt.legend(fontsize=12)

    return hf


# spatial prediction
def plot_pred_prediction2d(t, pred_data, save_obj_pred, scale=None, quantile=None,
                           m0_plot=None, nbins=None,
                           contour_lines=1, cl_color=None,
                           points=None, data_testing_points=None, data_training_points=None,
                           cmap_dots=None, dt_points=10, clim=None):
    tau0Htm, tau1, tau2 = save_obj_pred['tau_vec'][0]
    t_slice = t
    if t_slice is None or t_slice > tau2 - tau1:
        t_slice = tau2 - tau1
    if m0_plot is None or m0_plot < save_obj_pred['m0']:
        m0_plot = save_obj_pred['m0']
        print('Warning: lowest predicted magnitde is:', save_obj_pred['m0'])
    if scale is None:
        scale = 'linear'

    # plot definitions
    pSIZE = 20
    plt.rc('font', size=pSIZE)
    plt.rc('axes', titlesize=pSIZE)

    if pred_data is None:
        if save_obj_pred is not None:
            pred_data = save_obj_pred['pred_bgnew_and_offspring_with_Ht_offspring']
            if m0_plot is None:
                m0_plot = save_obj_pred['m0']
            if m0_plot < save_obj_pred['m0']:
                m0_plot = save_obj_pred['m0']
                print('Warning: lowest predicted magnitde is:', save_obj_pred['m0'])

    Ksim = len(pred_data)
    if m0_plot is None:
        m0_plot = save_obj_pred['data_obj'].domain.m0
    if nbins is None:
        nbins = 50
    L = nbins ** 2
    range_hist = save_obj_pred['data_obj'].domain.X_borders

    print(m0_plot)
    h2d_arr = np.empty([Ksim, nbins, nbins]) * np.nan
    for k in range(Ksim):
        data_k = pred_data[k]
        idx = np.logical_and(data_k[:, 1] >= m0_plot,
                             np.logical_and(data_k[:, 0] > tau1, data_k[:, 0] <= t_slice + tau1))
        H, xedges, yedges = np.histogram2d(np.array(data_k[idx, 2]),
                                           np.array(data_k[idx, 3]), bins=nbins,
                                           range=range_hist)
        h2d_arr[k, :, :] = H.T

    X_borders = save_obj_pred['data_obj'].domain.X_borders
    midpoints = np.zeros([2, nbins])
    delta = np.diff(X_borders) / nbins
    dx1 = delta[0]
    dx2 = delta[1]
    midpoints[0, :] = np.arange(X_borders[0, 0] + dx1 / 2, X_borders[0, 1], dx1)
    midpoints[1, :] = np.arange(X_borders[1, 0] + dx2 / 2, X_borders[1, 1], dx2)
    x = midpoints[0, :]
    y = midpoints[1, :]
    real_x = np.unique(x)
    real_y = np.unique(y)
    dx = (real_x[1] - real_x[0]) / 2.
    dy = (real_y[1] - real_y[0]) / 2.
    extent = [real_x[0] - dx, real_x[-1] + dx, real_y[0] - dy, real_y[-1] + dy]

    hf = plt.figure()
    if quantile is None:
        z = np.mean(h2d_arr, axis=0)
        plt.title('mean forecast: $\\tau_2$=%.1f days. $m\\geq%.2f. K_{\\rm sim}$=%i' % (t_slice, m0_plot, Ksim))
    else:
        z = np.quantile(a=h2d_arr, q=quantile, axis=0)
        plt.title('q=%.2f: $\\tau_2$=%.1f days. $m\\geq%.2f$' % (quantile, t_slice, m0_plot))
    z = z / (dx1 * dx2) / (tau2 - tau1)
    if scale == 'log10':
        z = np.log10(z)

    h = plt.imshow(z, origin='lower', extent=extent)
    cb = plt.colorbar(shrink=.5)
    if scale == 'linear':
        cb.set_label('$\hat\lambda$, [1/(day*deg$^2$)]')
    if scale == 'log10':
        cb.set_label('$\log_{10}\ \hat\lambda$')

    # clim
    if clim is not None:
        h.set_clim(clim[0], clim[1])

    # contour lines
    if contour_lines is not None:
        contour_lines = np.linspace(np.nanmin(z[z != -np.inf]), np.nanmax(z[z != -np.inf]), 10)
        if cl_color is None:
            cl_color = 'k'
        h = plt.contour(x, y, z, contour_lines.T, colors=cl_color, linestyles=':')
        if np.max(contour_lines) < 0.0001:
            plt.clabel(h, fontsize=9, inline=1, fmt='%2.2E')
        else:
            if scale == 'log10':
                plt.clabel(h, fontsize=9, inline=1, fmt='%2.1f')
            else:
                plt.clabel(h, fontsize=9, inline=1, fmt='%2.4f')

    # points
    if cmap_dots is None:
        cmap_dots = 'gray'
    data_obj = save_obj_pred['data_obj']
    if points is not None:
        plt.scatter(points[:, 0], points[:, 1], s=10, c='red')  # s=10
    if data_testing_points is not None:
        # idx_testing = np.where((data_obj.data_all.times >= data_obj.domain.T_borders_testing[0]))
        if dt_points is None:
            dt_points = 10.
        T1 = t_slice + tau1
        T2 = t_slice + tau1 + dt_points
        if T1 > data_obj.domain.T_borders_all[1]:
            T1 = data_obj.domain.T_borders_all[1]
            T2 = data_obj.domain.T_borders_all[1]
            print('Warning: no data for forecast period!')
        if T2 > data_obj.domain.T_borders_all[1]:
            T2 = data_obj.domain.T_borders_all[1]
            print('Warning: forecast period is shortend to %f days instead of %f days.' % (
                T2 - (t_slice + tau1), dt_points))
        idx_testing = np.logical_and(np.logical_and(data_obj.data_all.times >= T1,
                                                    data_obj.data_all.times <= T2),
                                     data_obj.data_all.magnitudes >= m0_plot)
        points = data_obj.data_all.positions[idx_testing]
        points_time = data_obj.data_all.times[idx_testing]
        if len(cmap_dots) == 1:
            im = plt.scatter(points[:, 0], points[:, 1], s=15, c=cmap_dots, vmin=0,
                             vmax=data_obj.domain.T_borders_all[1])
        else:
            im = plt.scatter(points[:, 0], points[:, 1], s=15, c=points_time, cmap=cmap_dots, vmin=0,
                             vmax=data_obj.domain.T_borders_all[1])
    if data_training_points is not None:
        idx_training = np.where((data_obj.data_all.times >= data_obj.domain.T_borders_training[0]) & (
                data_obj.data_all.times <= data_obj.domain.T_borders_training[1]))
        points = data_obj.data_all.positions[idx_training]
        points_time = data_obj.data_all.times[idx_training]
        if len(cmap_dots) == 1:
            im = plt.scatter(points[:, 0], points[:, 1], s=15, c=cmap_dots, vmin=0,
                             vmax=data_obj.domain.T_borders_training[1])
        else:
            im = plt.scatter(points[:, 0], points[:, 1], s=15, c=points_time, cmap=cmap_dots, vmin=0,
                             vmax=data_obj.domain.T_borders_training[1])

    clim_out = h.get_clim()
    plt.xlabel('$x_1$,  (Lon.)')
    plt.ylabel('$x_2$,  (Lat.)')
    print(np.sum(z) * (tau2 - tau1) * (dx1 * dx2))

    return hf, x, y, z, clim_out


# uncertainty in time
def plot_pred_uncertainty_in_time(save_obj_pred, save_obj_pred_mle=None,
                                  m0_plot=None, scale=None, res=5, q01_plot=None):
    if scale is None:
        scale = 'linear'
    tau0Htm, tau1, tau2 = save_obj_pred['tau_vec'][0]
    t_vec = np.linspace(0., tau2 - tau1, res)
    if (tau2 - tau1) >= 60:
        if sum(t_vec == 1.) == 0:
            t_vec = np.sort(np.append(t_vec, 1.))
        if sum(t_vec == 10.) == 0:
            t_vec = np.sort(np.append(t_vec, 10.))
        if sum(t_vec == 30.) == 0:
            t_vec = np.sort(np.append(t_vec, 30.))
        if sum(t_vec == 60.) == 0:
            t_vec = np.sort(np.append(t_vec, 60.))
    if (tau2 - tau1) >= 180:
        if sum(t_vec == 180.) == 0:
            t_vec = np.sort(np.append(t_vec, 180.))
    print(t_vec)
    if m0_plot is None:
        m0_plot = save_obj_pred['save_obj_GS']['data_obj'].domain.m0
    Ksim = len(save_obj_pred['pred_bgnew_and_offspring'])
    out_stats = np.zeros([len(t_vec) - 1, 9])
    out_stats_mle = np.zeros([len(t_vec) - 1, 9])

    for j in np.arange(1, len(t_vec)):
        t = t_vec[j]
        # gpetas
        N_t, Nobs = get_marginal_Nt_pred(t, save_obj_pred, m0_plot=m0_plot, which_events=None)

        if scale == 'log10':
            z = []
            z = np.log10(N_t)
            Nobs = np.log10(Nobs)
        if scale == 'linear':
            z = N_t

        # np.nanmin(z[z != -np.inf])
        m, v, q01, q05, q5, q95, q99 = [np.nanmean(z[z != -np.inf]),
                                        np.nanvar(z[z != -np.inf]),
                                        np.nanquantile(z[z != -np.inf], q=0.01),
                                        np.nanquantile(z[z != -np.inf], q=0.05),
                                        np.nanquantile(z[z != -np.inf], q=0.5),
                                        np.nanquantile(z[z != -np.inf], q=0.95),
                                        np.nanquantile(z[z != -np.inf], q=0.99)]
        out_stats[j - 1, :] = [t, Nobs, m, v, q01, q05, q5, q95, q99]

        if save_obj_pred_mle is not None:
            N_t_mle, Nobs_mle = get_marginal_Nt_pred(t, save_obj_pred_mle, m0_plot=m0_plot,
                                                     which_events=None)

            if scale == 'log10':
                z = []
                z = np.log10(N_t_mle)
                # Nobs = np.log10(Nobs)
            if scale == 'linear':
                z = N_t_mle

            # np.nanmin(z[z != -np.inf])
            m, v, q01, q05, q5, q95, q99 = [np.nanmean(z[z != -np.inf]),
                                            np.nanvar(z[z != -np.inf]),
                                            np.nanquantile(z[z != -np.inf], q=0.01),
                                            np.nanquantile(z[z != -np.inf], q=0.05),
                                            np.nanquantile(z[z != -np.inf], q=0.5),
                                            np.nanquantile(z[z != -np.inf], q=0.95),
                                            np.nanquantile(z[z != -np.inf], q=0.99)]
            out_stats_mle[j - 1, :] = [t, Nobs, m, v, q01, q05, q5, q95, q99]

    # plot definitions
    pSIZE = 20
    plt.rc('font', size=pSIZE)
    plt.rc('axes', titlesize=pSIZE)
    xlim = 1e-04

    hf = plt.figure()
    # plt.step(out_stats[:,0],out_stats[:,1], 'm', linewidth=3, where='post', label='Obs.')
    x = out_stats[:, 0]
    y = out_stats[:, 1]
    plt.plot(x, y, '-m',
             linewidth=3, label='Obs.')
    plt.plot(out_stats[:, 0], out_stats[:, 2], 'k',
             linewidth=3, label='GP-E')

    if q01_plot is not None:
        quantile = 0.01
        plt.fill_between(out_stats[:, 0],
                         y1=out_stats[:, 4],
                         y2=out_stats[:, 8],
                         color='whitesmoke',
                         label='$q_{%.2f,%.2f}$' % (quantile, 1 - quantile))

    quantile = 0.05
    plt.fill_between(out_stats[:, 0],
                     y1=out_stats[:, 5],
                     y2=out_stats[:, 7],
                     color='lightgrey',
                     label='$q_{%.2f,%.2f}$' % (quantile, 1 - quantile))

    plt.plot(out_stats_mle[:, 0], out_stats_mle[:, 2], '--b',
             linewidth=3, label='E')
    if q01_plot is not None:
        quantile = 0.01
        plt.plot(out_stats_mle[:, 0], out_stats_mle[:, 4], ':b',
                 linewidth=1, label='$q_{%.2f,%.2f}$' % (quantile, 1 - quantile))
        plt.plot(out_stats_mle[:, 0], out_stats_mle[:, 8], ':b',
                 linewidth=1)
    quantile = 0.05
    plt.plot(out_stats_mle[:, 0], out_stats_mle[:, 5], '--b',
             linewidth=1, label='$q_{%.2f,%.2f}$' % (quantile, 1 - quantile))
    plt.plot(out_stats_mle[:, 0], out_stats_mle[:, 7], '--b',
             linewidth=1)

    xticks = plt.gca().get_xticks()
    xticks = plt.gca().set_xticks(np.append(np.min(out_stats[:, 0]), xticks[xticks > 0]))
    plt.legend(bbox_to_anchor=(1.04, 1.), loc='upper left')
    plt.text(0.05, 0.9, '$K_{sim}$=%i\n $m\\geq$%.2f' % (Ksim, m0_plot), horizontalalignment='left',
             verticalalignment='center', transform=plt.gca().transAxes)
    plt.xlabel('time, days')
    if scale == 'log10':
        plt.ylabel('$\\log_{10}$ cumulative $N^\\ast$')
    if scale == 'linear':
        plt.ylabel('cumulative $N^\\ast$')

    return out_stats, out_stats_mle, hf


def generate_tau1_tau2_vec_seq_forecast(tau1_forecast, tau2_forecast, save_obj_GS, dt=1, m_star=None, eps=None):
    """
    Generates two vectors for sequential predictions in intervals
    :param tau1_forecast:
    :type tau1_forecast:
    :param tau2_forecast:
    :type tau2_forecast:
    :param save_obj_GS:
    :type save_obj_GS:
    :param dt:
    :type dt:
    :param m_star:
    :type m_star:
    :param eps:
    :type eps:
    :return:
    :rtype:
    """
    if m_star is None:
        m_star = 10.
    if eps is None:
        eps = 1e-6

    data_obj = save_obj_GS['data_obj']

    # big shocks with m_star
    time_big_events = data_obj.data_all.times[data_obj.data_all.magnitudes > m_star]
    mag_big_events = data_obj.data_all.magnitudes[data_obj.data_all.magnitudes > m_star]

    if len(time_big_events) != 0:
        # genereation of sequence of t_i,t_i+dt values
        tau1_vec = [tau1_forecast]
        tau2_vec = None
        for i in range(len(time_big_events)):
            t1 = time_big_events[i]
            if tau1_vec[-1] < t1:
                tau1_vec = np.append(tau1_vec[:-1], np.arange(tau1_vec[-1], t1, dt))
                tau1_vec = np.append(tau1_vec, t1 + eps)
        if tau1_vec[-1] < tau2_forecast:
            tau1_vec = np.append(tau1_vec[:-1], np.arange(tau1_vec[-1], tau2_forecast, dt))

        # tau2_vec (ending times of forecasts)
        tau2_vec = np.append(tau1_vec[1:], tau1_vec[-1] + dt)

        # check
        for i in range(len(time_big_events)):
            t1 = time_big_events[i]
            # print(np.where(tau2_vec == t1))
        for i in range(len(time_big_events)):
            t1 = time_big_events[i]
            # print(np.where(tau1_vec == t1))

    else:
        if dt == tau2_forecast - tau1_forecast:
            tau1_vec = np.append(tau1_forecast, [])
            tau2_vec = np.append(tau2_forecast, [])
        else:
            tau1_vec = np.append(tau1_forecast, np.arange(tau1_forecast, tau2_forecast, dt)[1:-1])
            tau2_vec = np.arange(tau1_forecast, tau2_forecast, dt)[1:]

    idx_in = tau1_vec <= tau2_forecast

    return tau1_vec[idx_in], tau2_vec[idx_in]


class forecast_sequential():
    def __init__(self, save_obj_GS, tau1_forecast, tau2_forecast, dt_update=None, tau0_Ht=None,
                 Ksim=None, sample_idx_vec=None, mle_obj=None, m0_plot=None, m_star=None):
        self.save_obj_GS = save_obj_GS
        self.mle_obj = mle_obj
        if Ksim is None:
            Ksim = 100
        self.Ksim = Ksim
        if sample_idx_vec is None:
            sample_idx_vec = np.arange(0, len(save_obj_GS['lambda_bar']), 1)  # all samples
        self.sample_idx_vec = sample_idx_vec
        dt = dt_update
        if dt is None:
            dt = int((tau2_forecast - tau1_forecast) / 10)  # in days
            if dt == 0:
                dt = (tau2_forecast - tau1_forecast) / 10.
        if tau1_forecast + dt > tau2_forecast:
            dt = tau2_forecast - tau1_forecast
            print('Warning: dt is too large, dt is set to:', dt, ' days.')
        self.dt_update = dt

        self.tau1_forecast = tau1_forecast
        self.tau2_forecast = tau2_forecast
        print(self.tau1_forecast, self.tau2_forecast, self.dt_update)
        # if dt == tau2_forecast - tau1_forecast:
        #    tau1_vec = np.append(tau1_forecast, [])
        #    tau2_vec = np.append(tau2_forecast, [])
        # else:
        #    tau1_vec = np.append(tau1_forecast, np.arange(tau1_forecast, tau2_forecast, dt)[1:-1])
        #    tau2_vec = np.arange(tau1_forecast, tau2_forecast, dt)[1:]

        tau1_vec, tau2_vec = generate_tau1_tau2_vec_seq_forecast(tau1_forecast, tau2_forecast, save_obj_GS, dt=dt,
                                                                 m_star=m_star, eps=None)
        print(tau1_vec, tau2_vec)

        # history
        if tau0_Ht is None:
            tau0_Ht = tau1_forecast - 100
        self.tau0_Ht = tau0_Ht

        N_t_array = []
        N_t_array_mle = []
        Nobs_array = []

        for i in range(len(tau2_vec)):
            tau1 = tau1_vec[i]
            tau2 = tau2_vec[i]
            blockPrint()
            # gpetas
            pred_obj = predictions_gpetas(save_obj_GS=save_obj_GS,
                                          tau1=tau1,
                                          tau2=tau2,
                                          tau0_Ht=tau0_Ht,
                                          sample_idx_vec=sample_idx_vec,
                                          Ksim=Ksim,
                                          seed=None,
                                          approx=None,
                                          randomized_samples=None,
                                          Bayesian_m_beta=None)

            N_t, Nobs = get_marginal_Nt_pred(tau2 - tau1,
                                             save_obj_pred=pred_obj.save_pred,
                                             m0_plot=m0_plot, which_events=None)
            print(tau2, Nobs)
            N_t_array.append(N_t)
            Nobs_array.append(Nobs)

            # mle
            if mle_obj is not None:
                pred_obj_mle = predictions_mle(mle_obj=mle_obj,
                                               tau1=tau1,
                                               tau2=tau2,
                                               tau0_Ht=tau0_Ht,
                                               Ksim=Ksim,
                                               seed=None,
                                               Bayesian_m_beta=None)
                N_t_mle, Nobs = get_marginal_Nt_pred(tau2 - tau1,
                                                     save_obj_pred=pred_obj_mle.save_pred,
                                                     m0_plot=m0_plot, which_events=None)
                N_t_array_mle.append(N_t_mle)
            enablePrint()
            print(i, ' of ', range(len(tau2_vec)))

        self.N_t_array = N_t_array
        if mle_obj is not None:
            self.N_t_array_mle = N_t_array_mle
        self.Nobs_array = Nobs_array
        self.tau1_vec = tau1_vec
        self.tau2_vec = tau2_vec
        self.pred_obj = pred_obj


### summary of plots and tables
def pred_summary(save_obj_pred=None, save_obj_pred_mle=None, save_obj_pred_mle_silverman=None, m0_plot=None, res=None):
    init_outdir()
    if save_obj_pred is not None:
        case_name = save_obj_pred['data_obj'].case_name
        t0Ht, tau1, tau2 = save_obj_pred['tau_vec'][0]
        if m0_plot is None: m0_plot = save_obj_pred['m0']
    if save_obj_pred_mle is not None:
        case_name = save_obj_pred_mle['data_obj'].case_name
        t0Ht, tau1, tau2 = save_obj_pred_mle['tau_vec'][0]
        if m0_plot is None: m0_plot = save_obj_pred_mle['m0']
    if save_obj_pred_mle_silverman is not None:
        case_name = save_obj_pred_mle_silverman['data_obj'].case_name
        t0Ht, tau1, tau2 = save_obj_pred_mle_silverman['tau_vec'][0]
        if m0_plot is None: m0_plot = save_obj_pred_mle_silverman['m0']
    if res is None:
        res = 5

    # FIG 01: sample path
    scales = ['linear', 'logy', 'logx', 'loglog']
    for i in range(len(scales)):
        scale = scales[i]
        hf = plot_pred_cumsum_Nt_path(save_obj_pred=save_obj_pred, m0_plot=m0_plot, save_obj_pred_mle=save_obj_pred_mle,
                                      save_obj_pred_mle_silverman=save_obj_pred_mle_silverman, scale=scale,
                                      which_events=None)
        hf.savefig(output_dir_figures + '/F001_pred_%s_%0i_%s.png' % (case_name, i, scales[i]), bbox_inches='tight')

    # FIG 02: Nt histograms at t
    # FIG 03: Nt histogram and kernel
    t_vec = np.linspace(0., tau2 - tau1, 10)
    if (tau2 - tau1) >= 60:
        if sum(t_vec == 1.) == 0:
            t_vec = np.sort(np.append(t_vec, 1.))
        if sum(t_vec == 10.) == 0:
            t_vec = np.sort(np.append(t_vec, 10.))
        if sum(t_vec == 30.) == 0:
            t_vec = np.sort(np.append(t_vec, 30.))
        if sum(t_vec == 60.) == 0:
            t_vec = np.sort(np.append(t_vec, 60.))
    if (tau2 - tau1) >= 180:
        if sum(t_vec == 180.) == 0:
            t_vec = np.sort(np.append(t_vec, 180.))
    print(t_vec)
    scales = ['linear', 'log10']
    for j in np.arange(1, len(t_vec)):
        t = t_vec[j]
        print(t)
        for i in range(len(scales)):
            scale = scales[i]
            hf = plot_pred_hist_cumsum_Nt_at_t(t=t, save_obj_pred=save_obj_pred, save_obj_pred_mle=save_obj_pred_mle,
                                               save_obj_pred_mle_silverman=save_obj_pred_mle_silverman, m0_plot=m0_plot,
                                               scale=scale)
            hf.savefig(output_dir_figures + '/F002_pred_%s_%0i_%0i_%s_m%i.pdf' % (
                case_name, i, j, scales[i], int(m0_plot * 10)), bbox_inches='tight')
            hf = plot_pred_histkernel_Nt_at_t(t, save_obj_pred=save_obj_pred,
                                              save_obj_pred_mle=save_obj_pred_mle,
                                              save_obj_pred_mle_silverman=save_obj_pred_mle_silverman,
                                              m0_plot=m0_plot,
                                              scale=scale,
                                              xlim=None,
                                              bw_method='silverman',
                                              nbins=None,
                                              hist=None)
            hf.savefig(output_dir_figures + '/F003_pred_kernel_%s_%0i_%0i_%s_m%i.pdf' % (
                case_name, i, j, scales[i], int(m0_plot * 10)), bbox_inches='tight')

            hf = plot_pred_quantile(t, save_obj_pred=save_obj_pred,
                                    save_obj_pred_mle=save_obj_pred_mle,
                                    save_obj_pred_mle_silverman=save_obj_pred_mle_silverman,
                                    m0_plot=m0_plot,
                                    scale=scale, xlim=None, quantile=0.05)
            hf.savefig(output_dir_figures + '/F004_pred_quantiles_%s_%0i_%0i_%s_m%i.pdf' % (
                case_name, i, j, scales[i], int(m0_plot * 10)), bbox_inches='tight')

            hf = plot_pred_boxplot(t, save_obj_pred=save_obj_pred,
                                   save_obj_pred_mle=save_obj_pred_mle,
                                   save_obj_pred_mle_silverman=save_obj_pred_mle_silverman,
                                   m0_plot=m0_plot, scale=scale, xlim=None)
            hf.savefig(output_dir_figures + '/F005_pred_boxplot_%s_%0i_%0i_%s_m%i.pdf' % (
                case_name, i, j, scales[i], int(m0_plot * 10)), bbox_inches='tight')

    # summary uncertainty in time
    scales = ['linear', 'log10']
    for i in range(len(scales)):
        scale = scales[i]
        out, out_mle, hf = gpetas.prediction_2d.plot_pred_uncertainty_in_time(save_obj_pred=save_obj_pred,
                                                                              save_obj_pred_mle=save_obj_pred_mle,
                                                                              m0_plot=m0_plot, scale=scale,
                                                                              res=res, q01_plot=None)
        hf.savefig(output_dir_figures + '/F007_summary_unc_t_%s_%0i_%s_m%i.pdf' % (
            case_name, i, scales[i], int(m0_plot * 10)), bbox_inches='tight')
        out, out_mle, hf = gpetas.prediction_2d.plot_pred_uncertainty_in_time(save_obj_pred=save_obj_pred,
                                                                              save_obj_pred_mle=save_obj_pred_mle,
                                                                              m0_plot=m0_plot, scale=scale,
                                                                              res=res, q01_plot=1)
        hf.savefig(output_dir_figures + '/F007_summary_unc_t_q01_%s_%0i_%s_m%i.pdf' % (
            case_name, i, scales[i], int(m0_plot * 10)), bbox_inches='tight')

    # 2D prediction
    scale = 'log10'
    t = tau2
    if save_obj_pred is not None:
        pred_data = save_obj_pred['pred_bgnew_and_offspring_with_Ht_offspring']
        save_obj_pred = save_obj_pred
        hf, x, y, z, clim_out = plot_pred_prediction2d(t, pred_data, save_obj_pred, scale=scale, quantile=None,
                                                       m0_plot=None, nbins=None,
                                                       contour_lines=None, cl_color=None,
                                                       points=None, data_testing_points=None, data_training_points=None,
                                                       cmap_dots=None, dt_points=10, clim=None)
        hf.savefig(output_dir_figures + '/F006_pred_2D_%s_%s_m%i_gpetas.pdf' % (
            case_name, scale, int(m0_plot * 10)), bbox_inches='tight')
    # mle
    if save_obj_pred_mle is not None:
        pred_data = save_obj_pred_mle['pred_bgnew_and_offspring_with_Ht_offspring']
        hf, x, y, z, clim_out = plot_pred_prediction2d(t, pred_data, save_obj_pred, scale=scale, quantile=None,
                                                       m0_plot=None, nbins=None,
                                                       contour_lines=None, cl_color=None,
                                                       points=None, data_testing_points=None, data_training_points=None,
                                                       cmap_dots=None, dt_points=10, clim=clim_out)
        hf.savefig(output_dir_figures + '/F006_pred_2D_%s_%s_m%i_mle.pdf' % (
            case_name, scale, int(m0_plot * 10)), bbox_inches='tight')
    # mle_silverman
    if save_obj_pred_mle_silverman is not None:
        pred_data = save_obj_pred_mle_silverman['pred_bgnew_and_offspring_with_Ht_offspring']
        hf, x, y, z, clim_out = plot_pred_prediction2d(t, pred_data, save_obj_pred, scale=scale, quantile=None,
                                                       m0_plot=None, nbins=None,
                                                       contour_lines=None, cl_color=None,
                                                       points=None, data_testing_points=None, data_training_points=None,
                                                       cmap_dots=None, dt_points=10, clim=clim_out)
        hf.savefig(output_dir_figures + '/F006_pred_2D_%s_%s_m%i_mle_silverman.pdf' % (
            case_name, scale, int(m0_plot * 10)), bbox_inches='tight')

    return


def plot_pred_seq_forecast_updated(pred_seq, mle_only=None, gpetas_only=None,
                                   quantile=None, ylim=None, NB_fit=None, yscale=None,
                                   markersize=None):
    if quantile is None:
        quantile = 0.05
    if yscale is None:
        yscale = 'log'
    if markersize is None:
        markersize = 15

    t = pred_seq.tau2_vec
    t0 = np.hstack([pred_seq.tau1_forecast, t])
    N_t_array = pred_seq.N_t_array
    Nobs_array = pred_seq.Nobs_array
    Ksim = pred_seq.Ksim
    tau1_forecast = pred_seq.tau1_forecast
    tau2_forecast = pred_seq.tau2_forecast

    h1 = plt.figure(figsize=[15, 5])

    if mle_only is None or gpetas_only is not None:
        plt.plot(t, np.mean(N_t_array, axis=1), 'k')
        plt.fill_between(t, y1=np.quantile(N_t_array, q=quantile, axis=1),
                         y2=np.quantile(N_t_array, q=1. - quantile, axis=1),
                         color='lightgrey',
                         label='$q_{%.2f,%.2f}$' % (quantile, 1 - quantile))
        # plt.plot(t,np.quantile(N_t_array,q=quantile,axis=1),':k')
        # plt.plot(t,np.quantile(N_t_array,q=1.-quantile,axis=1),':k')
    if pred_seq.mle_obj is not None and gpetas_only is None:
        plt.plot(t, np.mean(pred_seq.N_t_array_mle, axis=1), '--b', linewidth=1)
        plt.plot(t, np.quantile(pred_seq.N_t_array_mle, q=quantile, axis=1), ':b', linewidth=1)
        plt.plot(t, np.quantile(pred_seq.N_t_array_mle, q=1. - quantile, axis=1), ':b', linewidth=1)
        if mle_only is not None:
            plt.fill_between(t, y1=np.quantile(pred_seq.N_t_array_mle, q=quantile, axis=1),
                             y2=np.quantile(pred_seq.N_t_array_mle, q=1. - quantile, axis=1),
                             color='lightblue',
                             label='$q_{%.2f,%.2f}$' % (quantile, 1 - quantile))
    # obs
    plt.plot(t, Nobs_array, '.m', markersize=markersize)
    plt.yscale(yscale)

    N_t_array_updated = np.zeros(np.shape(N_t_array))
    for i in range(np.array(N_t_array).shape[0]):
        if i == 0:
            N_t_array_updated[i, :] = np.array(N_t_array)[i, :]
        if i > 0:
            N_t_array_updated[i, :] = np.array(N_t_array)[i, :] + Nobs_array[i]

    Nobs_array_cumsum = np.cumsum(Nobs_array)
    N_t_array_cumsum = np.cumsum(N_t_array)
    N_t_array_cumsum_updated = np.zeros(np.shape(N_t_array))
    for i in range(np.array(N_t_array).shape[0]):
        if i == 0:
            N_t_array_cumsum_updated[i, :] = np.array(N_t_array)[i, :] + 0.
        if i > 0:
            N_t_array_cumsum_updated[i, :] = np.array(N_t_array)[i, :] + Nobs_array_cumsum[i - 1]
    if pred_seq.mle_obj is not None:
        N_t_array_mle = pred_seq.N_t_array_mle
        N_t_array_cumsum_updated_mle = np.zeros(np.shape(N_t_array_mle))
        for i in range(np.array(N_t_array_mle).shape[0]):
            if i == 0:
                N_t_array_cumsum_updated_mle[i, :] = np.array(N_t_array_mle)[i, :] + 0.
            if i > 0:
                N_t_array_cumsum_updated_mle[i, :] = np.array(N_t_array_mle)[i, :] + Nobs_array_cumsum[i - 1]
    # plt.xlabel('time since %s in days'%data_obj.domain.time_origin)
    plt.xlabel('time in days')
    plt.ylabel('N')
    if ylim is not None:
        plt.ylim(ylim)

    print(Ksim)

    if NB_fit is None:
        return h1
    if NB_fit is not None:
        plt.clf()

        # estimate NB(n,p) params
        n_vec, p_vec = gpetas.some_fun.NB_n_p_methods_of_moments(np.array(pred_seq.N_t_array))
        n_vec_mle = None
        p_vec_mle = None
        if pred_seq.mle_obj is not None:
            n_vec_mle, p_vec_mle = gpetas.some_fun.NB_n_p_methods_of_moments(np.array(pred_seq.N_t_array_mle))

        # moments
        mean, var, skew, kurt = nbinom.stats(n=n_vec, p=p_vec, moments='mvsk')
        if pred_seq.mle_obj is not None:
            mean_mle, var_mle, skew_mle, kurt_mle = nbinom.stats(n=n_vec_mle, p=p_vec_mle, moments='mvsk')

        # quantiles
        q_up = nbinom.ppf(1. - quantile, n=n_vec, p=p_vec)
        q_down = nbinom.ppf(quantile, n=n_vec, p=p_vec)
        median = nbinom.ppf(0.5, n=n_vec, p=p_vec)
        if pred_seq.mle_obj is not None:
            q_up_mle = nbinom.ppf(1. - quantile, n=n_vec_mle, p=p_vec_mle)
            q_down_mle = nbinom.ppf(quantile, n=n_vec_mle, p=p_vec_mle)
            median_mle = nbinom.ppf(0.5, n=n_vec_mle, p=p_vec_mle)

        # plots
        if mle_only is None or gpetas_only is not None:
            plt.plot(t, np.mean(N_t_array, axis=1), 'k')
            plt.fill_between(t, y1=np.quantile(N_t_array, q=quantile, axis=1),
                             y2=np.quantile(N_t_array, q=1. - quantile, axis=1),
                             color='lightgrey',
                             label='$q_{%.2f,%.2f}$' % (quantile, 1 - quantile))

            plt.plot(t, mean, '--k', linewidth=1)
            plt.plot(t, q_up, ':k', linewidth=1)
            plt.plot(t, q_down, ':k', linewidth=1)

        if pred_seq.mle_obj is not None and gpetas_only is None:
            plt.plot(t, np.mean(pred_seq.N_t_array_mle, axis=1), '--b', linewidth=1)
            plt.plot(t, np.quantile(pred_seq.N_t_array_mle, q=quantile, axis=1), ':b', linewidth=1)
            plt.plot(t, np.quantile(pred_seq.N_t_array_mle, q=1. - quantile, axis=1), ':b', linewidth=1)
            if mle_only is not None:
                plt.fill_between(t, y1=np.quantile(pred_seq.N_t_array_mle, q=quantile, axis=1),
                                 y2=np.quantile(pred_seq.N_t_array_mle, q=1. - quantile, axis=1),
                                 color='lightblue',
                                 label='$q_{%.2f,%.2f}$' % (quantile, 1 - quantile))
            # NB
            plt.plot(t, mean_mle, '--', color='r', linewidth=1)
            plt.plot(t, q_up_mle, ':', color='r', linewidth=1)
            plt.plot(t, q_down_mle, ':', color='r', linewidth=1)

        # obs
        plt.plot(t, Nobs_array, '.m', markersize=markersize)
        plt.yscale(yscale)

        plt.xlabel('time in days')
        plt.ylabel('N')
        if ylim is not None:
            plt.ylim(ylim)

        return h1, n_vec, p_vec, n_vec_mle, p_vec_mle


# write report
def write_table_prediction_report(save_obj_pred, save_obj_pred_mle=None, m0_plot=None):
    # output dir
    init_outdir()
    # vars
    case_name = save_obj_pred['data_obj'].case_name
    time_origin = save_obj_pred['data_obj'].domain.time_origin
    if isinstance(time_origin, datetime.datetime):
        time_origin_obj = time_origin
    if isinstance(time_origin, str):
        time_origin_obj = datetime.datetime.strptime(time_origin, time_format)
    t0 = 0.
    t1, t2 = save_obj_pred['data_obj'].domain.T_borders_training
    t1, t3 = save_obj_pred['data_obj'].domain.T_borders_all
    time_end_training = time_origin_obj + datetime.timedelta(milliseconds=(t2) * 24. * 60. * 60. * 1000)
    time_end_data = time_origin_obj + datetime.timedelta(milliseconds=(t3) * 24. * 60. * 60. * 1000)
    X_borders = save_obj_pred['data_obj'].domain.X_borders
    tau0Htm, tau1, tau2 = save_obj_pred['tau_vec'][0]
    Ksim = len(save_obj_pred['pred_bgnew_and_offspring'])
    save_obj_GS = save_obj_pred['save_obj_GS']
    K, c, p, m_alpha, d, gamma, q, m_beta, m0 = save_obj_GS['setup_obj'].theta_start_Kcpadgqbm0
    n_start = gpetas.some_fun.n(m_alpha, m_beta, K, c, p, t_start=0.0, t_end=np.inf)
    stable_sampling = save_obj_GS['setup_obj'].stable_theta_sampling
    prior_dist_theta = save_obj_GS['setup_obj'].prior_theta_dist
    prior_params_theta = save_obj_GS['setup_obj'].prior_theta_params
    if m0_plot is None:
        m0_plot = save_obj_pred['data_obj'].domain.m0

    fid = open(output_dir_tables + '/report.tex', 'w')
    fid.write("\\documentclass[]{scrartcl}\n")
    fid.write("\\usepackage{amsmath}\n")
    fid.write("\\usepackage{graphicx}\n")
    fid.write("\n")
    fid.write("\n")
    fid.write("\n")

    fid.write("%opening\n")
    fid.write("\\title{GP-ETAS forecast report of case: \\texttt{%s}}\n" % case_name)
    fid.write("\\author{gpetas package}\n")

    fid.write("\\begin{document}\n")
    fid.write("\\maketitle\n")

    fid.write("\\newpage\n")
    fid.write("\\section{General info}\n")
    fid.write("Forecast period: $\\tau_1=%.1f$, $\\tau_2=%.1f$ with $\\tau_{H_t}$=-%.1f days.\n" % (
        tau1, tau2, tau0Htm - tau1))
    fid.write("\n\\noindent\n")
    fid.write("Forecast period: $|\mathcal{T^\\ast}|$=%.1f days.\n" % (tau2 - tau1))
    fid.write("\n\\noindent\n")
    fid.write("Training period: $t_1=%.1f$, $t_2=%.1f$ with $t_{H_t}=%.1f$ days.\n" % (t1, t2, t0))
    fid.write("\n\\noindent\n")
    fid.write("Total time window of data: $t_1=%.1f$, $t_3=%.1f$ with $t_{H_t}=%.1f$ and $t_3-t_2$=%.1f days.\n" % (
        t1, t3, t0, t3 - t2))
    fid.write("\n\\noindent\n")
    fid.write("Time origin is: %s\n" % (time_origin_obj).strftime(time_format))
    fid.write("\n\\noindent\n")
    fid.write("Time end of training: %s\n" % (time_end_training).strftime(time_format))
    fid.write("\n\\noindent\n")
    fid.write("Time end of data: %s\n" % (time_end_data).strftime(time_format))
    fid.write("\n\\noindent\n")
    fid.write("Spatial domain in $x_1$: [%.2f, %.2f]\n" % (X_borders[0, 0], X_borders[0, 1]))
    fid.write("\n\\noindent\n")
    fid.write("Spatial domain in $x_2$: [%.2f, %.2f]\n" % (X_borders[1, 0], X_borders[1, 1]))
    fid.write("\n\\noindent\n")
    fid.write("Spatial domain extent: $|\\mathcal{X}|$=%.1f\n" % (np.prod(np.diff(X_borders))))
    fid.write("\n\\noindent\n")
    fid.write("Magnitude distribution parameters: $\\beta_m$=%.3f and $m_0$=%.2f\n" % (m_beta, m0))
    fid.write("\n\\noindent\n")
    fid.write("Offspring sampling stable: %s\n" % (stable_sampling))
    fid.write("\n\\noindent\n")
    fid.write("Offspring, start values: $\\boldsymbol{\\theta_{{\\mathrm{start}}}}$=\n")
    fid.write("[$K$,$c$,$p$,$\\alpha_{m}$,$d$,$\gamma$,$q$]=[%.4f,%.4f,%.2f,%.2f,%.4f,%.2f,%.2f].\n" % (
        K, c, p, m_alpha, d, gamma, q))
    fid.write("\n\\noindent\n")
    fid.write("Inital branching ratio: $n_0$=%.2f.\n" % n_start)
    fid.write("\n\\noindent\n")
    fid.write("Prior distribution offspring: %s\n" % (prior_dist_theta))

    fid.write("\\newpage\n")
    fid.write("\\section{Setup}\n")
    save_obj_GS = save_obj_pred['save_obj_GS']
    hf = gpetas.plotting.plot_setting(save_obj_GS['data_obj'])
    hf[0].savefig(output_dir_figures + '/F_setup00.pdf', bbox_inches='tight')
    hf[1].savefig(output_dir_figures + '/F_setup01.pdf', bbox_inches='tight')
    hf[3].savefig(output_dir_figures + '/F_setup03.pdf', bbox_inches='tight')
    hf[4].savefig(output_dir_figures + '/F_setup04.pdf', bbox_inches='tight')
    fid.write("\\begin{figure}[h!]\n")
    fid.write("\\centering\n")
    fid.write("\\includegraphics[width=0.45\\textwidth]{../figures/F_setup00}\n")
    fid.write("\\includegraphics[width=0.425\\textwidth]{../figures/F_setup01}\n")
    fid.write("\\caption{All data.}\n")
    fid.write("\\end{figure}\n")
    fid.write("\\begin{figure}[h!]\n")
    fid.write("\\centering\n")
    fid.write("\\includegraphics[width=0.45\\textwidth]{../figures/F_setup03}\n")
    fid.write("\\includegraphics[width=0.45\\textwidth]{../figures/F_setup04}\n")
    fid.write("\\caption{Spatial distribution of training (left) and testing data (right).}\n")
    fid.write("\\end{figure}\n")

    fid.write("\\newpage\n")
    fid.write("\\section{Tables}\n")
    fid.write("\\begin{table}[h!]\n")
    fid.write("\centering\n")
    fid.write("\small\n")
    fid.write(
        "\caption{Forecasted number of events $N^\\ast$ for the entire region based on $K_{\\text{sim}}$=%i.}" % Ksim)
    fid.write("\n")
    fid.write("\\begin{tabular}{lccccccccc}")
    fid.write("\n")
    fid.write("\hline")
    fid.write("\n")
    fid.write("model &  $\\tau_2-\\tau_1$ & $N_{\\text{obs}}$ & mean & variance & 0.01 & 0.05 & 0.5 & 0.95 & 0.99 \\\ ")
    fid.write("\hline\n")

    t_vec = np.linspace(0., tau2 - tau1, 10)
    if (tau2 - tau1) >= 60:
        if sum(t_vec == 1.) == 0:
            t_vec = np.sort(np.append(t_vec, 1.))
        if sum(t_vec == 10.) == 0:
            t_vec = np.sort(np.append(t_vec, 10.))
        if sum(t_vec == 30.) == 0:
            t_vec = np.sort(np.append(t_vec, 30.))
        if sum(t_vec == 60.) == 0:
            t_vec = np.sort(np.append(t_vec, 60.))
    if (tau2 - tau1) >= 180:
        if sum(t_vec == 180.) == 0:
            t_vec = np.sort(np.append(t_vec, 180.))
    print(t_vec)

    for j in np.arange(1, len(t_vec)):
        t = t_vec[j]
        # gpetas
        N_t, Nobs = gpetas.prediction_2d.get_marginal_Nt_pred(t, save_obj_pred, m0_plot=m0_plot, which_events=None)
        m, v, q01, q05, q5, q95, q99 = [np.mean(N_t), np.var(N_t), np.quantile(N_t, q=0.01),
                                        np.quantile(N_t, q=0.05), np.quantile(N_t, q=0.5),
                                        np.quantile(N_t, q=0.95), np.quantile(N_t, q=0.99)]
        Line = "GP-E & %.1f & %.i & %.1f & %.1f & %.1f & %.1f & %.1f & %.1f  & %.1f \\\ \n" % (
            t, Nobs, m, v, q01, q05, q5, q95, q99)
        fid.write(Line)
        # mle
        N_t_mle, Nobs = gpetas.prediction_2d.get_marginal_Nt_pred(t, save_obj_pred_mle, m0_plot=m0_plot,
                                                                  which_events=None)
        m, v, q01, q05, q5, q95, q99 = [np.mean(N_t_mle), np.var(N_t_mle), np.quantile(N_t_mle, q=0.01),
                                        np.quantile(N_t_mle, q=0.05), np.quantile(N_t_mle, q=0.5),
                                        np.quantile(N_t_mle, q=0.95), np.quantile(N_t_mle, q=0.99)]
        Line = "E &  &  & %.1f & %.1f & %.1f & %.1f & %.1f & %.1f  & %.1f \\\ \n" % (m, v, q01, q05, q5, q95, q99)
        # Line = "E &  &  & %.0f & %.0f & %.0f & %.0f & %.0f & %.0f  & %.0f \\\ \n" %(m,v,q01,q05,q5,q95,q99)
        fid.write(Line)
        fid.write("\n")
        fid.write("\hline")
        fid.write("\n")

    # fid.write("\hline\n")
    fid.write("\end{tabular}\n")
    fid.write("\end{table}\n")

    # log10 table
    fid.write("\\begin{table}[h!]\n")
    fid.write("\centering\n")
    fid.write("\small\n")
    fid.write(
        "\caption{Forecasted logarithmic number of events $\\log_{10}N^\\ast$ for the entire region based on $K_{\\text{sim}}$=%i.}" % Ksim)
    fid.write("\n")
    fid.write("\\begin{tabular}{lccccccccc}")
    fid.write("\n")
    fid.write("\hline")
    fid.write("\n")
    fid.write("model &  $\\tau_2-\\tau_1$ & $N_{\\text{obs}}$ & mean & variance & 0.01 & 0.05 & 0.5 & 0.95 & 0.99 \\\ ")
    fid.write("\hline\n")

    for j in np.arange(1, len(t_vec)):
        t = t_vec[j]
        # gpetas
        N_t, Nobs = get_marginal_Nt_pred(t, save_obj_pred, m0_plot=m0_plot, which_events=None)
        z = []
        z = np.log10(N_t)
        Nobs = np.log10(Nobs)
        # np.nanmin(z[z != -np.inf])
        m, v, q01, q05, q5, q95, q99 = [np.nanmean(z[z != -np.inf]),
                                        np.nanvar(z[z != -np.inf]),
                                        np.nanquantile(z[z != -np.inf], q=0.01),
                                        np.nanquantile(z[z != -np.inf], q=0.05),
                                        np.nanquantile(z[z != -np.inf], q=0.5),
                                        np.nanquantile(z[z != -np.inf], q=0.95),
                                        np.nanquantile(z[z != -np.inf], q=0.99)]
        Line = "GP-E & %.1f & %.2f & %.2f & %.4f & %.2f & %.2f & %.2f & %.2f  & %.2f \\\ \n" % (
            t, Nobs, m, v, q01, q05, q5, q95, q99)
        fid.write(Line)
        # mle
        N_t_mle, Nobs = get_marginal_Nt_pred(t, save_obj_pred_mle, m0_plot=m0_plot, which_events=None)
        z = []
        z = np.log10(N_t_mle)
        Nobs = np.log10(Nobs)
        # np.nanmin(z[z != -np.inf])
        m, v, q01, q05, q5, q95, q99 = [np.nanmean(z[z != -np.inf]),
                                        np.nanvar(z[z != -np.inf]),
                                        np.nanquantile(z[z != -np.inf], q=0.01),
                                        np.nanquantile(z[z != -np.inf], q=0.05),
                                        np.nanquantile(z[z != -np.inf], q=0.5),
                                        np.nanquantile(z[z != -np.inf], q=0.95),
                                        np.nanquantile(z[z != -np.inf], q=0.99)]
        Line = "E &  &  & %.2f & %.4f & %.2f & %.2f & %.2f & %.2f  & %.2f \\\ \n" % (m, v, q01, q05, q5, q95, q99)
        # Line = "E &  &  & %.0f & %.0f & %.0f & %.0f & %.0f & %.0f  & %.0f \\\ \n" %(m,v,q01,q05,q5,q95,q99)
        fid.write(Line)
        fid.write("\n")
        fid.write("\hline")
        fid.write("\n")

    # fid.write("\hline\n")
    fid.write("\end{tabular}\n")
    fid.write("\end{table}\n")

    fid.write("\\newpage\n")
    fid.write("\\section{Figures}\n")

    fid.write("\\begin{figure}[h!]\n")
    fid.write("\\centering\n")
    for j in np.arange(1, len(t_vec)):
        fname = 'F003_pred_kernel_%s_%0i_%0i_%s_m%i' % (
            case_name, 0, j, 'linear', int(m0_plot * 10))
        fid.write("\\includegraphics[width=0.33\\textwidth]{../figures/%s}\n" % fname)
    fid.write("\\caption{Forecasted number of events $N^\\ast$ for different $\\tau$ in days.}\n")
    fid.write("\\end{figure}\n")

    fid.write("\\begin{figure}[h!]\n")
    fid.write("\\centering\n")
    for j in np.arange(1, len(t_vec)):
        fname = 'F003_pred_kernel_%s_%0i_%0i_%s_m%i' % (
            case_name, 1, j, 'log10', int(m0_plot * 10))
        fid.write("\\includegraphics[width=0.33\\textwidth]{../figures/%s}\n" % fname)
    fid.write("\\caption{Forecasted logarithmic number of events $\\log_{10}N^\\ast$ for different $\\tau$ in days.}\n")
    fid.write("\\end{figure}\n")

    fid.write("\\begin{figure}[h!]\n")
    fid.write("\\centering\n")
    for j in np.arange(1, len(t_vec)):
        fname = 'F002_pred_%s_%0i_%0i_%s_m%i' % (case_name, 1, j, 'log10', int(m0_plot * 10))
        fid.write("\\includegraphics[width=0.33\\textwidth]{../figures/%s}\n" % fname)
    fid.write(
        "\\caption{Histograms of forecasted logarithmic number of events $\\log_{10}N^\\ast$ for different $\\tau$ in days.}\n")
    fid.write("\\end{figure}\n")

    # quantiles
    fid.write("\\begin{figure}[h!]\n")
    fid.write("\\centering\n")
    for j in np.arange(1, len(t_vec)):
        fname = 'F004_pred_quantiles_%s_%0i_%0i_%s_m%i.pdf' % (
            case_name, 0, j, 'linear', int(m0_plot * 10))
        fid.write("\\includegraphics[width=0.33\\textwidth]{../figures/%s}\n" % fname)
    fid.write(
        "\\caption{Quantile plot of forecasted logarithmic number of events $\\log_{10}N^\\ast$ for different $\\tau$ in days.}\n")
    fid.write("\\end{figure}\n")

    # summary forecast uncertainty in time
    # '/F007_summary_unc_t_%s_%0i_%s_m%i.pdf' % (case_name, i, scales[i], int(m0_plot * 10)
    fid.write("\\begin{figure}[h!]\n")
    fid.write("\\centering\n")
    scales = ['linear', 'log10']
    for i in range(len(scales)):
        fname = '/F007_summary_unc_t_%s_%0i_%s_m%i.pdf' % (case_name, i, scales[i], int(m0_plot * 10))
        fid.write("\\includegraphics[width=0.45\\textwidth]{../figures/%s}\n" % fname)
    for i in range(len(scales)):
        fname = '/F007_summary_unc_t_q01_%s_%0i_%s_m%i.pdf' % (case_name, i, scales[i], int(m0_plot * 10))
        fid.write("\\includegraphics[width=0.45\\textwidth]{../figures/%s}\n" % fname)
    fid.write("\\caption{Forecaset in time, GP-ETAS (GP-E) and ETAS (E).}\n")
    fid.write("\\end{figure}\n")

    # prediction 2D
    fid.write("\\begin{figure}[h!]\n")
    fid.write("\\centering\n")
    fname = 'F006_pred_2D_%s_%s_m%i_gpetas' % (
        case_name, 'log10', int(m0_plot * 10))
    fid.write("\\includegraphics[width=0.75\\textwidth]{../figures/%s}\n" % fname)
    fname = 'F006_pred_2D_%s_%s_m%i_mle' % (
        case_name, 'log10', int(m0_plot * 10))
    fid.write("\\includegraphics[width=0.75\\textwidth]{../figures/%s}\n" % fname)
    fid.write("\\caption{Spatial forecast, GP-ETAS (left) and ETAS (right).}\n")
    fid.write("\\end{figure}\n")

    scales = ['linear', 'logy', 'logx', 'loglog']
    fid.write("\\begin{figure}[h!]\n")
    fid.write("\\centering\n")
    for i in range(len(scales)):
        scale = scales[i]
        fname = '/F001_pred_%s_%0i_%s' % (case_name, i, scales[i])
        fid.write("\\includegraphics[width=0.75\\textwidth]{../figures/%s}\n" % fname)
    fid.write("\\caption{Forecasts, GP-ETAS and ETAS over time with $K_{sim}$=%i.}\n" % Ksim)
    fid.write("\\end{figure}\n")

    fid.write("\\end{document}")
    fid.close()

    return
