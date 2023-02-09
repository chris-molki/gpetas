import numpy as np
from scipy.linalg import solve_triangular
import scipy as sc
import time
from gpetas.utils.some_fun import get_grid_data_for_a_point
import os
import pickle
import datetime

# some globals
time_format = "%Y-%m-%d %H:%M:%S.%f"
output_dir = "output_pred"
output_dir_tables = "output_pred/tables"
output_dir_figures = "output_pred/figures"
output_dir_data = "output_pred/data"


class setup_pred():
    def __init__(self, save_obj_GS=None, tau1=None, tau2=None, tau0_Ht=None,
                 Ksim=None, sample_idx_vec=None,
                 mle_obj=None, mle_obj_silverman=None, epsilon_after_mainshock=1e-4):
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
                xnew = np.vstack((xnew, np.random.exponential(1. / m_beta, np.size(xnew)) + m0)).transpose()

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
                xnew = np.vstack((xnew, np.random.exponential(1. / m_beta, np.size(xnew)) + m0)).transpose()

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
    def __init__(self, mle_obj, tau1, tau2, tau0_Ht=0., Ksim=None, data_obj=None, seed=None, Bayesian_m_beta=None):
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
        self.m_beta = mle_obj.m_beta_lower_T2
        if data_obj is None:
            data_obj = mle_obj.data_obj

        init_save_dictionary(self)
        self.save_pred['N495_true'] = np.copy(self.N495_true)
        self.save_pred['Nm0_true'] = np.copy(self.Nm0_true)
        self.save_pred['seed_orig'] = np.copy(seed)
        self.save_pred['seed'] = np.copy(self.seed)
        self.save_pred['data_obj'] = data_obj
        self.save_pred['m0'] = np.copy(self.data_obj.domain.m0)

        # some params
        mu_max_bg_mle = 10. * np.max(mle_obj.mu_grid)
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

            if np.max(mu_unthinned) > mu_max_bg_mle:
                raise ValueError("Upper bound mu_max is too low.")
            # thinning
            thinned_idx = np.where(mu_max_bg_mle * np.random.rand(len(X_unthinned)) <= mu_unthinned)[0]
            X_thinned = X_unthinned[thinned_idx]

            # writing to a matrix
            bg_events = np.zeros([int(len(X_thinned[:, 0])), 5])
            bg_events[:, 0] = np.random.rand(len(X_thinned[:, 0])) * (tau2 - tau1)
            bg_events[:, 1] = np.random.exponential(1. / self.m_beta,
                                                    len(X_thinned[:, 0])) + self.m0  # m_i: marks
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
                 randomized_samples='yes', Bayesian_m_beta=None):
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
                print('Fixed samples:randomized_samples is None.')
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
                bg_events[:, 1] = np.random.exponential(1. / self.m_beta, len(X_thinned[:, 0])) + self.m0  # m_i: marks
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
            Ht_pred, Ht_pred_aug = sim_offspring_from_Ht(self, theta_Kcpadgq=theta,
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
                                          tau1=self.tau1, tau2=self.tau2, m0=self.m0, which_events='all')
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
    :param save_obj_pred:
    :type save_obj_pred: dictionary
    :param tau1:
    :type tau1: float
    :param tau2:
    :type tau2: float
    :param m0:
    :type m0:float
    :param which_events:
    :type which_events:any
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

    if which_events is not None:
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

    if which_events == 2:
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

    return out
