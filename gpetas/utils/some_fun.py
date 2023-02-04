import numpy as np
import os
import pickle
import scipy as sc
from scipy.interpolate import griddata
from scipy.linalg import solve_triangular
from scipy.stats import norm

import gpetas

# some globals
time_format = "%Y-%m-%d %H:%M:%S.%f"
output_dir = "output"
output_dir_tables = "output/tables"
output_dir_figures = "output/figures"
output_dir_data = "output/data"


def init_outdir():
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    if not os.path.isdir(output_dir_tables):
        os.mkdir(output_dir_tables)
    if not os.path.isdir(output_dir_figures):
        os.mkdir(output_dir_figures)
    if not os.path.isdir(output_dir_data):
        os.mkdir(output_dir_data)


def eval_n(save_obj_GS=None, mle_obj=None, t_start=0.0, t_end=np.inf):
    """
    :param save_obj_GS:
    :type save_obj_GS:
    :param mle_obj:
    :type mle_obj:
    :param t_start:
    :type t_start:
    :param t_end:
    :type t_end:
    :return:
        n_vec, n_start, n_vec_absT if save_obj_GS is not None;
        n_mle, n_start, n_absT if mle_obj is not None;
            here absT stands for the integration limit of the kernel int_0^|T_training| instead of int_0^inf
    :rtype:
    """
    if save_obj_GS is not None:
        Ksamples = len(save_obj_GS['lambda_bar'])
        n_vec = np.zeros(Ksamples) * np.nan
        n_vec_absT = np.zeros(Ksamples) * np.nan
        absT = np.diff(save_obj_GS['data_obj'].domain.T_borders_training)
        m_beta = save_obj_GS['setup_obj'].m_beta
        for i in range(Ksamples):
            K, c, p, m_alpha, d, gamma, q = save_obj_GS['theta_tilde'][i]
            n_vec[i] = gpetas.some_fun.n(m_alpha=m_alpha, m_beta=m_beta, K=K, c=c, p=p, t_start=t_start, t_end=t_end)
            n_vec_absT[i] = gpetas.some_fun.n(m_alpha=m_alpha, m_beta=m_beta, K=K, c=c, p=p, t_start=t_start,
                                              t_end=absT)
        K, c, p, m_alpha, d, gamma, q = save_obj_GS['theta_start'][:7]
        n_start = gpetas.some_fun.n(m_alpha=m_alpha, m_beta=m_beta, K=K, c=c, p=p, t_start=t_start, t_end=t_end)
        return n_vec, n_start, n_vec_absT
    if mle_obj is not None:
        m_beta = mle_obj.setup_obj.m_beta
        absT = mle_obj.absT_training
        K, c, p, m_alpha, d, gamma, q = mle_obj.theta_mle_Kcpadgq
        n_mle = gpetas.some_fun.n(m_alpha=m_alpha, m_beta=m_beta, K=K, c=c, p=p, t_start=t_start, t_end=t_end)
        K, c, p, m_alpha, d, gamma, q = mle_obj.theta_start_Kcpadgq
        n_start = gpetas.some_fun.n(m_alpha=m_alpha, m_beta=m_beta, K=K, c=c, p=p, t_start=t_start, t_end=t_end)
        K, c, p, m_alpha, d, gamma, q = mle_obj.theta_mle_Kcpadgq
        n_absT = gpetas.some_fun.n(m_alpha=m_alpha, m_beta=m_beta, K=K, c=c, p=p, t_start=t_start, t_end=absT)
        return n_mle, n_start, n_absT


### new for plotting issues
def integral_offspring(save_obj_GS, T1, T2, sample_idx_vec=None, H_T1=None, mle_obj=None):
    if mle_obj is not None:
        if H_T1 is None:
            t_vec_all = mle_obj.data_obj.data_all.times.reshape(-1, 1)  # (Nall,1)
            m_vec_all = mle_obj.data_obj.data_all.magnitudes.reshape(-1, 1)  # (Nall,1)
        else:
            t_vec_all = H_T1.times.reshape(-1, 1)  # (Nall,1)
            m_vec_all = H_T1.magnitudes.reshape(-1, 1)  # (Nall,1)
        spatial_kernel = mle_obj.setup_obj.spatial_offspring
        m0 = mle_obj.setup_obj.m0
        theta_phi__Kcpadgq = mle_obj.theta_mle_Kcpadgq
        # theta
        if spatial_kernel == 'R':
            K, c, p, m_alpha, D, gamma, q = theta_phi__Kcpadgq
        if spatial_kernel == 'P':
            K, c, p, m_alpha, D, gamma, q = theta_phi__Kcpadgq
        if spatial_kernel == 'G':
            K, c, p, m_alpha, D = theta_phi__Kcpadgq[0:5]

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
        integral_offspring_all = np.sum(int_offspring)
        ### END WARNING ####

    if save_obj_GS is not None:
        if H_T1 is None:
            t_vec_all = save_obj_GS['data_obj'].data_all.times.reshape(-1, 1)  # (Nall,1)
            m_vec_all = save_obj_GS['data_obj'].data_all.magnitudes.reshape(-1, 1)  # (Nall,1)
        else:
            t_vec_all = H_T1.times.reshape(-1, 1)  # (Nall,1)
            m_vec_all = H_T1.magnitudes.reshape(-1, 1)  # (Nall,1)
        spatial_kernel = save_obj_GS['setup_obj'].spatial_offspring
        m0 = save_obj_GS['setup_obj'].m0

        # if sample_idx_vec is None:
        #    Ksamples = len(save_obj_GS['data_obj']

        if sample_idx_vec is None:
            # theta_phi__Kcpadgq = save_obj_GS['theta_tilde'][0]
            Ksamples = len(save_obj_GS['lambda_bar'])
            sample_idx_vec = np.arange(0, Ksamples)

        integral_offspring_all = np.zeros(len(sample_idx_vec))

        for k in range(len(sample_idx_vec)):
            idx = sample_idx_vec[k]
            theta_phi__Kcpadgq = save_obj_GS['theta_tilde'][idx]

            # theta
            if spatial_kernel == 'R':
                K, c, p, m_alpha, D, gamma, q = theta_phi__Kcpadgq
            if spatial_kernel == 'P':
                K, c, p, m_alpha, D, gamma, q = theta_phi__Kcpadgq
            if spatial_kernel == 'G':
                K, c, p, m_alpha, D = theta_phi__Kcpadgq[0:5]

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
            integral_offspring_all[k] = np.sum(int_offspring)
            ### END WARNING ####

    return integral_offspring_all


def Lambda_t_1D(t_end, tau1=0., save_obj_GS=None, sample_idx_vec=None, mle_obj=None, resolution=None):
    if resolution is None:
        resolution = 100

    if mle_obj is not None:
        # idx = np.logical_and(mle_obj.data_obj.data_all.times>=tau1,mle_obj.data_obj.data_all.times<=t_end)
        # t_eval_vec = np.sort(np.concatenate([[tau1],mle_obj.data_obj.data_all.times[idx],np.arange(tau1,t_end,int(t_end-tau1)/20)[1:-1],[t_end]]))
        # if mle_obj.data_obj.data_all.times[idx][0] == tau1:
        #    t_eval_vec = t_eval_vec[1:]
        # if mle_obj.data_obj.data_all.times[idx][-1] == t_end:
        #    t_eval_vec = t_eval_vec[:-1]
        t_eval_vec = np.linspace(tau1, t_end, resolution)
        LAM = np.zeros(len(t_eval_vec)) * np.nan
        for i in range(len(t_eval_vec)):
            t = t_eval_vec[i]
            LAM[i] = mle_obj.integral_mu_unit_time_mle * (t - tau1) + integral_offspring(save_obj_GS=None, T1=tau1,
                                                                                         T2=t, sample_idx_vec=None,
                                                                                         H_T1=None, mle_obj=mle_obj)
        return LAM, t_eval_vec
    if save_obj_GS is not None:
        # idx = np.logical_and(save_obj_GS['data_obj'].data_all.times>=tau1,save_obj_GS['data_obj'].data_all.times<=t_end)
        # t_eval_vec = np.sort(np.concatenate([[tau1],save_obj_GS['data_obj'].data_all.times[idx],np.arange(tau1,t_end,int(t_end-tau1)/20)[1:-1],[t_end]]))
        t_eval_vec = np.linspace(tau1, t_end, resolution)
        if sample_idx_vec is None:
            Ksamples = len(save_obj_GS['lambda_bar'])
        else:
            Ksamples = len(sample_idx_vec)
        LAM = np.zeros([len(t_eval_vec), Ksamples]) * np.nan
        for i in range(len(t_eval_vec)):
            t = t_eval_vec[i]
            integral_bg_ut = np.sum(save_obj_GS['mu_grid'], axis=1) * np.prod(
                np.diff(save_obj_GS['data_obj'].domain.X_borders)) / len(save_obj_GS['mu_grid'][0])
            LAM[i, :] = integral_bg_ut * (t - tau1) + integral_offspring(save_obj_GS=save_obj_GS, T1=tau1, T2=t,
                                                                         sample_idx_vec=sample_idx_vec)
        return LAM, t_eval_vec


### new

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


# grid interpolation
def mu_posterior_grid(save_obj_GS, bins=None, sample_idx_vec=None, method=None):
    # basic functions
    inv_sigmoid = lambda sigma: np.log(sigma / (1. - sigma))
    sigmoid = lambda x: 1. / (1 + np.exp(-x))
    data_obj = save_obj_GS['data_obj']
    X_grid = save_obj_GS['X_grid']
    if bins is not None:
        xprime = gpetas.some_fun.make_X_grid(X_borders=data_obj.domain.X_borders, nbins=bins)
    method_integral = 'Riemann_sum'
    absX = np.prod(np.diff(data_obj.domain.X_borders))

    if bins is None:
        bins = int(np.sqrt(save_obj_GS['X_grid'].shape[0]))
        mu_xprime_grid = np.array(save_obj_GS['mu_grid'])
        return mu_xprime_grid, X_grid
    if sample_idx_vec is None:
        Ksamples = len(save_obj_GS['lambda_bar'])
        sample_idx_vec = np.arange(0, Ksamples)
    if method is None:
        method = 'interpol_cubic'

    mu_xprime_grid = np.empty([bins * bins, len(sample_idx_vec)])
    for k in range(len(sample_idx_vec)):
        mu_grid = np.array(save_obj_GS['mu_grid'][k])
        # integral mu_x
        if method_integral == 'Riemann_sum':
            L = len(np.array(save_obj_GS['mu_grid'][0]))
            arr_integral_mu_x_unit_time = absX / L * np.sum(mu_grid)

        if method == 'interpol_cubic':
            mu_xprime_grid[:, k] = griddata(points=X_grid, values=mu_grid.reshape(-1), xi=xprime, method='cubic',
                                            fill_value=np.nan, rescale=False)

        if method == 'sparse_mean':
            lambda_bar = save_obj_GS['lambda_bar'][k]
            cov_params = save_obj_GS['cov_params'][k]
            f = inv_sigmoid(mu_grid / lambda_bar)
            x = np.copy(X_grid)
            k_ffprime = cov_func(x, xprime, cov_params=cov_params)  # K_ffprime
            K_ff = cov_func(x, x, cov_params=cov_params)
            K_ff_inv = inverse(K_ff)
            fprime_given_f_mean_approx = np.dot(k_ffprime.T, K_ff_inv.dot(f))
            mu_xprime_grid[:, k] = lambda_bar * sigmoid(fprime_given_f_mean_approx)

    return mu_xprime_grid, xprime


def get_grid_count_forecast(pred_obj, bins=None):
    '''
    Obtains rates per unit time and unit area,
        i.e., events per grid cell are counted for the
        prediction time-space window using 2D histogram function
    :param pred_obj: prediction object = forecast
    :type pred_obj: python class
    :param bins: number of bins per dimension
    :type bins: integer
    :return: 2d times Ksim array with rates per unit time and unit area
    :rtype: numpy array float
    '''
    in_list = pred_obj.save_pred['pred_bgnew_and_offspring_with_Ht_offspring']
    data_obj = pred_obj.save_pred['data_obj']
    if bins is None:
        bins = 50  # per dim
    Ksim = len(in_list)
    tau0_Ht, tau1, tau2 = pred_obj.tau_vec
    abs_T_testing = tau2 - tau1
    abs_X = np.prod(np.diff(data_obj.domain.X_borders))
    H2d = np.empty([bins, bins, Ksim]) * np.nan
    X_grid = gpetas.utils.some_fun.make_X_grid(X_borders=data_obj.domain.X_borders, nbins=bins)
    for k in range(Ksim):
        xi_k = in_list[k][:, 2]
        yi_k = in_list[k][:, 3]
        H2d[:, :, k], xed, yed = np.histogram2d(xi_k, yi_k, range=data_obj.domain.X_borders, bins=bins)
    H2d_GPetas_utua = np.copy(H2d) / abs_T_testing / (abs_X / bins ** 2.)
    return H2d_GPetas_utua, X_grid, bins


def get_grid_count_data(pred_obj, bins):
    data_obj = pred_obj.save_pred['data_obj']
    if bins is None:
        bins = 50  # per dim
    tau0_Ht, tau1, tau2 = pred_obj.tau_vec
    abs_T_testing = tau2 - tau1
    abs_X = np.prod(np.diff(data_obj.domain.X_borders))
    idx = np.logical_and(data_obj.data_all.times > tau1, data_obj.data_all.times <= tau2)
    data_cut = np.empty([np.sum(idx), 4])
    data_cut = np.empty([np.sum(idx), 4])
    data_cut[:, 0] = data_obj.data_all.times[idx]
    data_cut[:, 1] = data_obj.data_all.magnitudes[idx]
    data_cut[:, 2] = data_obj.data_all.positions[idx, 0]
    data_cut[:, 3] = data_obj.data_all.positions[idx, 1]
    xi_k = data_obj.data_all.positions[idx, 0]
    yi_k = data_obj.data_all.positions[idx, 1]
    H2d, xed, yed = np.histogram2d(xi_k, yi_k, range=data_obj.domain.X_borders, bins=bins)
    X_grid = gpetas.utils.some_fun.make_X_grid(X_borders=data_obj.domain.X_borders, nbins=bins)
    H2d_GPetas_utua = np.copy(H2d) / abs_T_testing / (abs_X / bins ** 2.)
    return H2d_GPetas_utua, X_grid, bins, data_cut


def cumsum_events(tau1, tau2, data_obj, m0=None):
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
    y = np.hstack([N_tau1, np.cumsum(np.ones(N_in_tau1_tau2)) + N_tau1]) - N_tau1
    x = np.hstack([tau1, data_obj.data_all.times[idx_test]]) - tau1
    return x, y


def cumsum_events_pred(tau1, tau2, pred_obj, data_obj, m0=None):
    N_tau1 = np.sum(data_obj.data_all.times <= tau1)
    if m0 is not None:
        N_tau1 = np.sum(np.logical_and(data_obj.data_all.times <= tau1, data_obj.data_all.magnitudes >= m0))
    Ksim = len(pred_obj.save_pred['pred_bgnew_and_offspring_with_Ht_offspring'])
    out = {'y': [],
           'x': [],
           'Ksim': Ksim}
    for i in range(Ksim):
        pred_data = pred_obj.save_pred['pred_bgnew_and_offspring_with_Ht_offspring'][i]
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

    return out


### ok
def autocorr(x, alpha=None):
    r = np.correlate(x - np.mean(x), x - np.mean(x), mode='full') / np.var(x) / len(x)
    if alpha is not None:
        u_alpha = norm.ppf(alpha / 2)
        uncorr_treshold = u_alpha / np.sqrt(len(x))
        return r[len(r) // 2:], uncorr_treshold
    else:
        return r[len(r) // 2:]


def autocorr2(x):
    r = np.fft.ifft(np.abs(np.fft.fft((x - np.mean(x)), n=2 ** int(np.ceil(np.log2(2 * len(x)))))) ** 2).real * 1. / (
        len(x))
    r_normed = r / np.var(x)
    return r_normed[:int(len(x) / 2)]


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


class create_data_obj_from_cat_file():
    """
    Reads data from a typical catalog file with format: idx, x_lon, y_lat, mag, time
    no header, time decimal in days
    with N data points the data file has dimension (N,5)
    e.g.,
    1	2.7523	2.1859	4.66	0.000000
    2	2.7447	2.0783	3.43	0.004436
    3	3.0142	2.1734	3.41	0.320997
    4	3.4610	1.5132	3.54	2.271304
    5	2.8646	2.2552	3.43	11.108739
    6	2.8682	3.0405	4.19	41.841438
    7	2.8948	3.0755	4.94	41.963901
    8	2.6674	3.0231	3.49	42.026355
    ...
    """

    def __init__(self, fname, X_borders=None, T_borders_all=None, T_borders_training=None, utm_yes=None,
                 T_borders_test=None, m0=None, outdir=None, case_name='case_01', time_origin=None,
                 domain_obj=None):
        """
        :param fname:
        :type fname:
        :param X_borders:
        :type X_borders:
        :param T_borders_all:
        :type T_borders_all:
        :param T_borders_training:
        :type T_borders_training:
        :param utm_yes:
        :type utm_yes:
        :param T_borders_test:
        :type T_borders_test:
        :param m0:
        :type m0:
        :param outdir:
        :type outdir:
        :param case_name:
        :type case_name:
        :param time_origin:
        :type time_origin:
        :param domain_obj:
        :type domain_obj:
        """
        init_outdir()
        outdir = output_dir + '/inference_results'

        # domain_obj
        if domain_obj is not None:
            X_borders = domain_obj.X_borders
            T_borders_all = domain_obj.T_borders_all
            T_borders_training = domain_obj.T_borders_training
            T_borders_test = domain_obj.T_borders_testing
            m0 = domain_obj.m0
            time_origin = domain_obj.time_origin

        # init
        self.fname = fname
        self.case_name = case_name
        self.domain = dom()
        self.domain.time_origin = time_origin
        self.data_all = obs()
        self.utm_yes = utm_yes

        # reading
        aux = np.loadtxt(fname)
        self.data_all.times = aux[:, 4]  # time
        self.data_all.magnitudes = aux[:, 3]  # magnitude
        self.data_all.positions = aux[:, 1:3]  # [x_lon,y_lat]
        np.savetxt(output_dir_data+'/cat_%s.dat'%self.case_name,aux)

        # set domain
        if X_borders is not None:
            self.domain.X_borders = X_borders
            self.domain.X_borders_original = X_borders
        else:
            X = self.data_all.positions
            self.domain.X_borders = np.array([[np.min(X[:, 0]), np.max(X[:, 0])], [np.min(X[:, 1]), np.max(X[:, 1])]])
            self.domain.X_borders_original = np.copy(self.domain.X_borders)
        if T_borders_all is not None:
            self.domain.T_borders_all = T_borders_all
        else:
            self.domain.T_borders_all = np.array([0, np.max(self.data_all.times)])
        if T_borders_training is None:
            self.domain.T_borders_training = np.copy(self.domain.T_borders_all)
        else:
            self.domain.T_borders_training = T_borders_training
        self.domain.T_borders_testing = T_borders_test
        if utm_yes == 1:
            self.utm_conversion()

        # mark domain: cut off magnitude
        if m0 is None:
            m0 = np.min(self.data_all.magnitudes)
        self.domain.m0 = m0
        # new: cut data set
        idx_m0 = aux[:, 3] >= m0
        self.data_all.times = aux[idx_m0, 4]  # time
        self.data_all.magnitudes = aux[idx_m0, 3]  # magnitude
        self.data_all.positions = aux[idx_m0, 1:3]  # [x_lon,y_lat]
        # end new
        if np.min(self.data_all.magnitudes) < self.domain.m0:
            # self.domain.m0 = np.min(self.data_all.magnitudes)
            # print('Warning: m0 input is to high; m0 is set to the minimum in the data, which is =', m0)
            print('Warning: m0 input might be to high; minimum magnitude in the data is =',
                  np.min(self.data_all.magnitudes))

        self.idx_training = np.where((self.data_all.times >= self.domain.T_borders_training[0]) & (
                self.data_all.times <= self.domain.T_borders_training[1]))
        # Warning for duplicates (same event times)
        if np.sum(np.diff(self.data_all.times) == 0) > 0:
            print('WARNING:', np.sum(np.diff(self.data_all.times) == 0), ' Duplicate(s) in the data set.')

        # output directory
        if outdir is None:
            outdir = './inference_results'
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        # write to file
        fname_data_obj = outdir + "/data_obj_%s.all" % (case_name)
        file = open(fname_data_obj, "wb")  # remember to open the file in binary mode
        pickle.dump(self, file)
        file.close()
        print('____________________________________________________')
        print('data_obj has been created and saved in:', fname_data_obj)

    def utm_conversion(self):
        """
        Converts coordinates in degree into coordinates in UTM
        Requires utm packages
        :return: no return type, directly alters the self object
        """
        import utm  # has be installed before
        utm_zone_number = utm.from_latlon(self.domain.X_borders[1, 0], self.domain.X_borders[0, 0])[2]
        utm_lat_argmin = -180 + (utm_zone_number - 1) * 6 + 3
        utm_lat_argmax = -180 + (utm_zone_number - 1) * 6
        self.utm_zone_number = utm_zone_number
        self.utm_zone_letter = utm.from_latlon(self.domain.X_borders[1, 0], self.domain.X_borders[0, 0])[3]
        xmin = utm.from_latlon(self.domain.X_borders[1, 0], self.domain.X_borders[0, 0])[0]
        xmax = utm.from_latlon(self.domain.X_borders[1, 0], self.domain.X_borders[0, 1])[0]
        ymin = utm.from_latlon(self.domain.X_borders[1, 0], utm_lat_argmin)[1]
        ymax = utm.from_latlon(self.domain.X_borders[1, 1], utm_lat_argmax)[1]
        self.domain.X_borders_UTM_km = np.array([[xmin, xmax], [ymin, ymax]]) / 1000.
        # adjusting utm domain in Northing
        if np.min(self.data_all.positions[:, 1]) == self.domain.X_borders[1, 0]:
            ymin = np.min(self.data_all.positions_UTM_km[:, 1])
            self.domain.X_borders_UTM_km[1, 0] = ymin
        if np.max(self.data_all.positions[:, 1]) == self.domain.X_borders[1, 1]:
            ymax = np.max(self.data_all.positions_UTM_km[:, 1])
            self.domain.X_borders_UTM_km[1, 1] = ymax
        self.data_all.positions_UTM_km = np.squeeze(
            utm.from_latlon(self.data_all.positions[:, 1], self.data_all.positions[:, 0])[:2]).T / 1000.


def silverman_scott_rule_d(X_data, individual_yes=None):
    """
    Silvermans rule: (4.15) page 86--87: h* = mean(sigma_ii)* N**(-1/(d+4))
     = mean_std * N^(-0.1666...)
     e.g., used for minimum bandwidth in classical kde ETAS
    :param X_data:
    :type X_data:
    :param individual_yes:
    :type individual_yes:
    :return:
    :rtype:
    """
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


def mu_xprime_gpetas(xprime, mu_grid, X_grid, X_borders, method=None, lambda_bar=None, cov_params=None):
    # basic functions
    inv_sigmoid = lambda sigma: np.log(sigma / (1. - sigma))
    sigmoid = lambda x: 1. / (1 + np.exp(-x))

    # evaluation depending on the method
    mu_xprime = None
    mu_grid = mu_grid.reshape(-1)

    # methods
    if method is None:
        method = 'nearest'
    if method == 'sparse':
        f = inv_sigmoid(mu_grid / lambda_bar)
        fprime = sample_from_cond_gp(xprime, f, X_grid, cov_params=cov_params)
        mu_xprime = lambda_bar * sigmoid(fprime)

    elif method == 'sparse_mean':
        f = inv_sigmoid(mu_grid / lambda_bar)
        x = np.copy(X_grid)
        k = cov_func(x, xprime, cov_params=cov_params)  # K_ffprime
        K_ff = cov_func(x, x, cov_params=cov_params)
        K_ff_inv = inverse(K_ff)
        fprime_given_f_mean_approx = np.dot(k.T, K_ff_inv.dot(f))
        mu_xprime = lambda_bar * sigmoid(fprime_given_f_mean_approx)

    elif method == 'sparse_mean_noise':
        f = inv_sigmoid(mu_grid / lambda_bar)
        x = np.copy(X_grid)
        k = cov_func(x, xprime, cov_params=cov_params)  # K_ffprime
        K_ff = cov_func(x, x, cov_params=cov_params)
        K_ff_inv = inverse(K_ff)
        fprime_given_f_mean_approx = np.dot(k.T, K_ff_inv.dot(f)) + \
                                     np.random.normal(loc=0, scale=cov_params[0], size=len(xprime))
        mu_xprime = lambda_bar * sigmoid(fprime_given_f_mean_approx)
    else:
        mu_xprime = get_grid_data_for_a_point(mu_grid.flatten(), xprime, X_borders=X_borders, method=method)

    '''


    # methods
    if method == 'grid_approx':
        mu_xprime = get_grid_data_for_a_point(mu_grid.flatten(), xprime, X_borders=X_borders)
    if method == 'interpol_linear':
        mu_xprime = griddata(points=X_grid, values=mu_grid.reshape(-1),
                             xi=xprime, method='linear', fill_value=np.nan, rescale=False)
    if method == 'interpol_cubic':
        mu_xprime = griddata(points=X_grid, values=mu_grid.reshape(-1),
                             xi=xprime, method='cubic', fill_value=np.nan, rescale=False)
    if method == 'interpol_nearest':
        mu_xprime = griddata(points=X_grid, values=mu_grid.reshape(-1),
                             xi=xprime, method='nearest', fill_value=np.nan, rescale=False)
    '''

    return mu_xprime


def sample_from_cond_gp(xprime, f, x, cov_params, noise=1e-4):
    """
    Samples GP conditioned on GP at (grid) positions.
    :param self:
    :param xprime:
    :return: fprime
    """
    k = cov_func(x, xprime, cov_params=cov_params)  # K_ffprime
    K_ff = cov_func(x, x, cov_params=cov_params)
    K_ff_inv = inverse(K_ff)
    mean = k.T.dot(K_ff_inv.dot(f))
    kprimeprime = cov_func(xprime, xprime, cov_params=cov_params)
    # var = (kprimeprime - k.T.dot(self.K_inv.dot(k))).diagonal()
    # gprime = mean + numpy.sqrt(var)*numpy.random.randn(xprime.shape[0])
    Sigma = (kprimeprime - k.T.dot(K_ff_inv.dot(k)))
    L = np.linalg.cholesky(Sigma + noise * np.eye(Sigma.shape[0]))
    fprime = mean + np.dot(L.T, np.random.randn(xprime.shape[0]))
    return fprime


def inverse(K, noise=1e-4):
    """
    Computes inverse of K using
        cholesky to get L
        solve triangular system based on L
    :param K:
    :param noise: jitter to enable inversion, often = 1e-5
    :return: K_inv, inverse of K
    """
    K = K + noise * np.eye(K.shape[0])
    L = np.linalg.cholesky(K)
    L_inv = solve_triangular(L, np.eye(L.shape[0]), lower=True, check_finite=False)
    K_inv = L_inv.T.dot(L_inv)
    return K_inv


def cov_func(x, x_prime, only_diagonal=False, cov_params=None):
    """ Computes the covariance functions between x and x_prime.

    :param x: numpy.ndarray [num_points x D]
        Contains coordinates for points of x
    :param x_prime: numpy.ndarray [num_points_prime x D]
        Contains coordinates for points of x_prime
    :param only_diagonal: bool
        If true only diagonal is computed (Works only if x and x_prime
        are the same, Default=False)
    :param cov_params: hypers of the cov function RBF,e.g.,cov_params=[1, np.array([.1,.1])]

    :return: numpy.ndarray [num_points x num_points_prime]
        Kernel matrix.
    """
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


def get_grid_data_for_a_point(grid_values_flattened, points_xy, X_borders=None, method=None):
    """
    get values for points (x,y) from grid data with shape (bins**2,)
    :param grid_values_flattened: float: array (bins**2,)
        Grid values of a bins times bins grid with bins**2 elements collapsed to one dimension (flattened).
    :param grid_values_flattened:
    :type grid_values_flattened:
    :param points_xy:
    :type points_xy:
    :param X_borders:
    :type X_borders:
    :param method:
    :type method:
    :return:
    :rtype:
    """

    # interpolation method
    if method is None:
        method = 'nearest'

    # domain and gridding
    assert X_borders is not None, "Please specify a spatial domain in terms of X_borders for each dimension, e.g. X_borders = np.array([[-118.,-116.],[34.,36.]])"
    nbins = int(np.sqrt(len(grid_values_flattened)))
    X_grid = gpetas.some_fun.make_X_grid(X_borders=X_borders, nbins=nbins)

    # setup interpolator object
    points = (np.unique(X_grid[:, 0]), np.unique(X_grid[:, 1]))
    values = grid_values_flattened.reshape([nbins, nbins]).T
    get_grid_value = sc.interpolate.RegularGridInterpolator(points, values, bounds_error=True, fill_value=np.nan)

    return get_grid_value(points_xy, method=method)

    '''
    # domain 
    assert X_borders is not None, "Please specify a spatial domain in terms of X_borders for each dimension, e.g. X_borders = np.array([[-118.,-116.],[34.,36.]])"
    X_range = np.diff(X_borders)

    
    # xy data 
    points_xy = points_xy.reshape(-1, 2)
    x_vec = points_xy[:, 0] - X_borders[0, 0]
    y_vec = points_xy[:, 1] - X_borders[1, 0]

    # the grid 
    n_cells = len(grid_values_flattened)
    NperDim = np.int(np.sqrt(n_cells))
    grid_ij = np.reshape(grid_values_flattened, [NperDim, NperDim])  # [z,y,x]
    # jx_vec = (np.ceil(x_vec * NperDim / np.sqrt(R)) - 1).astype(int)
    # iy_vec = (np.ceil(y_vec * NperDim / np.sqrt(R)) - 1).astype(int)
    # old version:
    # jx_vec = (np.ceil(x_vec * NperDim / X_range[0]) - 1).astype(int)
    # iy_vec = (np.ceil(y_vec * NperDim / X_range[1]) - 1).astype(int)
    # new: adjusted
    jx_vec = (np.ceil(x_vec * NperDim / X_range[0])).astype(int)
    iy_vec = (np.ceil(y_vec * NperDim / X_range[1])).astype(int)

    if (np.array(jx_vec[jx_vec == NperDim].shape) > 0):
        # print('Warning, ... %i data point(s) in xcoord at the bounds of X domain' % (jx_vec[jx_vec == NperDim].shape))
        jx_vec[jx_vec == NperDim] = NperDim - 1
    if (np.array(iy_vec[iy_vec == NperDim].shape) > 0):
        # print('Warning, ... %i data point(s) in ycoord at the bounds of X domain' % (iy_vec[iy_vec == NperDim].shape))
        iy_vec[iy_vec == NperDim] = NperDim - 1

    return grid_ij[iy_vec, jx_vec]
    '''


def write_table_l_test_real_data(testing_periods, N_test, l_test_GP, l_test_kde_default, l_test_kde_silverman,
                                 fout_dir=None, idx_samples=None):
    # output dir
    if fout_dir is None:
        fout_dir = "output/tables"
    if not os.path.isdir(fout_dir):
        os.mkdir(fout_dir)

    fid = open(fout_dir + '/table_l_test_real_data.tex', 'w')
    fid.write("\\begin{table}[h!]\n")
    fid.write("\centering\n")
    fid.write("\small\n")
    if idx_samples is None:
        fid.write("\caption{Test likelihood $\\ell_{\\rm test}$ of unseen test data sets.}")
    if idx_samples is not None:
        fid.write(
            "\caption{Test likelihood $\\ell_{\\rm test}$ of unseen test data sets using $K=%i$ posterior samples.}" % len(
                idx_samples))
    fid.write("\n")
    fid.write("\\begin{tabular}{lcccc}")
    fid.write("\n")
    fid.write("\hline")
    fid.write("\n")
    fid.write("testing period & $N_{\\rm test}$ & ETAS-classical & ETAS–Silverman & GP-ETAS \\\ ")
    fid.write("\hline\n")
    for i in range(len(testing_periods[:, 0])):
        if i == 0:
            Line = "%.0f days & %i & %.1f & %.1f & \\textbf{%.1f} \\\ \n" % (
                np.diff(testing_periods[i, :]) / 1., N_test[i], l_test_kde_default[i], l_test_kde_silverman[i],
                l_test_GP[i])
        elif i == 1:
            if np.diff(testing_periods[i, :]) / 365.25 >= 0.5:
                Line = "%.0f years & %i & %.1f & %.1f & \\textbf{%.1f} \\\ \n" % (
                    np.diff(testing_periods[i, :]) / 365.25, N_test[i], l_test_kde_default[i], l_test_kde_silverman[i],
                    l_test_GP[i])
            else:
                Line = "%.0f days & %i & %.1f & %.1f & \\textbf{%.1f} \\\ \n" % (
                    np.diff(testing_periods[i, :]) / 1., N_test[i], l_test_kde_default[i], l_test_kde_silverman[i],
                    l_test_GP[i])
        elif i == len(testing_periods[:, 0]) - 1:
            Line = "total test period ($\\approx %.1f$ years) & %i & %.1f & %.1f & \\textbf{%.1f} \\\ \n" % (
                np.diff(testing_periods[i, :]) / 365.25, N_test[i], l_test_kde_default[i], l_test_kde_silverman[i],
                l_test_GP[i])
        else:
            if np.diff(testing_periods[i, :]) / 365.25 >= 0.5:
                Line = "%.0f years & %i & %.1f & %.1f & \\textbf{%.1f} \\\ \n" % (
                    np.diff(testing_periods[i, :]) / 365.25, N_test[i], l_test_kde_default[i], l_test_kde_silverman[i],
                    l_test_GP[i])
            else:
                Line = "%.0f days & %i & %.1f & %.1f & \\textbf{%.1f} \\\ \n" % (
                    np.diff(testing_periods[i, :]) / 1., N_test[i], l_test_kde_default[i], l_test_kde_silverman[i],
                    l_test_GP[i])
        fid.write(Line)
    fid.write("\hline\n")
    fid.write("\end{tabular}\n")
    fid.write("\end{table}\n")
    fid.close()


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
        self.time_origin = None


class obs():
    def __init__(self):
        self.times = None
        self.magnitudes = None
        self.positions = None
        self.positions_UTM_km = None
