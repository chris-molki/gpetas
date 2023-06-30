import numpy as np
from scipy.linalg import solve_triangular
import scipy as sc
from scipy import stats
import time
import gpetas
from gpetas.utils.some_fun import get_grid_data_for_a_point
import sys

# some globals
time_format = "%Y-%m-%d %H:%M:%S.%f"
output_dir = "output_pred"
output_dir_tables = "output_pred/tables"
output_dir_figures = "output_pred/figures"
output_dir_data = "output_pred/data"


# new 1D implementation
def simulation(obj, mu, theta_off):
    # Ht and b and Xdomain
    tt, mm, la, lo, T, Lat, Lon, Mmin, Mmax, b = obj.general_params_Ht

    # theta offspring
    K, c, p, m_alpha, d, gamma, q = theta_off
    alpha = m_alpha / np.log(10)
    D = np.copy(d)

    # some coord conversion ?
    # km2lat = 1.0 / dist(np.mean(Lat)-0.5, np.mean(Lon), np.mean(Lat)+0.5, np.mean(Lon))
    # km2lon = 1.0 / dist(np.mean(Lat), np.mean(Lon)-0.5, np.mean(Lat), np.mean(Lon)+0.5)

    # new background events:
    # N = np.random.poisson(mu*(T[1]-T[0]))
    # tt = np.append(tt, np.random.uniform(T[0], T[1], N))
    # la = np.append(la, np.random.uniform(Lat[0], Lat[1], N))
    # lo = np.append(lo, np.random.uniform(Lon[0], Lon[1], N))
    # mm = np.append(mm, GRsampling(b, Mmin, Mmax, N))

    # bg_events in mu
    bg_events = mu
    if len(bg_events) > 0:
        tt = np.append(tt, bg_events[:, 0])
        mm = np.append(mm, bg_events[:, 1])
        lo = np.append(lo, bg_events[:, 2])
        la = np.append(la, bg_events[:, 3])
    idx = np.argsort(tt)
    tt = tt[idx]
    mm = mm[idx]
    lo = lo[idx]
    la = la[idx]

    # aftershocks with t<T[1]:
    k = 0
    while k < len(tt):
        m0 = mm[k]
        la0 = la[k]
        lo0 = lo[k]
        if tt[k] > T[0]:
            timeaft = 0.0
        else:
            timeaft = T[0] - tt[k]
        timerem = T[1] - tt[k]
        rho = K * np.power(10.0, alpha * (m0 - Mmin))
        while timeaft < timerem:
            dum = np.random.uniform(0, 1, 1)
            if p == 1.0:
                TT = np.exp(np.log(c + timeaft) - (1 / rho) * dum) - c - timeaft;
                timeaft = timeaft + TT
            else:
                n3 = np.exp((rho / (1.0 - p)) * (timeaft + c) ** (1.0 - p))
                if p > 1.0 and dum < n3:
                    timeaft = timerem + 1.0
                else:
                    n1 = (timeaft + c) ** (1.0 - p)
                    n2 = (1.0 - p) * np.log(dum) / rho
                    TT = (n1 - n2) ** (1.0 / (1.0 - p)) - c - timeaft
                    timeaft = timeaft + TT
            if timeaft < timerem:
                # lata, lona = probr(la0, lo0, m0, d, gamma, q, km2lat, km2lon)
                xy_array = sample_long_range_decay_RL_pwl(N=1, x_center=lo0, y_center=la0, m_center=m0, q=q, D=D,
                                                          gamma_m=gamma)
                lona = xy_array[:, 0]
                lata = xy_array[:, 1]
                tt = np.append(tt, tt[k] + timeaft)
                mm = np.append(mm, GRsampling(b, Mmin, Mmax, 1))
                la = np.append(la, lata)
                lo = np.append(lo, lona)
        k += 1
    # N_star
    N_star = len(tt)

    # N_star_T events only in the forecast time window no matter where
    idx = ((tt >= T[0]) & (tt <= T[1]))
    N_star_in_Tstar = len(tt[idx])

    # select only events within the forecast period and the defined region:
    ind = ((tt >= T[0]) & (la >= Lat[0]) & (la <= Lat[1]) & (lo >= Lon[0]) & (lo <= Lon[1]))
    tt = tt[ind]
    mm = mm[ind]
    la = la[ind]
    lo = lo[ind]
    # ordering in time
    ind = np.argsort(tt)
    # return tt[ind], mm[ind], la[ind], lo[ind]
    N_star_in_Tstar_X = len(tt[ind])
    return N_star, N_star_in_Tstar, N_star_in_Tstar_X


class predictions_1d:
    def __init__(self, save_obj_GS, tau1, tau2, tau0_Ht=None,
                 Ksim_per_sample=None,
                 m_max=None,
                 sample_idx_vec=None,
                 seed=None, approx=None,
                 randomized_samples='yes',
                 Bayesian_m_beta=None, mle_obj=None):

        # mle
        self.mle_obj = mle_obj

        # data
        if save_obj_GS is None:
            self.data_obj = mle_obj.data_obj
        else:
            self.data_obj = save_obj_GS['data_obj']

        # inference results
        self.save_obj_GS = save_obj_GS  # new ... maybe delete all individual sub attributes
        self.save_pred = None
        if sample_idx_vec is None:
            sample_idx_vec = [0]
        self.sample_idx_vec = sample_idx_vec

        # simulation params
        if Ksim_per_sample is None:
            Ksim_per_sample = 1
        self.Ksim_per_sample = Ksim_per_sample
        if save_obj_GS is not None:
            self.Ksamples_total = len(save_obj_GS['lambda_bar'])
            self.Ksamples_used = len(self.sample_idx_vec)
            self.Ksim_total = self.Ksamples_used * self.Ksim_per_sample
            print('Ksamples_total =', self.Ksamples_total)
            print('Ksamples_used  =', self.Ksamples_used)
            print('Ksim_per_sample=', self.Ksim_per_sample)
            print('Ksim_total     =', self.Ksim_total)
        else:
            self.Ksamples_total = 1
            self.Ksamples_used = 1
            self.Ksim_total = self.Ksamples_used * self.Ksim_per_sample
            print('MLE mode: equivalent to 1 sample case')
            print('Ksamples_total =', self.Ksamples_total)
            print('Ksamples_used  =', self.Ksamples_used)
            print('Ksim_per_sample=', self.Ksim_per_sample)
            print('Ksim_total     =', self.Ksim_total)

        # parameters marks
        self.m0 = self.data_obj.domain.m0
        if m_max is None:
            m_max = 7.0
            print('m_max of simulations is not specified, thus set to m=%.2f' % m_max)
        self.m_max = m_max
        if save_obj_GS is not None:
            self.m_beta = save_obj_GS['setup_obj'].m_beta
        else:
            self.m_beta = mle_obj.m_beta_lower_T2

        # forecast time window
        self.tau1 = tau1
        self.tau2 = tau2

        # considered history H_tau1
        if tau0_Ht is None:
            tau0_Ht = 0.
        self.tau0_Ht = tau0_Ht
        self.tau_vec = np.array([tau0_Ht, tau1, tau2])

        # get Ht
        self.Ht = self.get_Ht()

        # get inputs
        self.init_general_fixed_params()


        # init save_dict
        init_save_dictionary(self)

        # simulate background
        for i in range(self.Ksim_per_sample):
            if save_obj_GS is not None:
                self.save_pred['pred_bgnew'].append(self.sim_background_gpetas(static_background='yes'))
            else:
                self.save_pred['pred_bgnew'].append(self.sim_background_mle(static_background='yes'))

        # simulate offspring
        N_star_array = np.zeros([self.Ksim_per_sample, self.Ksamples_used, 6])
        for k_sim in range(self.Ksim_per_sample):
            for k_sample in range(self.Ksamples_used):
                bg_events = self.save_pred['pred_bgnew'][k_sim][k_sample]
                if save_obj_GS is not None:
                    k = self.sample_idx_vec[k_sample]
                    theta_off = self.save_obj_GS['theta_tilde'][k]
                else:
                    k = 1
                    theta_off = self.mle_obj.theta_mle_Kcpadgq
                mu = np.copy(bg_events)
                N_0_Tstar = len(mu)
                N_star_array[k_sim, k_sample, :3] = simulation(self, mu, theta_off)
                N_star_array[k_sim, k_sample, 3] = k
                N_star_array[k_sim, k_sample, 4] = k_sim
                N_star_array[k_sim, k_sample, 5] = N_0_Tstar
            sys.stdout.write('\r' + str('\t simulation %3d/%d: N=%d. N0=%d\r.' % (k_sim + 1, self.Ksim_per_sample, N_star_array[k_sim, k_sample, 1], N_0_Tstar)))
            sys.stdout.flush()

        self.N_star_array = N_star_array

    def init_general_fixed_params(self):
        Ht = np.copy(self.Ht)
        tt = Ht[:, 0]
        mm = Ht[:, 1]
        lo = Ht[:, 2]
        la = Ht[:, 3]
        T = [self.tau1, self.tau2]  # [0.0, 100.0]
        Lon = self.data_obj.domain.X_borders[0, :]
        Lat = self.data_obj.domain.X_borders[1, :]
        Mmin = self.m0
        Mmax = self.m_max
        b = self.m_beta / np.log(10)
        self.general_params_Ht = [tt, mm, la, lo, T, Lat, Lon, Mmin, Mmax, b]

    def get_Ht(self):
        tau0, tau1, tau2 = np.copy(self.tau_vec)
        idx = np.where((self.data_obj.data_all.times >= tau0) & (self.data_obj.data_all.times <= tau1))
        Ht = np.empty([len(self.data_obj.data_all.times[idx]), 4])
        Ht[:, 0] = np.copy(self.data_obj.data_all.times[idx])
        Ht[:, 1] = np.copy(self.data_obj.data_all.magnitudes[idx])
        Ht[:, 2] = np.copy(self.data_obj.data_all.positions[idx, 0])
        Ht[:, 3] = np.copy(self.data_obj.data_all.positions[idx, 1])
        return Ht

    def sim_background_gpetas(self, static_background=None):
        tt, mm, la, lo, T, Lat, Lon, Mmin, Mmax, b = self.general_params_Ht
        if static_background is not None:
            idx_samples = self.sample_idx_vec.astype(int)
            mu_vec = np.sum(np.array(self.save_obj_GS['mu_grid'])[idx_samples], axis=1) * np.prod(
                np.diff(self.data_obj.domain.X_borders)) / len(
                self.save_obj_GS['X_grid'])
            N_0 = np.random.poisson(lam=mu_vec * (self.tau2 - self.tau1))
            bg_events = []
            for size in N_0:
                if size > 0:
                    bg_events_k = np.zeros([size, 5])
                    bg_events_k[:, 0] = np.random.rand(size) * (self.tau2 - self.tau1) + self.tau1
                    bg_events_k[:, 1] = GRsampling(b, Mmin, Mmax, size)
                    bg_events_k[:, 2] = np.random.rand(size) * np.diff(self.data_obj.domain.X_borders[0, :]) + \
                                        self.data_obj.domain.X_borders[0, 0]
                    bg_events_k[:, 3] = np.random.rand(size) * np.diff(self.data_obj.domain.X_borders[1, :]) + \
                                        self.data_obj.domain.X_borders[1, 0]
                    sort_idx = np.argsort(bg_events_k[:, 0])
                    bg_events.append(bg_events_k[sort_idx, :])
                else:
                    bg_events.append(np.array([]))
        else:
            bg_events = -1

        return bg_events

    def sim_background_mle(self, static_background=None):
        tt, mm, la, lo, T, Lat, Lon, Mmin, Mmax, b = self.general_params_Ht
        if static_background is not None:
            mu_vec = np.sum(self.mle_obj.mu_grid) * np.prod(np.diff(self.mle_obj.data_obj.domain.X_borders)) / len(
                self.mle_obj.X_grid)
            N_0 = np.random.poisson(lam=mu_vec * (self.tau2 - self.tau1))
            bg_events = []
            if N_0 > 0:
                size = N_0
                bg_events_k = np.zeros([size, 5])
                bg_events_k[:, 0] = np.random.rand(size) * (self.tau2 - self.tau1) + self.tau1
                bg_events_k[:, 1] = GRsampling(b, Mmin, Mmax, size)
                bg_events_k[:, 2] = np.random.rand(size) * np.diff(self.data_obj.domain.X_borders[0, :]) + \
                                    self.data_obj.domain.X_borders[0, 0]
                bg_events_k[:, 3] = np.random.rand(size) * np.diff(self.data_obj.domain.X_borders[1, :]) + \
                                    self.data_obj.domain.X_borders[1, 0]
                sort_idx = np.argsort(bg_events_k[:, 0])
                bg_events.append(bg_events_k[sort_idx, :])
            else:
                bg_events.append(np.array([]))
        else:
            bg_events = -1

        return bg_events


def init_save_dictionary(obj):
    obj.save_pred = {'pred_bgnew': [],
                     'pred_offspring_Ht': [],
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


def GRsampling(b, Mmin, Mmax, N):
    # beta = np.log(10.0) * b
    # dum = np.random.uniform(0, 1, N)
    # M = Mmin - np.log(1.0 - dum * (1 - np.exp(-beta*(Mmax-Mmin)))) / beta
    M = sample_from_truncated_exponential_rv(beta=np.log(10.) * b, a=Mmin, b=Mmax, sample_size=N, seed=None)
    return M


def dist(lat1, lon1, lat2, lon2):
    """
    Distance (in [km]) between points given as [lat,lon]
    """
    R0 = 6367.3
    D = R0 * np.arccos(
        np.sin(np.radians(lat1)) * np.sin(np.radians(lat2)) +
        np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(np.radians(lon1 - lon2)))
    return D


def probr(lat0, lon0, m0, d, gamma, q, km2lat, km2lon):
    D = d * np.power(10.0, gamma * m0)
    dum1 = np.random.uniform(0, 1)
    dum2 = np.random.uniform(0, 1)
    rr = D * np.sqrt((1.0 - dum1) ** (-1.0 / (q - 1.0)) - 1.0)
    lata = lat0 + km2lat * rr * np.cos(2.0 * np.pi * dum2)
    lona = lon0 + km2lon * rr * np.sin(2.0 * np.pi * dum2)
    return lata, lona


def sample_from_truncated_exponential_rv(beta, a, b=None, sample_size=1, seed=None):
    if seed is not None:
        np.random.seed(seed)
    if b is None:
        b = np.inf
    X = stats.truncexpon(b=beta * (b - a), loc=a, scale=1. / beta)

    return X.rvs(sample_size)


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










####################################
# old stuff to be deleted in future
####################################

class predictions_1d_gpetas:
    def __init__(self,save_obj_GS, tau1, tau2, tau0_Ht=0., sample_idx_vec=None, seed=None, approx=None, Ksim=None,
                 randomized_samples='yes', Bayesian_m_beta=None):
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

        init_save_dictionary(self)
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

            # bg
            mu = np.sum(save_obj_GS['mu_grid'][k]) * np.prod(np.diff(self.data_obj.domain.X_borders)) / len(
                save_obj_GS['X_grid'])
            N_0 = np.random.poisson(lam=mu * (tau2 - tau1), size=1).item()

            # offspring params
            self.theta_Kcpadgq = None
            self.theta_Kcpadgq = np.copy(save_obj_GS['theta_tilde'][k])

            if N_0 == 0:
                self.bgnew = np.zeros([int(len([])), 5])
                self.bgnew_and_offspring = np.zeros([int(len([])), 5])
            else:
                # writing to a matrix
                bg_events = np.zeros([N_0, 5])
                bg_events[:, 0] = np.random.rand(N_0) * (tau2 - tau1)
                bg_events[:, 1] = np.random.exponential(1. / self.m_beta,
                                                        N_0) + self.m0  # m_i: marks
                bg_events[:, 2] = None  # x_coord
                bg_events[:, 3] = None  # y_coord
                bg_events[:, 4] = np.zeros(N_0)  # branching z=0
                sort_idx = np.argsort(bg_events[:, 0])
                bg_events = bg_events[sort_idx, :]

                # offspring (aftershocks) using: mle theta_phi
                pred, pred_aug = sim_add_offspring(self, bg_events)
                bgnew_and_offspring = np.copy(pred_aug)

                # info 1: bg+off
                self.bgnew = np.copy(bg_events)
                self.bgnew[:, 0] += tau1
                self.bgnew_and_offspring = np.copy(bgnew_and_offspring)
                self.bgnew_and_offspring[:, 0] += tau1

            # offspring from Ht
            Ht_pred, Ht_pred_aug = sim_offspring_from_Ht(self)

            # adding together: offspring from Ht + new background + offspring of background
            if N_0 == 0:
                added = np.copy(Ht_pred_aug)
            else:
                added = np.vstack([Ht_pred_aug, self.bgnew_and_offspring])
            events_all_with_Ht_offspring = added[np.argsort(added[:, 0])]

            # info 2: off_from_Ht
            self.Ht_offspring = np.copy(Ht_pred_aug)
            self.events_all_with_Ht_offspring = np.copy(events_all_with_Ht_offspring)

            # save results
            self.save_pred['pred_offspring_Ht'].append(self.Ht_offspring)
            self.save_pred['pred_bgnew_and_offspring'].append(self.bgnew_and_offspring)
            self.save_pred['pred_bgnew_and_offspring_with_Ht_offspring'].append(
                self.events_all_with_Ht_offspring)
            if len(self.events_all_with_Ht_offspring[:, 1]) > 0:
                self.save_pred['M_max_all'].append(np.max(self.events_all_with_Ht_offspring[:, 1]))
            self.save_pred['k_sample'].append(k)
            self.save_pred['theta_Kcpadgq_k'].append(self.theta_Kcpadgq)
            self.save_pred['N495_offspring_Ht'].append(np.sum(self.Ht_offspring[:, 1] >= 4.95))
            self.save_pred['N495_all_with_Ht'].append(np.sum(self.events_all_with_Ht_offspring[:, 1] >= 4.95))
            self.save_pred['N495_all_without_Ht'].append(np.sum(self.bgnew_and_offspring[:, 1] >= 4.95))
            self.save_pred['N495_bgnew'].append(np.sum(self.bgnew[:, 1] >= 4.95))
            self.save_pred['Nm0_offspring_Ht'].append(np.sum(self.Ht_offspring[:, 1] >= self.m0))
            self.save_pred['Nm0_all_with_Ht'].append(np.sum(self.events_all_with_Ht_offspring[:, 1] >= self.m0))
            self.save_pred['Nm0_all_without_Ht'].append(np.sum(self.bgnew_and_offspring[:, 1] >= self.m0))
            self.save_pred['Nm0_bgnew'].append(np.sum(self.bgnew[:, 1] >= self.m0))
            K, c, p, m_alpha = self.theta_Kcpadgq[:4]
            m_beta = np.copy(self.m_beta)
            self.n_stability = gpetas.utils.some_fun.n(m_alpha, m_beta, K, c, p, t_start=tau1, t_end=tau2)
            self.n_stability_inf = gpetas.utils.some_fun.n(m_alpha, m_beta, K, c, p, t_start=0., t_end=np.inf)
            self.save_pred['n_tau1_tau2'].append(self.n_stability)
            self.save_pred['n_inf'].append(self.n_stability_inf)

            if np.mod(k, 1) == 0:  # info every 10th event
                tictoc = time.perf_counter() - self.tic
                print('current simulation is k =', k + 1, 'of K =', Ksim,
                      '%.2f sec. elapsed time.' % tictoc,
                      ' approx. done %.1f percent.' % (100. * (k / float(Ksim))))

            # some postprocessing
            self.save_pred['cumsum'] = gpetas.prediction_2d.cumsum_events_pred(save_obj_pred=self.save_pred,
                                                                               tau1=self.tau1, tau2=self.tau2,
                                                                               m0=self.m0,
                                                                               which_events='all')






class predictions_1d_mle:
    def __init__(self, mle_obj, tau1, tau2, tau0_Ht=0., Ksim=None, data_obj=None, seed=None, Bayesian_m_beta=None):
        '''

        :param mle_obj:
        :type mle_obj:
        :param tau1:
        :type tau1:
        :param tau2:
        :type tau2:
        :param tau0_Ht:
        :type tau0_Ht:
        :param Ksim:
        :type Ksim:
        :param data_obj:
        :type data_obj:
        :param seed:
        :type seed:
        :param Bayesian_m_beta:
        :type Bayesian_m_beta:
        '''
        if Ksim is None:
            Ksim = 1
        dim = 1

        self.mle_obj = mle_obj
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

        # offspring params
        self.theta_Kcpadgq = np.copy(mle_obj.theta_mle_Kcpadgq)

        # fixed params
        self.m0 = self.data_obj.domain.m0
        self.m_beta = mle_obj.m_beta_lower_T2
        self.Bayesian_m_beta = Bayesian_m_beta
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

        # bg
        mu = np.sum(mle_obj.mu_grid) * mle_obj.absX / len(mle_obj.X_grid)
        N_0_Ksim = np.random.poisson(lam=mu * (tau2 - tau1), size=Ksim)

        for k in range(Ksim):
            N_0 = N_0_Ksim[k]
            if N_0 == 0:
                self.bgnew = np.zeros([int(len([])), 5])
                self.bgnew_and_offspring = np.zeros([int(len([])), 5])
            else:
                # writing to a matrix
                bg_events = np.zeros([N_0, 5])
                bg_events[:, 0] = np.random.rand(N_0) * (tau2 - tau1)
                bg_events[:, 1] = np.random.exponential(1. / self.m_beta,
                                                        N_0) + self.m0  # m_i: marks
                bg_events[:, 2] = None  # x_coord
                bg_events[:, 3] = None  # y_coord
                bg_events[:, 4] = np.zeros(N_0)  # branching z=0
                sort_idx = np.argsort(bg_events[:, 0])
                bg_events = bg_events[sort_idx, :]

                # offspring (aftershocks) using: mle theta_phi
                pred, pred_aug = sim_add_offspring(self, bg_events)
                bgnew_and_offspring = np.copy(pred_aug)

                # info 1: bg+off
                self.bgnew = np.copy(bg_events)
                self.bgnew[:, 0] += tau1
                self.bgnew_and_offspring = np.copy(bgnew_and_offspring)
                self.bgnew_and_offspring[:, 0] += tau1

            # offspring from Ht
            Ht_pred, Ht_pred_aug = sim_offspring_from_Ht(self)

            # adding together: offspring from Ht + new background + offspring of background
            if N_0 == 0:
                added = np.copy(Ht_pred_aug)
            else:
                added = np.vstack([Ht_pred_aug, self.bgnew_and_offspring])
            events_all_with_Ht_offspring = added[np.argsort(added[:, 0])]

            # info 2: off_from_Ht
            self.Ht_offspring = np.copy(Ht_pred_aug)
            self.events_all_with_Ht_offspring = np.copy(events_all_with_Ht_offspring)


            # save results
            self.save_pred['pred_offspring_Ht'].append(self.Ht_offspring)
            self.save_pred['pred_bgnew_and_offspring'].append(self.bgnew_and_offspring)
            self.save_pred['pred_bgnew_and_offspring_with_Ht_offspring'].append(
                self.events_all_with_Ht_offspring)
            if len(self.events_all_with_Ht_offspring[:, 1]) > 0:
                self.save_pred['M_max_all'].append(np.max(self.events_all_with_Ht_offspring[:, 1]))
            self.save_pred['k_sample'].append(k)
            self.save_pred['theta_Kcpadgq_k'].append(self.theta_Kcpadgq)
            self.save_pred['N495_offspring_Ht'].append(np.sum(self.Ht_offspring[:, 1] >= 4.95))
            self.save_pred['N495_all_with_Ht'].append(np.sum(self.events_all_with_Ht_offspring[:, 1] >= 4.95))
            self.save_pred['N495_all_without_Ht'].append(np.sum(self.bgnew_and_offspring[:, 1] >= 4.95))
            self.save_pred['N495_bgnew'].append(np.sum(self.bgnew[:, 1] >= 4.95))
            self.save_pred['Nm0_offspring_Ht'].append(np.sum(self.Ht_offspring[:, 1] >= self.m0))
            self.save_pred['Nm0_all_with_Ht'].append(np.sum(self.events_all_with_Ht_offspring[:, 1] >= self.m0))
            self.save_pred['Nm0_all_without_Ht'].append(np.sum(self.bgnew_and_offspring[:, 1] >= self.m0))
            self.save_pred['Nm0_bgnew'].append(np.sum(self.bgnew[:, 1] >= self.m0))
            K, c, p, m_alpha = self.theta_Kcpadgq[:4]
            m_beta = np.copy(self.m_beta)
            self.n_stability = gpetas.utils.some_fun.n(m_alpha, m_beta, K, c, p, t_start=tau1, t_end=tau2)
            self.n_stability_inf = gpetas.utils.some_fun.n(m_alpha, m_beta, K, c, p, t_start=0., t_end=np.inf)
            self.save_pred['n_tau1_tau2'].append(self.n_stability)
            self.save_pred['n_inf'].append(self.n_stability_inf)

            if np.mod(k, 1) == 0:  # info every 10th event
                tictoc = time.perf_counter() - self.tic
                print('current simulation is k =', k + 1, 'of K =', Ksim,
                      '%.2f sec. elapsed time.' % tictoc,
                      ' approx. done %.1f percent.' % (100. * (k / float(Ksim))))

            # some postprocessing
            self.save_pred['cumsum'] = gpetas.prediction_2d.cumsum_events_pred(save_obj_pred=self.save_pred,
                                                          tau1=self.tau1, tau2=self.tau2, m0=self.m0,
                                                          which_events='all')


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
    K, c, p, m_alpha = obj.theta_Kcpadgq[:4]

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

        fparams = np.array([t_i, m_i, K, c, p, m_alpha, m0])
        max_lambda = MOL_fun(t_i, t_i, m_i, K, c, p, m_alpha, m0)
        xnew = NHPPsim(t_i, tau2 - tau1, lambda_fun=MOL_fun, fargs=fparams, max_lambda=max_lambda, dim=1)
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
            # empty xcoordinates
            xnew = np.hstack((xnew, np.zeros([Nnew, 2])))
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

    print('in add_offspring K:', K)
    return data_tmxyz, complete_branching_structure_tmxyzcg


def NHPPsim(tau1, tau2, lambda_fun, fargs, max_lambda, dim=1):
    '''a function which simulates a 1D NHPP in [t1,t2)'''
    Nc = np.random.poisson(lam=(tau2 - tau1) * max_lambda)
    X_unthinned = np.random.uniform(tau1, tau2, size=Nc)
    idx = (np.random.uniform(0., max_lambda, size=Nc) <= lambda_fun(X_unthinned, *fargs))
    return np.sort(X_unthinned[idx])

'''
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
    '''


def sim_offspring_from_Ht(obj, Ht=None, all_pts_yes=None, print_info=None):
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
    K, c, p, m_alpha = obj.theta_Kcpadgq[:4]

    x0 = X0_events[:, 0:4]
    if print_info is not None:
        print('tau_vec              =', obj.tau_vec)
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

            # empty xcoordinates
            xnew = np.hstack((xnew, np.zeros([Nnew, 2])))

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
