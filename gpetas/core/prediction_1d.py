import numpy as np
from scipy.linalg import solve_triangular
import scipy as sc
import time
from gpetas.utils.some_fun import get_grid_data_for_a_point



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
    '''a function which simulates a 1D NHPP in [t1,t2)'''
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
