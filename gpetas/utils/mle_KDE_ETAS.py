import os
import sys
import numpy as np
from scipy.stats import norm
# from scipy.optimize import minimize
import scipy as sc
from scipy.optimize import (Bounds, NonlinearConstraint, LinearConstraint, minimize)
import pickle

import gpetas


class setup_mle:
    def __init__(self, data_obj, theta_start_Kcpadgqbm0=None, spatial_offspring=None, Nnearest=15, h_min_degree=0.05,
                 spatial_units='degree', utm_yes=None, bins=50, X_grid=None, outdir=None, stable_theta=None,
                 case_name='case_01',silverman=None):

        # data
        self.data_obj = data_obj

        # spatial coordinates (usually degrees)
        self.utm_yes = utm_yes
        self.spatial_units = spatial_units
        if self.utm_yes is not None:
            self.spatial_units = 'km'
            data_obj.domain.X_borders = np.copy(data_obj.domain.X_borders_UTM_km)
            data_obj.data_all.positions = np.copy(data_obj.data_all.positions_UTM_km)
            print('Note: all positions are in UTM now!')

        # data
        idx_training = np.where((data_obj.data_all.times >= data_obj.domain.T_borders_training[0]) & (
                data_obj.data_all.times <= data_obj.domain.T_borders_training[1]))
        self.N_training = len(data_obj.data_all.times[idx_training])
        self.absX_training = np.prod(np.diff(data_obj.domain.X_borders))
        self.absT_training = np.prod(np.diff(data_obj.domain.T_borders_training))
        self.case_name = case_name

        # start values: offspring params K,c,p,m_alpha,d,gamma,q, m_beta, m0
        self.theta_start_Kcpadgqbm0 = theta_start_Kcpadgqbm0
        self.m0 = np.min(data_obj.data_all.magnitudes[idx_training])
        self.m_beta = 1. / np.mean(data_obj.data_all.magnitudes[idx_training] - self.m0)  # np.log(10.)

        if self.theta_start_Kcpadgqbm0 is None:
            self.m0 = np.min(data_obj.data_all.magnitudes[idx_training])
            self.m_beta = 1. / np.mean(data_obj.data_all.magnitudes[idx_training] - self.m0)  # np.log(10.)
            ### self.theta_start_Kcpadgqbm0 = [0.01 / 4., 0.01, 1.2, 2.3 - 0.1, 0.05, 0.5, 2., self.m_beta, self.m0]
            self.theta_start_Kcpadgqbm0 = [0.01 / 4., 0.01, 1.2, 1.0, 0.05, 0.5, 2.-0.2, self.m_beta, self.m0]
            if spatial_offspring == 'G':
                self.theta_start_Kcpadgqbm0[5:7] = np.zeros(2) + 9999.  # dummies for gamma, q
        if spatial_offspring == 'G':
            self.theta_start_Kcpadgqbm0[5:7] = np.zeros(2) + 9999.  # dummies for gamma, q

        if self.utm_yes is not None:
            self.theta_start_Kcpadgqbm0[4] = self.theta_start_Kcpadgqbm0[4] * 111.
        self.spatial_offspring = spatial_offspring
        if self.spatial_offspring is None:
            self.spatial_offspring = 'R'  # rupture length version

        # start values:  bg params
        self.mu_start = self.N_training / 2. / self.absX_training / self.absT_training
        X_borders_NN = data_obj.domain.X_borders - np.array(
            [[data_obj.domain.X_borders[0, 0], data_obj.domain.X_borders[0, 0]],
             [data_obj.domain.X_borders[1, 0], data_obj.domain.X_borders[1, 0]]])
        self.X_borders_NN = X_borders_NN
        self.X_grid = X_grid
        if self.X_grid is None:
            self.X_grid = gpetas.some_fun.make_X_grid(data_obj.domain.X_borders, nbins=bins)
        self.bins = bins

        # kde parameters
        self.Nnearest = Nnearest  # 5 to 15 # distance of the n. nearest neighbor is used for background smoothing often (n=15)
        self.h_min_degree = h_min_degree  # 0.05 #  minimum distance for smoothing, often called stdmin
        # h_min often = 0.05 degrees which is the range of localizaton error
        if silverman is not None:
            self.h_min_degree = gpetas.some_fun.silverman_scott_rule_d(data_obj.data_all.positions)
            print('h_min_silverman is selected=',self.h_min_degree)
        self.dLL = 10.0  # difference in log-likelihood value between iterations

        # constraints: stable theta
        self.stable_theta = stable_theta

        # create subdirectory for output
        if outdir is None:
            outdir = './inference_results'
        if os.path.isdir(outdir):
            print(outdir + ' subdirectory exists')
        else:
            os.mkdir(outdir)
            print(outdir + ' subdirectory has been created.')
        self.outdir = outdir

        # write to file
        fname_setup_obj = outdir + "/setup_obj_default_%s_mle.all" % (case_name)
        if silverman is not None:
            fname_setup_obj = outdir + "/setup_obj_silverman_%s_mle.all" % (case_name)
        file = open(fname_setup_obj, "wb")  # remember to open the file in binary mode
        pickle.dump(self, file)
        file.close()
        print('mle_setup_obj has been created and saved:', fname_setup_obj)


class mle_units():
    """
    mle classical for the KDE-Epidemic Type Aftershock Sequence (ETAS) model.
    Spaghetti code translated from the C_code routine GFZ
    """

    def __init__(self, data_obj, setup_obj=None, optmethod='BFGS', fout_name=None):

        # spatial coordinates, possibly UTM conversion
        self.utm_yes = setup_obj.utm_yes
        if self.utm_yes is not None:
            data_obj.domain.X_borders = np.copy(data_obj.domain.X_borders_UTM_km)
            data_obj.data_all.positions = np.copy(data_obj.data_all.positions_UTM_km)
            print("Note: all positions are in UTM now!")

        # set all positions to NN in X_domain
        data_obj.data_all.positions[:, 0] = data_obj.data_all.positions[:, 0] - data_obj.domain.X_borders[0, 0]
        data_obj.data_all.positions[:, 1] = data_obj.data_all.positions[:, 1] - data_obj.domain.X_borders[1, 0]
        data_obj.domain.X_borders = data_obj.domain.X_borders - np.array(
            [[data_obj.domain.X_borders[0, 0]], [data_obj.domain.X_borders[1, 0]]])
        print("All positions are shifted such that the origin of X domain is (0,0)")

        # data
        self.data_obj = data_obj
        self.fname_catalog = data_obj.case_name
        self.optmethod = optmethod
        self.idx_training = np.where((data_obj.data_all.times >= data_obj.domain.T_borders_training[0]) & (
                data_obj.data_all.times <= data_obj.domain.T_borders_training[1]))

        # read training data
        self._read_training_data()

        # setup start parameters
        self.setup_obj = setup_obj
        if self.setup_obj is None:
            self.setup_obj = setup_mle(data_obj)
        spatial_units = self.setup_obj.spatial_units

        # info of the setup
        self._print_info()

        # external version for testing
        m0 = self.setup_obj.m0
        outname = 'estimated-parameters-%s.out' % (optmethod)

        Mcut = m0  # cutoff magnitude
        T1 = data_obj.domain.T_borders_training[0]  # 0.0  # start time of fitting period [days]
        T2 = data_obj.domain.T_borders_training[
            1]  # 1e10  # end time of fitting period [days] ... if T2>T_catalog_end: T2=T_catalog_end
        self.absT_training = np.diff(data_obj.domain.T_borders_training)
        self.T1 = T1
        self.T2 = T2
        Nnearest = self.setup_obj.Nnearest  # 5  # distance of the n. nearest neighbor is used for background smoothing

        if spatial_units == 'km':
            stdmin = self.setup_obj.h_min_degree * 111.13  # 0.05 * dist(1.0, 0, 0, 0)  # [km] minimum distance for smoothing
        else:
            stdmin = self.setup_obj.h_min_degree  # 0.05 * dist(1.0, 0, 0, 0)  # [km] minimum distance for smoothing
        self.stdmin = stdmin
        catname = self.fname_catalog

        # all data
        idx = data_obj.data_all.times <= T2
        t = data_obj.data_all.times[idx]
        lon = data_obj.data_all.positions[idx, 0]  # x
        lat = data_obj.data_all.positions[idx, 1]  # y
        m = data_obj.data_all.magnitudes[idx]
        self.m_beta_lower_T2 = 1. / np.mean(data_obj.data_all.magnitudes[idx] - self.setup_obj.m0)
        print('\n\t X domain of inference is:', data_obj.domain.X_borders)
        print('\n\t %s read with %d (m>=%.2f & t<=%.1f) events\n' % (catname, len(t), Mcut, T2))

        ti = t[(t >= T1)]
        lati = lat[(t >= T1)]
        loni = lon[(t >= T1)]
        Ri = np.zeros((len(lati), len(lat)))
        Rii = np.zeros((len(lati), len(lati)))
        stdi = stdmin * np.ones(len(lati))
        for i in range(len(lati)):
            Ri[i, :] = self._dist(lati[i], loni[i], lat, lon)
            Rii[i, :] = self._dist(lati[i], loni[i], lati, loni)
            Rii[i, i] = 0.0
            ri = np.sort(Rii[i, :])
            if ri[Nnearest] > stdi[i]:
                stdi[i] = ri[Nnearest]

        pbacki = 0.5 * np.ones(len(lati))
        mui, mutot = calculate_mu(Rii, stdi, pbacki, T1, T2)

        print(" --> RESULT for time interval [%.1f  %.1f]:\n" % (T1, T2))

        dLL = 10.0
        nround = 0
        while dLL > 0.1:
            nround += 1
            mufac, K, alpha, c, p, d, gamma, q, LL = self._LLoptimize(t, m, ti, Ri, mui, mutot, Mcut, T1, T2,
                                                                      optmethod)
            mui *= mufac
            for i in range(10):
                pbacki = self._update_prob(t, m, ti, Ri, mui, Mcut, c, p, K, alpha, q, d, gamma)
                mui, mutot = calculate_mu(Rii, stdi, pbacki, T1, T2)
            if nround == 1:
                dLL = 10.0
            else:
                dLL = np.abs(LL - LL0)
            LL0 = LL
            print(" nround=%d:  mufac=%f  c=%f   p=%f  K=%f  alpha=%f   q=%f  d0=%f  gamma=%f   LL=%f  dLL=%f" % (
                nround, mufac, c, p, K, alpha, q, d, gamma, LL, dLL))

        print('\n Estimated values: mu=%f\tK=%f\talpha=%f\tc=%f\tp=%f\td=%f\tgamma=%f\tq=%f\n' % (
            mufac, K, alpha, c, p, d, gamma, q))

        # save info
        self.theta_mle_Kcpadgq = np.array([K, c, p, alpha, d, gamma, q])
        self.C_mui = mui
        self.C_mutot = mutot
        self.p_i_vec = pbacki
        self.C_LL = LL
        self.C_dLL = dLL
        self.C_mufac = mufac
        self.h_i_vec = stdi
        self.nround = nround

        # construct mu grid
        self.X_grid = np.copy(self.setup_obj.X_grid)
        self.X_grid_NN = np.copy(self.X_grid) - self.data_obj.domain.X_borders_original[:, 0][np.newaxis]
        self.absX = np.prod(np.diff(self.setup_obj.X_borders_NN))
        # data_positions_training = np.array(self.data.positions)
        data_positions_training = np.array(self.data_obj.data_all.positions[self.idx_training])
        sfac, integ = self._scaling_factor_kde(data_positions_training=data_positions_training, h_vec=self.h_i_vec,
                                               p_vec=self.p_i_vec, X_borders=self.setup_obj.X_borders_NN)
        self.integral_mu_unit_time_mle = np.sum(self.p_i_vec) / self.absT_training
        self.sfac_integ = integ
        self.sfac = sfac
        self.kde_unscaled_grid = self.eval_kde_unscaled(x_prime=self.X_grid_NN,
                                                        data_positions_training=data_positions_training,
                                                        h_vec=self.h_i_vec, p_vec=self.p_i_vec,
                                                        absT_training=self.absT_training)
        self.sfac_riemann_approx = np.sum(self.p_i_vec) / (
                self.absT_training * (self.absX / len(self.kde_unscaled_grid)) * np.sum(self.kde_unscaled_grid))
        self.sfac_riemann_approx_integ = (
                self.absT_training * (self.absX / len(self.kde_unscaled_grid)) * np.sum(self.kde_unscaled_grid))
        self.mu_xi_at_all_data = self.eval_kde_unscaled(x_prime=self.data_obj.data_all.positions,
                                                        data_positions_training=data_positions_training,
                                                        h_vec=self.h_i_vec,
                                                        p_vec=self.p_i_vec,
                                                        absT_training=self.absT_training) * self.sfac_riemann_approx  # or sfac
        self.mu_grid = self.kde_unscaled_grid * self.sfac_riemann_approx
        scaling_factor_kde_used = np.copy(self.sfac_riemann_approx)

        # compute likelihood: training (T_borders_training)
        lnl = gpetas.loglike.eval_lnl(data_obj=data_obj,
                                      mu_xi_at_all_data=self.mu_xi_at_all_data,
                                      integral_mu_x_unit_time=self.integral_mu_unit_time_mle,
                                      theta_phi__Kcpadgq=self.theta_mle_Kcpadgq,
                                      m0=self.data_obj.domain.m0,
                                      X_borders_eval_l=data_obj.domain.X_borders,
                                      T_borders_eval_l=[T1, T2],
                                      spatial_kernel=self.setup_obj.spatial_offspring)
        self.lnl_mle_training = np.copy(lnl.lnl_value)
        print(self.lnl_mle_training)

        # Shifting back to original X domain: X_borders, positions of data_all, X_grid
        self.data_obj.data_all.positions = np.copy(self.data_obj.data_all.positions) + \
                                           self.data_obj.domain.X_borders_original[:, 0][np.newaxis]
        self.data_obj.domain.X_borders = np.copy(self.data_obj.domain.X_borders_original)
        print("All positions are rescaled with origin of original X domain", self.data_obj.domain.X_borders_original)

        # save output
        self.save_data = {'theta_phi_mle__Kcpadgq': self.theta_mle_Kcpadgq,
                          'theta_phi_start__Kcpadgq': self.theta_start_Kcpadgq,
                          'lnl_mle_training': self.lnl_mle_training,
                          'p_i_vec': self.p_i_vec,
                          'h_i_vec': self.h_i_vec,
                          'data_training': self.data,
                          'mu_grid': self.mu_grid,
                          'mu_xi_at_all_data': self.mu_xi_at_all_data,
                          'integral_mu_unit_time_mle': self.integral_mu_unit_time_mle,
                          'Nnearest': self.setup_obj.Nnearest,
                          'h_min': self.setup_obj.h_min_degree,
                          'm0': self.setup_obj.m0,
                          'm_beta_training': self.setup_obj.m_beta,
                          'm_beta_lower_T2': self.m_beta_lower_T2,
                          'spatial_offspring': self.setup_obj.spatial_offspring,
                          'data_obj': self.data_obj,
                          'X_grid': self.X_grid,
                          'X_grid_NN': self.X_grid_NN,
                          'N_training': self.N_training,
                          'T_borders_training': self.data_obj.domain.T_borders_training,
                          'absT_training': np.diff(self.data_obj.domain.T_borders_training),
                          'X_borders': self.data_obj.domain.X_borders,
                          'absX': self.absX,
                          'spatial_units': self.setup_obj.spatial_units,
                          'scaling_factor_kde_used': scaling_factor_kde_used,
                          'C_mufac': self.C_mufac,
                          'C_LL_lnl_value': self.C_LL,
                          'C_mui': self.C_mui,
                          'C_mutot': self.C_mutot,
                          'C_dLL': self.C_dLL,
                          'C_nround': self.nround,
                          'time': None,
                          'sfac_integ': self.sfac_integ,
                          'sfac': self.sfac,
                          'sfac_riemann_approx_integ': self.sfac_riemann_approx_integ,
                          'sfac_riemann_approx': self.sfac_riemann_approx,
                          'h_min_degree': self.setup_obj.h_min_degree}
        # write to file
        if fout_name is None:
            fout_name = "mle_default_hmin_"
        file = open(self.setup_obj.outdir + "/" + fout_name + data_obj.case_name + ".all", "wb")
        pickle.dump(self, file)
        file.close()

    def eval_kde_xprime(self, x_prime):
        data_positions_training = np.array(self.data_obj.data_all.positions[self.idx_training])
        h_vec = np.copy(self.h_i_vec)
        p_vec = np.copy(self.p_i_vec)
        absT_training = np.copy(self.absT_training)
        X_borders = np.copy(self.data_obj.domain.X_borders)
        kde_xprime_unscaled = self.eval_kde_unscaled(x_prime, data_positions_training, h_vec, p_vec, absT_training)
        scaling_fac_analytical, integral = self._scaling_factor_kde(data_positions_training, h_vec, p_vec, X_borders)
        kde_xprime = kde_xprime_unscaled * scaling_fac_analytical
        return kde_xprime

    @staticmethod
    def _scaling_factor_kde(data_positions_training, h_vec, p_vec, X_borders=None):
        """
        computes scaling factor of the kde background intensity
        integral: int_|X|int_|T| 1/|T| sum_i^Ndata (pi0 * k(x-x_i|h_i))
        scaling_fac: sum(p_i0)/integ_kde
        where
        k(x-x_i|h_i): is isotropic Gaussian kernel with individual
                      bandwidth

        :param data_positions_training: (N,2) array, with N x_training points
        :param h_vec:
        :param p_vec:
        :param X_borders:
        :return:
        """
        X = data_positions_training
        dim = X.shape[1]
        N = int(np.size(X) / dim)
        assert h_vec.size == N
        assert p_vec.size == N
        # proper shaping
        h_vec = h_vec.reshape(N)
        p_vec = p_vec.reshape(N)
        integral_dim = np.empty([X.shape[0], dim])
        for i in range(dim):
            A_to_substract = norm.cdf(X_borders[i, 0], loc=X[:, i], scale=h_vec)
            A_all = norm.cdf(X_borders[i, 1], loc=X[:, i], scale=h_vec)
            integral_dim[:, i] = A_all - A_to_substract
        integral = np.sum(p_vec.T * np.prod(integral_dim, axis=1))
        scaling_fac = np.sum(p_vec) / integral
        return scaling_fac, integral

    @staticmethod
    def eval_kde_unscaled(x_prime, data_positions_training, h_vec, p_vec, absT_training):
        """
        Computes KDE at x_prime positions assuming domain X is R^2, therefore unscaled
        :param x_prime: (N,2) array, with N x_prime points
        :param data_positions_training: (N,2) array, with N x_training points
        :param p_vec: (N,1) array
        :param h_vec: (N,1) array
        :param absT_training:
        :return:
        """
        # correct shaping in order to have broadcasts correct
        h_vec = h_vec.reshape(-1, 1)
        p_vec = p_vec.reshape(-1, 1)
        x_prime = x_prime.reshape(-1, 2)
        # computation
        kde_x = np.zeros(x_prime.shape[0]) * np.nan
        T = absT_training
        for ii in np.arange(x_prime.shape[0]):
            x = x_prime[ii, :]
            r_squared = np.sum((x[None].T - data_positions_training.T).T ** 2, axis=1)
            r_squared = r_squared[None].T  # (N,1)
            Z = 2. * np.pi * h_vec ** 2  # (N,1)
            kde_x[ii] = 1. / T * np.sum(p_vec * 1. / Z * np.exp(-1. / 2 * 1. / h_vec ** 2 * r_squared))
        return kde_x


    def _read_training_data(self):
        idx = self.idx_training
        data_obj = self.data_obj
        self.data = data_structure()
        self.data.times = data_obj.data_all.times[idx]
        self.data.magnitudes = data_obj.data_all.magnitudes[idx]
        self.data.positions = data_obj.data_all.positions[np.squeeze(idx), :]
        self.data.positions[:, 0] = self.data.positions[:, 0] - data_obj.domain.X_borders[0, 0]
        self.data.positions[:, 1] = self.data.positions[:, 1] - data_obj.domain.X_borders[1, 0]
        self.N_training = len(np.squeeze(idx))
        self.N_all = len(data_obj.data_all.times)

    def _print_info(self):
        data_obj = self.data_obj
        m0 = self.setup_obj.m0
        print('MLE routine   : mle_units()')
        print('\n\t %s read with %d (m>=%.2f & %.1f <= t <=%.1f) events\n' % (
            self.fname_catalog, self.N_training, m0, data_obj.domain.T_borders_training[0],
            data_obj.domain.T_borders_training[1]))
        print('Start values of the params: mu_start=', self.setup_obj.mu_start)
        print('spatial kernel for offspring         :', self.setup_obj.spatial_offspring)
        print('Check setup theta_start [mufac, K, alpha, c, p, d, gamma, q - 1]:', self._setstartparameter())
        if self.setup_obj.spatial_offspring == 'R':
            print('Rupture length parameterization with:')
            print('Start values of the params: K,c,p,a,d,g,q,b,m0=', self.setup_obj.theta_start_Kcpadgqbm0)
        if self.setup_obj.spatial_offspring == 'P':
            print('Classical power law parameterization with:')
            print('Start values of the params: K,c,p,a,d,g,q,b,m0=', self.setup_obj.theta_start_Kcpadgqbm0)
        if self.setup_obj.spatial_offspring == 'G':
            print('Gaussian kernel for offspring (low range) parameterization with:')
            print('Start values of the params: K,c,p,a,D,b,m0=', self.setup_obj.theta_start_Kcpadgqbm0)
        print('--------------------------------------------------------------')
        print('KDE parameters:')
        print('Nnearest     =', self.setup_obj.Nnearest)
        print('h_min_degree =', self.setup_obj.h_min_degree, ' (units) degrees.')
        if self.setup_obj.spatial_units == 'km':
            print('UTM km are used:')
            print('h_min_km :', self.setup_obj.h_min_degree * 111.13, ' km')

    def _dist(self, lat1, lon1, lat2, lon2):
        """
        Euklidean Distance (in degree or [km]) between 2 points
        """
        R = np.sqrt(np.square(lon2 - lon1) + np.square(lat2 - lat1))
        return R

    def _setstartparameter(self):
        theta_start_Kcpadgqbm0 = self.setup_obj.theta_start_Kcpadgqbm0
        if theta_start_Kcpadgqbm0 is None:
            mufac = 1.0
            K = 0.05
            alpha = 1.0 * np.log(10.)  # exponent for base e
            c = 0.01  # [days]
            p = 1.3
            d = 0.013  # ... with D = d0 * 10^(gamma*M)
            gamma = 0.5  # L(M) = d0*10^(gamma*M); d0=0.013  gamma=0.5  (Wells & Coppersmith 1994; RLD normal faulting)
            q = 1.5  # spatial probability density: f(r) = (q-1)/(pi*D^2) * ( 1 + r^2/D^2 )^(-q)
        else:
            K, c, p, m_alpha, D, gamma, q, m_beta, m0 = theta_start_Kcpadgqbm0
            mufac = 1.0
            alpha = m_alpha  # exponent for base e
            d = D
        self.theta_start_Kcpadgq = np.array([K, c, p, alpha, d, gamma, q])
        return np.asarray([mufac, K, alpha, c, p, d, gamma, q - 1])

    def _LLoptimize(self, t, m, ti, Ri, mui, mutot, Mmin, T1, T2, optmethod):
        x0 = self._setstartparameter()
        if self.setup_obj.stable_theta is None:
            res = minimize(self._nLLETAS, np.sqrt(x0), args=(t, m, ti, Ri, mui, mutot, Mmin, T1, T2), method=optmethod)
        else:
            # dim(x)=8, i.e. x[0] = mufac, x[1]=K, x[2]=m_alpha, x[3]=c, x[4]=p, x[5]=d, x[6]=gamma, x[7]=q-1)
            problem = mle_stable_hawkes(m_beta=self.setup_obj.m_beta)
            nonlinear_constraint = NonlinearConstraint(problem.nl_constraint, 0.0, 1.0,
                                                       jac=problem.jacobian_nl_cons,
                                                       hess=sc.optimize.BFGS())
            bounds = Bounds([-np.inf, -np.inf, -np.inf, -np.inf, 1., -np.inf, -np.inf, -np.inf],
                            [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])

            #            linear_constraint = LinearConstraint(problem.l_constraint,
            #                                                 [-np.inf, -np.inf, -np.inf, -np.inf, 1., -np.inf, -np.inf, -np.inf],
            #                                                 [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])

            res = minimize(self._nLLETAS,
                           x0=np.sqrt(x0),
                           args=(t, m, ti, Ri, mui, mutot, Mmin, T1, T2),
                           method='trust-constr',
                           hess=sc.optimize.BFGS(),
                           jac='cs',
                           constraints=[nonlinear_constraint],
                           options={'verbose': 1, 'maxiter': 1000},
                           bounds=bounds)
            print('n problem', problem.n_problem)

        mufac = np.square(res.x[0])
        K = np.square(res.x[1])
        alpha = np.square(res.x[2])
        c = np.square(res.x[3])
        p = np.square(res.x[4])
        d = np.square(res.x[5])
        gamma = np.square(res.x[6])
        q = 1 + np.square(res.x[7])
        return mufac, K, alpha, c, p, d, gamma, q, -res.fun

    def _nLLETAS(self, arguments, t, m, ti, Ri, mui, mutot, Mmin, T1, T2):
        mufac = np.square(arguments[0])
        K = np.square(arguments[1])
        alpha = np.square(arguments[2])
        c = np.square(arguments[3])
        p = np.square(arguments[4])
        d = np.square(arguments[5])
        gamma = np.square(arguments[6])
        q = 1 + np.square(arguments[7])
        Na = K * np.exp(alpha * (m - Mmin))
        # calculation of SUM log(R(t_i):
        ff = 0.0
        for i in range(len(ti)):
            rate = self._calculate_ETASrate(ti[i], t, m, Ri[i, :], mufac * mui[i], Na, c, p, d, gamma, q, alpha)
            ff += np.log(rate)
        # calculation of INTEGRAL R(t) dt:
        ft = integrateETAS(T1, T2, t, mufac * mutot, Na, c, p)
        nLL = ft - ff
        self.C_ff_intensity_part = ff
        self.C_ft_integral_part = ft
        self.C_mutot_times_mufac = mufac * mutot
        self.C_Na = Na
        if np.isnan(nLL):
            nLL = 1e10
        sys.stdout.write('\r' + str(
            'search: mu=%.3f  K=%.4f  alpha=%.2f  c=%.4f  p=%.2f d=%.4f  gamma=%.2f  q=%.2f --> Nback=%.1f Ntot=%.1f (Z=%d) nLL=%f\r' % (
                mufac, K, alpha, c, p, d, gamma, q, mufac * mutot, ft, len(ti), nLL)))
        sys.stdout.flush()
        # print('\nsearch: mu=%.3f  K=%.4f  alpha=%.2f  c=%.4f  p=%.2f d=%.4f  gamma=%.2f  q=%.2f --> nLL=%f\n' % (mufac, K, alpha, c, p, d, gamma, q, nLL))
        return nLL

    def _calculate_ETASrate(self, time, t, m, r, mu, Na, c, p, d, gamma, q, alpha):
        ''' spatial offspring kernel '''
        spatial_offspring = self.setup_obj.spatial_offspring
        m0 = self.setup_obj.m0

        if spatial_offspring == 'R':
            R0 = mu + np.sum(
                Na[(t < time)] * np.power(c + time - t[(t < time)], -p) * fR(r[(t < time)], m[(t < time)], d, gamma,
                                                                             q))
        if spatial_offspring == 'P':
            R0 = mu + np.sum(
                Na[(t < time)] * np.power(c + time - t[(t < time)], -p) * fP(r[(t < time)], (m[(t < time)] - m0), d,
                                                                             gamma, q))
        if spatial_offspring == 'G':
            R0 = mu + np.sum(
                Na[(t < time)] * np.power(c + time - t[(t < time)], -p) * fG(r[(t < time)], (m[(t < time)] - m0), d,
                                                                             alpha))
        return R0

    def _update_prob(self, t, m, ti, Ri, mui, Mmin, c, p, K, alpha, q, d, gamma):
        Na = K * np.exp(alpha * (m - Mmin))
        prob = np.zeros(len(ti))
        for i in range(len(ti)):
            rate = self._calculate_ETASrate(ti[i], t, m, Ri[i, :], mui[i], Na, c, p, d, gamma, q, alpha)
            # probability to be background event (see Eq.14 of Zhuang & Ogata JGR 2004):
            prob[i] = mui[i] / rate
        return prob


# additional functions
def calculate_mu(Rii, stdi, pbacki, T1, T2):
    mu = np.zeros(len(stdi))
    for i in range(len(stdi)):
        mu[i] = np.sum(pbacki * gaussian(Rii[i, :], stdi)) / (T2 - T1)
    mutot = np.sum(pbacki)
    return mu, mutot


def gaussian(r, sig):
    res = np.exp(-0.5 * np.power(r / sig, 2.)) / (2 * np.pi * np.power(sig, 2.))
    return res


def fR(r, m, d, gamma, q):
    D = d * np.power(10., gamma * m);
    x = 1.0 + np.square(r / D)
    fr = ((q - 1.) / (np.pi * np.square(D))) * np.power(x, -q)
    return fr


def fP(r, m_reduced, d, gamma, q):
    # D = d * np.power(10., gamma * m)
    D = d * np.exp(0.5 * gamma * m_reduced)
    x = 1.0 + np.square(r / D)
    fr = ((q - 1.) / (np.pi * np.square(D))) * np.power(x, -q)
    return fr


def fG(r, m_reduced, d, alpha):
    D = d * np.exp(0.5 * alpha * m_reduced)
    x = -0.5 * np.square(r / D)
    fr = 1. / (2. * np.pi * np.square(D)) * np.exp(x)
    return fr


def integrateETAS(T1, T2, t, mutot, Na, c, p):
    t0 = T1 - t[(t < T2)]
    t0[(t0 < 0)] = 0.0
    t1 = t0 + c
    t2 = T2 - t[(t < T2)] + c
    if p == 1:
        dum1 = np.log(t1)
        dum2 = np.log(t2)
    else:
        dum1 = t1 ** (1.0 - p) / (1.0 - p)
        dum2 = t2 ** (1.0 - p) / (1.0 - p)
    ft = mutot + np.sum(Na[(t < T2)] * (dum2 - dum1))
    return ft


def dist(lat1, lon1, lat2, lon2):
    """
    Distance (in [km]) between points given as [lat,lon]
    """
    R0 = 6367.3
    R = R0 * np.arccos(
        np.sin(np.radians(lat1)) * np.sin(np.radians(lat2)) +
        np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(np.radians(lon1 - lon2)))
    return R


# subclasses
class data_structure():
    def __init__(self):
        self.times = None
        self.magnitudes = None
        self.positions = None


class mle_stable_hawkes():
    def __init__(self, m_beta=np.log(10)):
        self.m_beta = m_beta
        self.n_problem = None
        # dim(x)=8, i.e. x[0] = mufac, x[1]=K, x[2]=m_alpha, x[3]=c, x[4]=p, x[5]=d, x[6]=gamma, x[7]=q-1

    def nl_constraint(self, x):
        xsquared = x ** 2.
        mufac, K, m_alpha, c, p, d, gamma, qminus1 = xsquared
        n_problem = c ** (1 - p) / (p - 1) * K / (1. - m_alpha / self.m_beta)
        self.n_problem = n_problem
        return n_problem

    def jacobian_nl_cons(self, x):
        xsquared = x ** 2.
        mufac, K, m_alpha, c, p, d, gamma, qminus1 = xsquared
        j1 = c ** (1 - p) / (p - 1.) * 1. / (1. - m_alpha / self.m_beta) * 2. * np.sqrt(K)  # d/dsqrt(K)
        j2 = c ** (1 - p) / (p - 1.) * K * (2 * np.sqrt(m_alpha) * self.m_beta) / (
                    m_alpha - self.m_beta) ** 2.  # d/dsqrt(m_alpha)
        j3 = K / ((p - 1.) * (1. - m_alpha / self.m_beta)) * 2. * (1. - p) * c ** (0.5 - p)  # d/dsqrt(c)
        j4 = K / (1. - m_alpha / self.m_beta) * (-2 * np.sqrt(p) * c ** (1. - p) * ((p - 1.) * np.log(c) + 1)) / (
                    p - 1) ** 2.  # d/dsqrt(p)
        jac = np.array([0., j1, j2, j3, j4, 0., 0., 0.]).T
        return jac

    def l_constraint(self, x):
        return x[4]
