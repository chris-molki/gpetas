import numpy as np
from scipy.linalg import solve_triangular
import scipy as sc
from scipy import stats
from scipy.special import logsumexp
import time
import gpetas
from gpetas.utils.some_fun import get_grid_data_for_a_point
import sys
import os
import pickle
import matplotlib.pyplot as plt

# some globals
time_format = "%Y-%m-%d %H:%M:%S.%f"
output_dir = "output_LTF"
output_dir_tables = "output_LTF/tables"
output_dir_figures = "output_LTF/figures"
output_dir_data = "output_LTF/data"


def init_outdir():
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    if not os.path.isdir(output_dir_tables):
        os.mkdir(output_dir_tables)
    if not os.path.isdir(output_dir_figures):
        os.mkdir(output_dir_figures)
    if not os.path.isdir(output_dir_data):
        os.mkdir(output_dir_data)


def plot_LTF(perfLTF_obj, clim=None):
    # plot definitions
    pSIZE = 16
    plt.rc('font', size=pSIZE)
    plt.rc('axes', titlesize=pSIZE)
    bins = 'auto'
    x = np.linspace(0., 1.1 * np.max([perfLTF_obj.Nstar, perfLTF_obj.Nstar_mle]), 1000)
    x_N0 = np.linspace(0., 1.1 * np.max([perfLTF_obj.N0_star, perfLTF_obj.N0_star_mle]), 1000)

    # z plots
    if clim == 'yes':
        c2 = np.log10(max(np.max(perfLTF_obj.mu_HE07_m0_sim_ref), np.max(perfLTF_obj.mu_HE07_m0_mle)))
        c1 = np.log10(min(np.min(perfLTF_obj.mu_HE07_m0_sim_ref), np.min(perfLTF_obj.mu_HE07_m0_mle)))
        clim = [c1, c2]

    hf6 = plt.figure(figsize=(20, 8))
    data_star = None
    plt.subplot(2, 4, 2)
    lam = (np.tile(perfLTF_obj.mu_res_obj.mu_x_norm,
                   (perfLTF_obj.pred_obj_1D.Ksim_per_sample, 1)).T * perfLTF_obj.Nstar).T
    lam_mean = np.mean(lam, axis=0)
    plot_2D_z(z=lam_mean, X_grid_plot=perfLTF_obj.X_grid, data_star=data_star, clim=clim, show_colorbar=1)
    plt.subplot(2, 4, 3)
    lam = (np.reshape(perfLTF_obj.mu_res_obj_mle.mu_x_norm, [-1, 1]) * perfLTF_obj.Nstar_mle).T
    lam_mean = np.mean(lam, axis=0)
    plot_2D_z(z=lam_mean, X_grid_plot=perfLTF_obj.X_grid, data_star=data_star, clim=clim, show_colorbar=1)
    plt.subplot(2, 4, 4)
    lam = (np.tile(perfLTF_obj.mu_res_obj.mu_x_norm,
                   (perfLTF_obj.pred_obj_1D.Ksim_per_sample, 1)).T * perfLTF_obj.Nstar).T
    lam_mean = lam[len(perfLTF_obj.mu_res_obj.mu_xprime) - 1, :]
    lam_mean = lam[-1, :]
    print(len(perfLTF_obj.mu_res_obj.mu_xprime) - 1)
    print(lam.shape)
    plot_2D_z(z=lam_mean, X_grid_plot=perfLTF_obj.X_grid, data_star=data_star, clim=clim, show_colorbar=1)
    plt.subplot(2, 4, 5)
    lam = perfLTF_obj.mu_HE07_m0_sim_ref
    lam_mean = np.mean(lam, axis=0)
    plot_2D_z(z=lam_mean, X_grid_plot=perfLTF_obj.HE07_X_grid, data_star=data_star, clim=clim, show_colorbar=1)
    plt.subplot(2, 4, 6)
    lam = perfLTF_obj.mu_HE07_m0_gpetas
    lam_mean = np.mean(lam, axis=0)
    plot_2D_z(z=lam_mean, X_grid_plot=perfLTF_obj.HE07_X_grid, data_star=data_star, clim=clim, show_colorbar=1)
    plt.subplot(2, 4, 7)
    lam = perfLTF_obj.mu_HE07_m0_mle
    lam_mean = np.mean(lam, axis=0)
    plot_2D_z(z=lam_mean, X_grid_plot=perfLTF_obj.HE07_X_grid, data_star=data_star, clim=clim, show_colorbar=1)
    plt.subplot(2, 4, 8)
    lam = perfLTF_obj.mu_HE07_m0_gpetas
    lam_mean = lam[len(perfLTF_obj.mu_res_obj.mu_xprime) - 1]
    plot_2D_z(z=lam_mean, X_grid_plot=perfLTF_obj.HE07_X_grid, data_star=data_star, clim=clim, show_colorbar=1)

    hf7 = plt.figure(figsize=(20, 8))
    data_star = perfLTF_obj.data_star
    plt.subplot(2, 4, 2)
    lam = (np.tile(perfLTF_obj.mu_res_obj.mu_x_norm,
                   (perfLTF_obj.pred_obj_1D.Ksim_per_sample, 1)).T * perfLTF_obj.Nstar).T
    lam_mean = np.mean(lam, axis=0)
    plot_2D_z(z=lam_mean, X_grid_plot=perfLTF_obj.X_grid, data_star=data_star, clim=clim, show_colorbar=1)
    plt.subplot(2, 4, 3)
    lam = (np.reshape(perfLTF_obj.mu_res_obj_mle.mu_x_norm, [-1, 1]) * perfLTF_obj.Nstar_mle).T
    lam_mean = np.mean(lam, axis=0)
    plot_2D_z(z=lam_mean, X_grid_plot=perfLTF_obj.X_grid, data_star=data_star, clim=clim, show_colorbar=1)
    plt.subplot(2, 4, 4)
    lam = (np.tile(perfLTF_obj.mu_res_obj.mu_x_norm,
                   (perfLTF_obj.pred_obj_1D.Ksim_per_sample, 1)).T * perfLTF_obj.Nstar).T
    lam_mean = lam[len(perfLTF_obj.mu_res_obj.mu_xprime) - 1]
    plot_2D_z(z=lam_mean, X_grid_plot=perfLTF_obj.X_grid, data_star=data_star, clim=clim, show_colorbar=1)
    plt.subplot(2, 4, 5)
    lam = perfLTF_obj.mu_HE07_m0_sim_ref
    lam_mean = np.mean(lam, axis=0)
    plot_2D_z(z=lam_mean, X_grid_plot=perfLTF_obj.HE07_X_grid, data_star=data_star, clim=clim, show_colorbar=1)
    plt.subplot(2, 4, 6)
    lam = perfLTF_obj.mu_HE07_m0_gpetas
    lam_mean = np.mean(lam, axis=0)
    plot_2D_z(z=lam_mean, X_grid_plot=perfLTF_obj.HE07_X_grid, data_star=data_star, clim=clim, show_colorbar=1)
    plt.subplot(2, 4, 7)
    lam = perfLTF_obj.mu_HE07_m0_mle
    lam_mean = np.mean(lam, axis=0)
    plot_2D_z(z=lam_mean, X_grid_plot=perfLTF_obj.HE07_X_grid, data_star=data_star, clim=clim, show_colorbar=1)
    plt.subplot(2, 4, 8)
    lam = perfLTF_obj.mu_HE07_m0_gpetas
    lam_mean = lam[len(perfLTF_obj.mu_res_obj.mu_xprime) - 1]
    plot_2D_z(z=lam_mean, X_grid_plot=perfLTF_obj.HE07_X_grid, data_star=data_star, clim=clim, show_colorbar=1)

    hf1 = plt.figure()
    plt.subplot(2, 1, 1)
    hf = plt.hist(perfLTF_obj.Nstar, bins=bins, density=True, color='k', label='GP-E')
    hf = plt.hist(perfLTF_obj.Nstar_mle, bins=bins, density=True, color='b', alpha=0.7, label='E')
    plt.axvline(perfLTF_obj.N_obs, color='m', label='$N_{obs}$')
    hf = plt.hist(perfLTF_obj.HE07_Nstar_m0_sim, bins=bins, density=True, color='g', alpha=0.5, label='HE07')
    plt.ylabel('density')
    plt.xlabel('$N^*$')
    plt.legend(bbox_to_anchor=(1, 1))

    hf2 = plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(perfLTF_obj.Nstar, color='k', label='GP-E')
    plt.plot(perfLTF_obj.Nstar_mle, ':b', label='E')
    plt.plot(perfLTF_obj.HE07_Nstar_m0_sim, '--g', label='HE07')
    plt.axhline(perfLTF_obj.N_obs, color='m', label='$N_{obs}$')
    plt.ylabel('$N^*$')
    plt.xlabel('simulation')

    plt.subplot(1, 2, 2)
    # plt.gca().set(yticklabels=[])
    plt.xlabel('simulation')
    plt.plot(perfLTF_obj.N0_star, 'darkgray')
    plt.plot(perfLTF_obj.N0_star_mle, 'steelblue', linestyle=':')
    plt.axhline(perfLTF_obj.N_obs, color='m', linestyle='--')
    plt.gca().yaxis.tick_right()
    plt.ylabel('$N_0^*$ background events')
    plt.gca().yaxis.set_label_position("right")
    plt.gca().yaxis.tick_right()

    hf3 = plt.figure(figsize=(7, 10))
    plt.subplot(2, 1, 1)
    plt.plot(x, perfLTF_obj.kde_Nstar.pdf(x), 'k')
    plt.plot(x, perfLTF_obj.kde_Nstar_mle.pdf(x), ':b')
    plt.axvline(perfLTF_obj.N_obs, color='m')
    # plt.axvline(Nm0_HE07,color='g')
    plt.plot(x, perfLTF_obj.kde_Nstar_HE07.pdf(x), '--g')
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(4))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(3))
    plt.ylabel('density')
    plt.xlabel('$N^*$')
    print('GP-E  kde(Nobs)=', perfLTF_obj.kde_Nstar(perfLTF_obj.N_obs))
    print('E mle kde(Nobs)=', perfLTF_obj.kde_Nstar_mle(perfLTF_obj.N_obs))

    plt.subplot(2, 1, 2)
    plt.plot(x_N0, perfLTF_obj.kde_N0_star.pdf(x_N0), 'darkgray', linewidth=2)
    plt.plot(x_N0, perfLTF_obj.kde_N0_star_mle.pdf(x_N0), 'steelblue', linestyle=':', linewidth=2)
    plt.axvline(perfLTF_obj.N_obs, color='m', linestyle='--')
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(4))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(3))
    plt.ylabel('density')
    plt.xlabel('$N_0^*$ ')
    plt.show()

    hf4 = plt.figure()
    plt.plot(perfLTF_obj.loglike_gpetas, '.k', label='GP-E')
    plt.plot(perfLTF_obj.loglike_mle, '.b', label='E', alpha=0.7)
    plt.plot(perfLTF_obj.loglike_HE07_ref, '.g', label='HE07', alpha=0.7)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(3))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(3))
    plt.legend(bbox_to_anchor=(1, 1))
    plt.xlabel('simulation')
    plt.ylabel('$\\ln \\mathcal{L}$')

    hf5 = plt.figure()
    hf = plt.hist(perfLTF_obj.loglike_gpetas, color='k', bins='auto', density=True, label='GP-E')
    hf = plt.hist(perfLTF_obj.loglike_mle, color='b', bins='auto', density=True, label='E', alpha=0.7)
    hf = plt.hist(perfLTF_obj.loglike_HE07_ref, color='g', bins='auto', density=True, label='HE07')
    plt.legend(bbox_to_anchor=(1, 1))
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(3))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(3))
    plt.xlabel('$\\ln \\mathcal{L}$')
    plt.ylabel('density')
    print('GP-E   :%.2f' % perfLTF_obj.log_E_L_gpetas)
    print('E mle  :%.2f' % perfLTF_obj.log_E_L_mle)
    print('HE07   :%.2f' % perfLTF_obj.log_E_L_HE07_ref)
    print('===========================')
    ref = perfLTF_obj.log_E_L_gpetas
    print('delta per obs. event GP-E   :%.2f' % ((perfLTF_obj.log_E_L_gpetas - ref) / perfLTF_obj.N_obs))
    print('delta per obs. event E mle  :%.2f' % ((perfLTF_obj.log_E_L_mle - ref) / perfLTF_obj.N_obs))
    print('delta per obs. event HE07   :%.2f' % ((perfLTF_obj.log_E_L_HE07_ref - ref) / perfLTF_obj.N_obs))

    return hf1, hf2, hf3, hf4, hf5, hf6, hf7


def plot_2D_z(z, X_grid_plot=None, data_star=None, clim=None, show_colorbar=None):
    nbins_plot = int(np.sqrt(len(z)))
    if X_grid_plot is not None:
        plt.pcolor(X_grid_plot[:, 0].reshape([nbins_plot, nbins_plot]),
                   X_grid_plot[:, 1].reshape([nbins_plot, nbins_plot]),
                   np.log10(z.reshape([nbins_plot, nbins_plot])))
    else:
        plt.pcolor(np.log10(z.reshape([nbins_plot, nbins_plot])))
    if show_colorbar is not None:
        plt.colorbar(shrink=0.25)
    if data_star is not None:
        plt.plot(data_star[:, 2], data_star[:, 3], '.r')  # ,markersize=3)
    if clim is not None:
        plt.clim(clim)
    plt.axis('square')
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(2))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(2))
    return


def get_data_star(save_obj_GS, tau1, tau2, mstar=None):
    data_obj = save_obj_GS['data_obj']
    m0 = data_obj.domain.m0
    if mstar is None:
        mstar = m0
    idx = np.where(
        (data_obj.data_all.times >= tau1) & (data_obj.data_all.times <= tau2) & (data_obj.data_all.magnitudes >= mstar))
    data_star = np.empty([len(data_obj.data_all.times[idx]), 4])
    data_star[:, 0] = np.copy(data_obj.data_all.times[idx])
    data_star[:, 1] = np.copy(data_obj.data_all.magnitudes[idx])
    data_star[:, 2] = np.copy(data_obj.data_all.positions[idx, 0])
    data_star[:, 3] = np.copy(data_obj.data_all.positions[idx, 1])
    return data_star


class performance_LTF_HE07_m495():
    """
    a scaled variant of m0
    """

    def __init__(self, save_obj_GS, pred_obj_1D,
                 forecast_5yrs_495_HE07,
                 sample_idx_vec=None,
                 mle_obj=None, pred_obj_1D_mle=None,
                 data_star=None,
                 abs_T_data_star=None):

        self.abs_T_forecast_ref = 5. * 365.25  # 5yrs CSEP convention

        # gpetas
        self.save_obj_GS = save_obj_GS
        self.pred_obj_1D = pred_obj_1D
        self.case_name = str(save_obj_GS['case_name'])
        self.X_borders = self.save_obj_GS['data_obj'].domain.X_borders
        self.abs_X = np.prod(np.diff(self.X_borders))
        self.X_grid = save_obj_GS['X_grid']

        # mle if given
        # self.mle_obj = save_obj_GS
        self.mle_obj = mle_obj
        self.pred_obj_1D_mle = pred_obj_1D_mle

        # Extract HE07 for region X
        self.forecast_5yrs_495_HE07 = forecast_5yrs_495_HE07
        regions = gpetas.R00x_setup.R00x_california_set_domain()
        region = eval('regions.polygon_%s' % (self.case_name))
        x, y, z, xx, yy, idx = gpetas.prediction_1d.new_extract_forecast_original_units(forecast=forecast_5yrs_495_HE07,
                                                                                        region=region, plot_yes=None)
        X_grid_HE07_xy = np.zeros([len(x), 2])
        X_grid_HE07_xy[:, 0] = x
        X_grid_HE07_xy[:, 1] = y
        self.HE07_X_grid = X_grid_HE07_xy
        self.L = len(self.HE07_X_grid)
        self.HE07_mu_x_per_0p1x0p1deg_m495 = np.copy(z)
        self.HE07_mu_x_ua_m495 = self.HE07_mu_x_per_0p1x0p1deg_m495 / (0.1 ** 2.)  # adjusting per unit area 'ua'
        self.m_beta = save_obj_GS['setup_obj'].m_beta
        self.m0 = self.save_obj_GS['data_obj'].domain.m0
        self.m0_factor = np.exp(self.m_beta * (4.95 - self.m0))
        self.HE07_mean_Nstar_m0 = np.sum(self.HE07_mu_x_per_0p1x0p1deg_m495) * self.m0_factor
        self.HE07_mu_x_ua_m0 = self.HE07_mu_x_ua_m495 * self.m0_factor
        self.HE07_mu_x_per_0p1x0p1deg_m0 = self.HE07_mu_x_ua_m0 * (0.1) ** 2.
        # normalization
        self.HE07_Zprime_m0 = np.sum(self.HE07_mu_x_ua_m0) * self.abs_X / self.L
        self.HE07_mu_x_ua_m0_xprime_norm = self.HE07_mu_x_ua_m0 / self.HE07_Zprime_m0

        # mu resolution
        # gpetas
        K_samples_total = len(save_obj_GS['lambda_bar'])
        if sample_idx_vec is None:
            n = int(self.pred_obj_1D.Ksim_total / self.pred_obj_1D.Ksamples_total)
            vector = np.arange(0, self.pred_obj_1D.Ksamples_total, 1)
            sample_idx_vec = np.tile(vector, n)  # Repeat the vector n times
        self.sample_idx_vec = sample_idx_vec
        self.mu_res_obj = gpetas.prediction_1d.resolution_mu_gpetas(save_obj_GS,
                                                                    X_grid_prime=self.HE07_X_grid,
                                                                    sample_idx_vec=self.sample_idx_vec,
                                                                    summary=None)
        # mle
        if mle_obj is not None:
            self.mu_res_obj_mle = gpetas.prediction_1d.resolution_mu_mle(mle_obj, X_grid_prime=self.HE07_X_grid)

        # Nstar forecasts for abs_T in X
        if abs_T_data_star is None:
            abs_T_data_star = 5 * 365.25  # CSEP forecast window 5yrs
        self.abs_T_data_star = abs_T_data_star
        self.abs_T_fac = self.abs_T_data_star / self.abs_T_forecast_ref
        self.Nstar = self.pred_obj_1D.N_star_array[:, :, 2].flatten() / self.m0_factor * self.abs_T_fac  # N in T and X
        self.Nstar_mle = self.pred_obj_1D_mle.N_star_array[:, :, 2].flatten() / self.m0_factor * self.abs_T_fac
        self.HE07_Nstar_m0_sim = np.random.poisson(lam=self.HE07_mean_Nstar_m0 * self.abs_T_fac,
                                                   size=len(self.Nstar)) / self.m0_factor
        self.N0_star = self.pred_obj_1D.N_star_array[:, :,
                       -1].flatten() / self.m0_factor * self.abs_T_fac  # N0star in T and X
        self.N0_star_mle = self.pred_obj_1D_mle.N_star_array[:, :,
                           -1].flatten() / self.m0_factor * self.abs_T_fac  # N0star in T and X

        # mu with m>=m0 forecast
        # self.mu_HE07_m0_gpetas = (self.mu_res_obj.mu_xprime_norm.T * self.Nstar).T
        self.mu_HE07_m0_gpetas = (
                np.tile(self.mu_res_obj.mu_xprime_norm, (pred_obj_1D.Ksim_per_sample, 1)).T * self.Nstar).T
        self.mu_HE07_m0_mle = (np.reshape(self.mu_res_obj_mle.mu_xprime_norm, [-1, 1]) * self.Nstar_mle).T
        self.mu_HE07_m0_sim_ref = ((np.reshape(self.HE07_mu_x_ua_m0_xprime_norm, [-1, 1])) * self.HE07_Nstar_m0_sim).T

        # kde for marginals of Nstar
        bw_method = 'silverman'
        self.bw_method = bw_method
        self.kde_Nstar = sc.stats.gaussian_kde(dataset=self.Nstar, bw_method=bw_method, weights=None)
        self.kde_Nstar_HE07 = sc.stats.gaussian_kde(dataset=self.HE07_Nstar_m0_sim, bw_method=bw_method, weights=None)
        if mle_obj is not None:
            self.kde_Nstar_mle = sc.stats.gaussian_kde(dataset=self.Nstar_mle, bw_method=bw_method, weights=None)
        # N0 kde
        self.kde_N0_star = sc.stats.gaussian_kde(dataset=self.N0_star, bw_method=bw_method, weights=None)
        self.kde_N0_star_mle = sc.stats.gaussian_kde(dataset=self.N0_star_mle, bw_method=bw_method, weights=None)

        # test data
        if data_star is None:
            tau1 = pred_obj_1D.tau1
            tau2 = pred_obj_1D.tau2
            data_obj = self.save_obj_GS['data_obj']
            idx = np.where((data_obj.data_all.times >= tau1) & (data_obj.data_all.times <= tau2))
            data_star = np.empty([len(data_obj.data_all.times[idx]), 4])
            data_star[:, 0] = np.copy(data_obj.data_all.times[idx])
            data_star[:, 1] = np.copy(data_obj.data_all.magnitudes[idx])
            data_star[:, 2] = np.copy(data_obj.data_all.positions[idx, 0])
            data_star[:, 3] = np.copy(data_obj.data_all.positions[idx, 1])
        self.N_obs = len(data_star)
        print(self.N_obs)
        self.data_star = data_star

        # Poisson likelihood for NHPP
        Ksim = len(self.mu_HE07_m0_gpetas)
        loglike_gpetas = []
        mu_xi_gpetas = []
        integral_part_gpetas = []
        loglike_HE07_ref = []
        mu_xi_HE07_ref = []
        integral_part_HE07_ref = []
        loglike_mle = []
        mu_xi_mle = []
        integral_part_mle = []
        X_grid = self.HE07_X_grid
        data_star = self.data_star
        for i in range(Ksim):
            # gpetas
            mu_grid = self.mu_HE07_m0_gpetas[i]
            Nstar_in_absT = self.Nstar[i]
            loglike, mu_xi, integral_part = self.poisson_likelihood(mu_grid, X_grid, Nstar_in_absT, data_star, X_borders=self.X_borders)
            loglike_gpetas.append(loglike)
            mu_xi_gpetas.append(mu_xi)
            integral_part_gpetas.append(integral_part)

            # HE07 reference
            mu_grid = self.mu_HE07_m0_sim_ref[i]
            Nstar_in_absT = self.HE07_Nstar_m0_sim[i]
            loglike, mu_xi, integral_part = self.poisson_likelihood(mu_grid, X_grid, Nstar_in_absT, data_star, X_borders=self.X_borders)
            loglike_HE07_ref.append(loglike)
            mu_xi_HE07_ref.append(mu_xi)
            integral_part_HE07_ref.append(integral_part)

            # mle-obj
            if mle_obj is not None:
                mu_grid = self.mu_HE07_m0_mle[i]
                Nstar_in_absT = self.Nstar_mle[i]
                loglike, mu_xi, integral_part = self.poisson_likelihood(mu_grid, X_grid, Nstar_in_absT, data_star, X_borders=self.X_borders)
                loglike_mle.append(loglike)
                mu_xi_mle.append(mu_xi)
                integral_part_mle.append(integral_part)

        self.loglike_gpetas = loglike_gpetas
        self.loglike_HE07_ref = loglike_HE07_ref
        self.loglike_mle = loglike_mle
        self.loglike_mu_xi_gpetas = mu_xi_gpetas
        self.loglike_mu_xi_HE07_ref = mu_xi_HE07_ref
        self.loglike_mu_xi_mle = mu_xi_mle
        self.loglike_integral_part_gpetas = integral_part_gpetas
        self.loglike_integral_part_HE07_ref = integral_part_HE07_ref
        self.loglike_integral_part_mle = integral_part_mle


        # log(E[L])
        self.log_E_L_gpetas = logsumexp(self.loglike_gpetas) - np.log(len(self.loglike_gpetas))
        self.log_E_L_HE07_ref = logsumexp(self.loglike_HE07_ref) - np.log(len(self.loglike_HE07_ref))
        self.log_E_L_mle = logsumexp(self.loglike_mle) - np.log(len(self.loglike_mle))

        # approx branching ratio: n \approx N_varphi/N = (N-N_0)/N
        self.n_branching_approx_gpetas = (self.Nstar - self.N0_star) / self.Nstar
        self.n_branching_approx_mle = (self.Nstar_mle - self.N0_star_mle) / self.Nstar_mle

    def poisson_likelihood(self, mu_grid, X_grid, Nstar_in_absT, data_star, X_borders=None):
        mu_xi = gpetas.some_fun.mu_xprime_interpol(xprime=data_star[:, 2:4], mu_grid=mu_grid,
                                                   X_grid=X_grid,
                                                   X_borders=X_borders,
                                                   method=None, print_method=None)
        integral_part = Nstar_in_absT
        log_like = np.sum(np.log(mu_xi)) - integral_part

        return log_like, mu_xi, integral_part


class performance_LTF_HE07_m0():
    def __init__(self, save_obj_GS, pred_obj_1D,
                 forecast_5yrs_495_HE07,
                 sample_idx_vec=None,
                 mle_obj=None, pred_obj_1D_mle=None,
                 data_star=None,
                 abs_T_data_star=None, m_beta_scaling=None):

        self.abs_T_forecast_ref = 5. * 365.25  # 5yrs CSEP convention

        # gpetas
        self.save_obj_GS = save_obj_GS
        self.pred_obj_1D = pred_obj_1D
        self.case_name = str(save_obj_GS['case_name'])
        self.X_borders = self.save_obj_GS['data_obj'].domain.X_borders
        self.abs_X = np.prod(np.diff(self.X_borders))
        self.X_grid = save_obj_GS['X_grid']

        # mle if given
        # self.mle_obj = save_obj_GS
        self.mle_obj = mle_obj
        self.pred_obj_1D_mle = pred_obj_1D_mle

        # Extract HE07 for region X
        self.forecast_5yrs_495_HE07 = forecast_5yrs_495_HE07
        regions = gpetas.R00x_setup.R00x_california_set_domain()
        region = eval('regions.polygon_%s' % (self.case_name))
        x, y, z, xx, yy, idx = gpetas.prediction_1d.new_extract_forecast_original_units(forecast=forecast_5yrs_495_HE07,
                                                                                        region=region, plot_yes=None)
        X_grid_HE07_xy = np.zeros([len(x), 2])
        X_grid_HE07_xy[:, 0] = x
        X_grid_HE07_xy[:, 1] = y
        self.HE07_X_grid = X_grid_HE07_xy
        self.L = len(self.HE07_X_grid)
        self.HE07_mu_x_per_0p1x0p1deg_m495 = np.copy(z)
        self.HE07_mu_x_ua_m495 = self.HE07_mu_x_per_0p1x0p1deg_m495 / (0.1 ** 2.)  # adjusting per unit area 'ua'
        self.m_beta = save_obj_GS['setup_obj'].m_beta
        self.m0 = self.save_obj_GS['data_obj'].domain.m0
        if m_beta_scaling is None:
            self.m_beta_scaling = self.m_beta
        else:
            self.m_beta_scaling = m_beta_scaling
        self.m0_factor = np.exp(self.m_beta_scaling * (4.95 - self.m0))
        self.HE07_mean_Nstar_m0 = np.sum(self.HE07_mu_x_per_0p1x0p1deg_m495) * self.m0_factor
        self.HE07_mu_x_ua_m0 = self.HE07_mu_x_ua_m495 * self.m0_factor
        self.HE07_mu_x_per_0p1x0p1deg_m0 = self.HE07_mu_x_ua_m0 * (0.1) ** 2.
        # normalization
        self.HE07_Zprime_m0 = np.sum(self.HE07_mu_x_ua_m0) * self.abs_X / self.L
        self.HE07_mu_x_ua_m0_xprime_norm = self.HE07_mu_x_ua_m0 / self.HE07_Zprime_m0

        # mu resolution
        # gpetas
        K_samples_total = len(save_obj_GS['lambda_bar'])
        if sample_idx_vec is None:
            n = int(self.pred_obj_1D.Ksim_total / self.pred_obj_1D.Ksamples_total)
            vector = np.arange(0, self.pred_obj_1D.Ksamples_total, 1)
            # Repeat the vector n times
            sample_idx_vec = np.tile(vector, n)
            # sample_idx_vec = np.arange(0, K_samples_total, 1)
        self.sample_idx_vec = sample_idx_vec
        self.mu_res_obj = gpetas.prediction_1d.resolution_mu_gpetas(save_obj_GS,
                                                                    X_grid_prime=self.HE07_X_grid,
                                                                    sample_idx_vec=self.sample_idx_vec,
                                                                    summary=None)
        # mle
        if mle_obj is not None:
            self.mu_res_obj_mle = gpetas.prediction_1d.resolution_mu_mle(mle_obj, X_grid_prime=self.HE07_X_grid)

        # Nstar forecasts for abs_T in X
        if abs_T_data_star is None:
            abs_T_data_star = 5 * 365.25  # CSEP forecast window 5yrs
        self.abs_T_data_star = abs_T_data_star
        self.abs_T_fac = self.abs_T_data_star / self.abs_T_forecast_ref
        self.Nstar = self.pred_obj_1D.N_star_array[:, :, 2].flatten() * self.abs_T_fac  # N in T and X
        self.sample_idx_vec_all_sim = self.pred_obj_1D.N_star_array[:, :, 3].flatten()
        self.Nstar_mle = self.pred_obj_1D_mle.N_star_array[:, :, 2].flatten() * self.abs_T_fac
        self.HE07_Nstar_m0_sim = np.random.poisson(lam=self.HE07_mean_Nstar_m0 * self.abs_T_fac, size=len(self.Nstar))
        self.N0_star = self.pred_obj_1D.N_star_array[:, :, -1].flatten() * self.abs_T_fac  # N0star in T and X
        self.N0_star_mle = self.pred_obj_1D_mle.N_star_array[:, :, -1].flatten() * self.abs_T_fac  # N0star in T and X

        # mu with m>=m0 forecast
        # self.mu_HE07_m0_gpetas = (self.mu_res_obj.mu_xprime_norm.T * self.Nstar).T
        self.mu_HE07_m0_gpetas = (
                np.tile(self.mu_res_obj.mu_xprime_norm, (pred_obj_1D.Ksim_per_sample, 1)).T * self.Nstar).T
        self.mu_HE07_m0_mle = (np.reshape(self.mu_res_obj_mle.mu_xprime_norm, [-1, 1]) * self.Nstar_mle).T
        self.mu_HE07_m0_sim_ref = ((np.reshape(self.HE07_mu_x_ua_m0_xprime_norm, [-1, 1])) * self.HE07_Nstar_m0_sim).T

        # kde for marginals of Nstar
        bw_method = 'silverman'
        self.bw_method = bw_method
        self.kde_Nstar = sc.stats.gaussian_kde(dataset=self.Nstar, bw_method=bw_method, weights=None)
        self.kde_Nstar_HE07 = sc.stats.gaussian_kde(dataset=self.HE07_Nstar_m0_sim, bw_method=bw_method, weights=None)
        if mle_obj is not None:
            self.kde_Nstar_mle = sc.stats.gaussian_kde(dataset=self.Nstar_mle, bw_method=bw_method, weights=None)
        # N0 kde
        self.kde_N0_star = sc.stats.gaussian_kde(dataset=self.N0_star, bw_method=bw_method, weights=None)
        self.kde_N0_star_mle = sc.stats.gaussian_kde(dataset=self.N0_star_mle, bw_method=bw_method, weights=None)

        # test data
        if data_star is None:
            tau1 = pred_obj_1D.tau1  # todo might be wrong times
            tau2 = pred_obj_1D.tau2  # todo
            data_obj = self.save_obj_GS['data_obj']
            idx = np.where((data_obj.data_all.times >= tau1) & (data_obj.data_all.times <= tau2))
            data_star = np.empty([len(data_obj.data_all.times[idx]), 4])
            data_star[:, 0] = np.copy(data_obj.data_all.times[idx])
            data_star[:, 1] = np.copy(data_obj.data_all.magnitudes[idx])
            data_star[:, 2] = np.copy(data_obj.data_all.positions[idx, 0])
            data_star[:, 3] = np.copy(data_obj.data_all.positions[idx, 1])
        self.N_obs = len(data_star)
        print(self.N_obs)
        self.data_star = data_star

        # Poisson likelihood for NHPP
        Ksim = len(self.mu_HE07_m0_gpetas)
        loglike_gpetas = []
        mu_xi_gpetas = []
        integral_part_gpetas = []
        loglike_HE07_ref = []
        mu_xi_HE07_ref = []
        integral_part_HE07_ref = []
        loglike_mle = []
        mu_xi_mle = []
        integral_part_mle = []
        X_grid = self.HE07_X_grid
        data_star = self.data_star
        for i in range(Ksim):
            # gpetas
            mu_grid = self.mu_HE07_m0_gpetas[i]
            Nstar_in_absT = self.Nstar[i]
            loglike, mu_xi, integral_part = self.poisson_likelihood(mu_grid, X_grid, Nstar_in_absT, data_star,
                                                                    X_borders=self.X_borders)
            loglike_gpetas.append(loglike)
            mu_xi_gpetas.append(mu_xi)
            integral_part_gpetas.append(integral_part)

            # HE07 reference
            mu_grid = self.mu_HE07_m0_sim_ref[i]
            Nstar_in_absT = self.HE07_Nstar_m0_sim[i]
            loglike, mu_xi, integral_part = self.poisson_likelihood(mu_grid, X_grid, Nstar_in_absT, data_star,
                                                                    X_borders=self.X_borders)
            loglike_HE07_ref.append(loglike)
            mu_xi_HE07_ref.append(mu_xi)
            integral_part_HE07_ref.append(integral_part)

            # mle-obj
            if mle_obj is not None:
                mu_grid = self.mu_HE07_m0_mle[i]
                Nstar_in_absT = self.Nstar_mle[i]
                loglike, mu_xi, integral_part = self.poisson_likelihood(mu_grid, X_grid, Nstar_in_absT, data_star,
                                                                        X_borders=self.X_borders)
                loglike_mle.append(loglike)
                mu_xi_mle.append(mu_xi)
                integral_part_mle.append(integral_part)

        self.loglike_gpetas = loglike_gpetas
        self.loglike_HE07_ref = loglike_HE07_ref
        self.loglike_mle = loglike_mle

        self.loglike_mu_xi_gpetas = mu_xi_gpetas
        self.loglike_mu_xi_HE07_ref = mu_xi_HE07_ref
        self.loglike_mu_xi_mle = mu_xi_mle
        self.loglike_integral_part_gpetas = integral_part_gpetas
        self.loglike_integral_part_HE07_ref = integral_part_HE07_ref
        self.loglike_integral_part_mle = integral_part_mle


        # log(E[L])
        self.log_E_L_gpetas = logsumexp(self.loglike_gpetas) - np.log(len(self.loglike_gpetas))
        self.log_E_L_HE07_ref = logsumexp(self.loglike_HE07_ref) - np.log(len(self.loglike_HE07_ref))
        self.log_E_L_mle = logsumexp(self.loglike_mle) - np.log(len(self.loglike_mle))

        # approx branching ratio: n \approx N_varphi/N = (N-N_0)/N
        self.n_branching_approx_gpetas = (self.Nstar-self.N0_star)/self.Nstar
        self.n_branching_approx_mle = (self.Nstar_mle - self.N0_star_mle) / self.Nstar_mle

    def poisson_likelihood(self, mu_grid, X_grid, Nstar_in_absT, data_star, X_borders=None):
        mu_xi = gpetas.some_fun.mu_xprime_interpol(xprime=data_star[:, 2:4], mu_grid=mu_grid,
                                                   X_grid=X_grid,
                                                   X_borders=X_borders,
                                                   method=None, print_method=None)
        integral_part = Nstar_in_absT
        log_like = np.sum(np.log(mu_xi)) - integral_part

        return log_like, mu_xi, integral_part


class resolution_mu_mle:
    def __init__(self, mle_obj, X_grid_prime=None):
        X_grid = mle_obj.X_grid
        X_borders = mle_obj.data_obj.domain.X_borders
        self.X_grid = X_grid
        self.x = X_grid
        self.L = len(self.X_grid)
        self.abs_X = np.prod(np.diff(mle_obj.data_obj.domain.X_borders))
        self.X_borders = X_borders
        self.mle_obj = mle_obj
        self.X_grid_prime = X_grid_prime
        if X_grid_prime is None:
            self.Lprime = np.copy(self.L)
            xprime = X_borders
            mu_xprime = np.copy(mle_obj.mu_grid)
            print('No new resolution: same output mu as input mu')
        else:
            self.Lprime = len(X_grid_prime)
            xprime = X_grid_prime
            mu_x = mle_obj.mu_grid
            mu = gpetas.some_fun.mu_xprime_interpol(xprime, mu_x, X_grid, X_borders, method=None)
            # mu=mle_obj.eval_kde_xprime(X_grid_HE07_xy)
            norm_fac = self.Lprime / self.L * np.sum(mu_x) / np.sum(mu)
            mu_xprime = mu * norm_fac
        self.mu_xprime = mu_xprime
        self.xprime = xprime

        # normalization
        self.Zprime = self.abs_X / self.Lprime * np.sum(self.mu_xprime)
        self.mu_xprime_norm = (self.mu_xprime.T / self.Zprime).T
        self.L = len(self.X_grid)
        self.Z = self.abs_X / self.L * np.sum(mle_obj.mu_grid)
        self.mu_x = mle_obj.mu_grid
        self.mu_x_norm = (self.mu_x.T / self.Z).T


class resolution_mu_gpetas:
    def __init__(self, save_obj_GS, X_grid_prime=None, sample_idx_vec=None, summary=None):
        X_grid = save_obj_GS['X_grid']
        X_borders = save_obj_GS['data_obj'].domain.X_borders
        K_samples_total = len(save_obj_GS['lambda_bar'])
        self.X_grid = X_grid
        self.L = len(self.X_grid)
        self.abs_X = np.prod(np.diff(save_obj_GS['data_obj'].domain.X_borders))
        self.x = X_grid
        self.X_borders = X_borders
        self.K_samples_total = K_samples_total
        self.save_obj_GS = save_obj_GS
        self.X_grid_prime = X_grid_prime
        self.sample_idx_vec = sample_idx_vec
        self.summary = summary
        if sample_idx_vec is None:
            sample_idx_vec = np.array([K_samples_total - 1])
            if summary is None:
                print('No sample_idx_vec is given. Posterior sample k=%i is taken.' % sample_idx_vec)
        self.sample_idx_vec = sample_idx_vec
        if X_grid_prime is None:
            self.Lprime = np.copy(self.L)
            xprime = X_grid
            mu_xprime = np.empty([len(sample_idx_vec), len(xprime)]) * np.nan
            mu_x_arr = np.empty([len(sample_idx_vec), len(X_grid)]) * np.nan
            for i in range(len(sample_idx_vec)):
                k = sample_idx_vec[i]
                mu_gpetas_k = np.copy(save_obj_GS['mu_grid'][int(k)])
                mu = gpetas.some_fun.mu_xprime_gpetas(xprime, mu_gpetas_k, X_grid, X_borders, method=None,
                                                      lambda_bar=None, cov_params=None)
                norm_fac = self.Lprime / self.L * np.sum(mu_gpetas_k) / np.sum(mu)
                mu_xprime[i, :] = mu * norm_fac
                mu_x_arr[i, :] = mu_gpetas_k
            # mu_xprime = np.copy(save_obj_GS['mu_grid'])
            print('No new resolution: same output mu as input mu')
        else:
            self.Lprime = len(X_grid_prime)
            xprime = X_grid_prime
            if summary == 'mean':
                mu_gpetas_k = np.mean(save_obj_GS['mu_grid'], axis=0)
                mu = gpetas.some_fun.mu_xprime_gpetas(xprime, mu_gpetas_k, X_grid, X_borders, method=None,
                                                      lambda_bar=None, cov_params=None)
                self.sample_idx_vec = None
                self.summary = 'mean'
                norm_fac = self.Lprime / self.L * np.sum(mu_gpetas_k) / np.sum(mu)
                mu_xprime = mu * norm_fac
            if summary == 'median':
                mu_gpetas_k = np.median(save_obj_GS['mu_grid'], axis=0)
                mu = gpetas.some_fun.mu_xprime_gpetas(xprime, mu_gpetas_k, X_grid, X_borders, method=None,
                                                      lambda_bar=None, cov_params=None)
                self.sample_idx_vec = None
                self.summary = 'median'
                norm_fac = self.Lprime / self.L * np.sum(mu_gpetas_k) / np.sum(mu)
                mu_xprime = mu * norm_fac
            else:
                mu_xprime = np.empty([len(sample_idx_vec), len(xprime)]) * np.nan
                mu_x_arr = np.empty([len(sample_idx_vec), len(X_grid)]) * np.nan
                for i in range(len(sample_idx_vec)):
                    k = sample_idx_vec[i]
                    mu_gpetas_k = np.copy(save_obj_GS['mu_grid'][int(k)])
                    mu = gpetas.some_fun.mu_xprime_gpetas(xprime, mu_gpetas_k, X_grid, X_borders, method=None,
                                                          lambda_bar=None, cov_params=None)
                    norm_fac = self.Lprime / self.L * np.sum(mu_gpetas_k) / np.sum(mu)
                    mu_xprime[i, :] = mu * norm_fac
                    mu_x_arr[i, :] = mu_gpetas_k
        self.mu_xprime = mu_xprime
        self.xprime = xprime

        # normalization
        self.Zprime = self.abs_X / self.Lprime * np.sum(self.mu_xprime, axis=1)
        self.mu_xprime_norm = (self.mu_xprime.T / self.Zprime).T
        self.L = len(self.X_grid)
        # self.Z = self.abs_X / self.L * np.sum(save_obj_GS['mu_grid'], axis=1)
        self.Z = self.abs_X / self.L * np.sum(mu_x_arr, axis=1)
        self.mu_x = mu_x_arr  # np.array(save_obj_GS['mu_grid'])
        self.mu_x_norm = (self.mu_x.T / self.Z).T


def new_extract_forecast_original_units(forecast, region, plot_yes=None):
    '''
    Uses midpoints saved in the region object

    '''
    x_bool = np.logical_and(forecast.region.midpoints()[:, 0] > np.min(region[:, 0]),
                            forecast.region.midpoints()[:, 0] < np.max(region[:, 0]))
    y_bool = np.logical_and(forecast.region.midpoints()[:, 1] > np.min(region[:, 1]),
                            forecast.region.midpoints()[:, 1] < np.max(region[:, 1]))
    midpoints = forecast.region.midpoints()[x_bool * y_bool]

    idx_forecast_data = forecast.get_index_of(lons=midpoints[:, 0], lats=midpoints[:, 1])
    mu_forecast_mag_gt495 = np.sum(forecast.data[idx_forecast_data], axis=1)
    x = midpoints[:, 0]
    y = midpoints[:, 1]

    x_sorted = np.sort(midpoints[:, 0])
    y_sorted = np.sort(midpoints[:, 1])
    xx, yy = np.meshgrid(x_sorted, y_sorted)
    try:
        idx_forecast_data_long = forecast.get_index_of(lons=xx, lats=yy)
        mu_forecast_mag_gt495_long = np.sum(forecast.data[idx_forecast_data_long], axis=2)
        if plot_yes is not None:
            real_x = np.unique(x)
            real_y = np.unique(y)
            dx = (real_x[1] - real_x[0]) / 2.
            dy = (real_y[1] - real_y[0]) / 2.
            extent = [real_x[0] - dx, real_x[-1] + dx, real_y[0] - dy, real_y[-1] + dy]
            hf1 = plt.figure(figsize=(20, 20))
            plt.plot(forecast.region.midpoints()[:, 0], forecast.region.midpoints()[:, 1], 'k.', zorder=-2)
            plt.imshow(np.log10(mu_forecast_mag_gt495_long), origin='lower', extent=extent)
            plt.plot(region[:, 0], region[:, 1], 'r', linewidth=3)
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.show()
    except ValueError:
        print('Region has grid cells outside CSEP California test site region.')
        if plot_yes is not None:
            hf1 = plt.figure()
            plt.scatter(x, y, c=np.log10(np.sum(forecast.data[x_bool * y_bool], axis=1)))
            plt.axis('square')
            plt.plot(region[:, 0], region[:, 1], '--k', linewidth=1)
            plt.show()

    return x, y, mu_forecast_mag_gt495, xx, yy, idx_forecast_data


# new 1D implementation
def simulation(obj, mu, theta_off):
    # Ht and b and Xdomain
    tt, mm, la, lo, T, Lat, Lon, Mmin, Mmax, b = obj.general_params_Ht

    # theta offspring
    K, c, p, m_alpha, d, gamma, q = theta_off
    alpha = m_alpha / np.log(10)
    D = np.copy(d)

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
        """
        Simulates (forecast) Nstar number of events in T and X assuming a static, homogeneous background rate (named 1D).
        :param save_obj_GS: if None and mle_obj=mle_obj prediction is for mle case.
        :type save_obj_GS:
        :param tau1:
        :type tau1:
        :param tau2:
        :type tau2:
        :param tau0_Ht:
        :type tau0_Ht:
        :param Ksim_per_sample:
        :type Ksim_per_sample:
        :param m_max:
        :type m_max:
        :param sample_idx_vec:
        :type sample_idx_vec:
        :param seed:
        :type seed:
        :param approx:
        :type approx:
        :param randomized_samples:
        :type randomized_samples:
        :param Bayesian_m_beta:
        :type Bayesian_m_beta:
        :param mle_obj:
        :type mle_obj:
        """

        # mle
        self.mle_obj = mle_obj

        # data
        if save_obj_GS is None:
            self.data_obj = mle_obj.data_obj
        else:
            self.data_obj = save_obj_GS['data_obj']
        self.case_name = self.data_obj.case_name

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
            print('\n MLE mode: equivalent to 1 sample case')
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
        self.absT_star = tau2 - tau1
        self.absT_HE07 = 5. * 365.25

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
                N_star_array[k_sim, k_sample, 0] = N_star_array[k_sim, k_sample, 0] * self.absT_HE07 / self.absT_star
                N_star_array[k_sim, k_sample, 1] = N_star_array[k_sim, k_sample, 1] * self.absT_HE07 / self.absT_star
                N_star_array[k_sim, k_sample, 2] = N_star_array[k_sim, k_sample, 2] * self.absT_HE07 / self.absT_star
                N_star_array[k_sim, k_sample, 3] = k
                N_star_array[k_sim, k_sample, 4] = k_sim
                N_star_array[k_sim, k_sample, 5] = N_0_Tstar
            sys.stdout.write('\r' + str('\t simulation %3d/%d: N=%d. N0=%d\r.' % (
                k_sim + 1, self.Ksim_per_sample, N_star_array[k_sim, k_sample, 1], N_0_Tstar)))
            sys.stdout.flush()

        self.N_star_array = N_star_array
        self.sample_idx_vec_ksim_all = self.N_star_array[:, :, 3].flatten()

        # save simulations
        init_outdir()
        if save_obj_GS is not None:
            fname = output_dir + "/pred_obj_1D_%s.all" % self.case_name
            file = open(fname, "wb")  # remember to open the file in binary mode
            pickle.dump(self, file)
            file.close()
            print('\n GP-E pred_obj_1D has been created and saved:', fname)
        if (save_obj_GS is None) & (mle_obj is not None):
            fname = output_dir + "/pred_obj_1D_%s_mle.all" % self.case_name
            file = open(fname, "wb")  # remember to open the file in binary mode
            pickle.dump(self, file)
            file.close()
            print('\n E mle pred_obj_1D has been created and saved:', fname)

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
    def __init__(self, save_obj_GS, tau1, tau2, tau0_Ht=0., sample_idx_vec=None, seed=None, approx=None, Ksim=None,
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
                                                                               tau1=self.tau1, tau2=self.tau2,
                                                                               m0=self.m0,
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
