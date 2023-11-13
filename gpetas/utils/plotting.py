import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy as sc
import gpetas



def plot_slice_x(intensity_ensemble, X_grid=None, intensity_1_grid=None, intensity_2_grid=None,
                 xidx=None, quantile=0.05, log10scale='yes', X_borders=None, label_pos='yes',
                 points=None,size_points=None,clim_where=None):
    Ksamples, L = intensity_ensemble.shape
    xbins = int(np.sqrt(L))
    ybins = xbins
    x = np.arange(0, xbins, 1)
    y = np.arange(0, ybins, 1)
    dx = 1
    dy = 1
    if X_grid is not None:
        x = np.unique(X_grid[:, 0].reshape(xbins, -1))
        y = np.unique(X_grid[:, 1].reshape(ybins, -1))
        if X_borders is None:
            X_borders = np.array([[np.min(X_grid[:, 0]), np.max(X_grid[:, 0])],
                                  [np.min(X_grid[:, 1]), np.max(X_grid[:, 1])]])
        dx = np.diff(X_borders[0, :]) / xbins
        dy = np.diff(X_borders[1, :]) / ybins

    # interpolated artifacts
    replace_value = np.min(intensity_ensemble[intensity_ensemble > 0]) / 2.
    intensity_ensemble[intensity_ensemble < 0] = replace_value

    if log10scale is not None:
        z_array = np.log10(intensity_ensemble.reshape(-1, xbins, ybins))
    else:
        z_array = intensity_ensemble.reshape(-1, xbins, ybins)

    if xidx is not None:
        z_slice = np.mean(z_array[:, :, xidx], axis=0).squeeze() # or better median?
        h1_xslice = plt.figure()
        plt.plot(y, z_slice, 'k', linewidth=3,
                 label='GP-E (this study)')
        # print(np.median(z_array[:, :, xidx], axis=0).squeeze().shape)
        plt.fill_between(x=y, y1=np.quantile(z_array[:, :, xidx], 1. - quantile, axis=0).squeeze(),
                         y2=np.quantile(z_array[:, :, xidx], quantile, axis=0).squeeze(),
                         facecolor='gray', alpha=0.5)

        if intensity_1_grid is not None:
            if log10scale is not None:
                z_grid = np.log10(intensity_1_grid.reshape(xbins, -1).T)
            else:
                z_grid = intensity_1_grid.reshape(xbins, -1).T
            plt.plot(y, z_grid[xidx, :].squeeze(), '--r', linewidth=1.5, label='E')
            # zlimits = [0, np.max(mle_obj.mu_grid)]
        if log10scale is not None:
            plt.ylabel('$\\log_{10} \ \\lambda(x,t^\\ast)$')
        else:
            plt.ylabel('$\\lambda(x,t^\\ast)$')
        plt.xlabel('$x_2$, (Lat.)')
        if label_pos is not None:
            print((xidx[0] + 1) * dx + X_borders[0, 0])
            plt.annotate('$x_1=$%.3f' % ((xidx[0] + 1) * dx + X_borders[0, 0]), xy=(1, 0.025),
                         xycoords='axes fraction',
                         horizontalalignment='right',
                         fontsize=16)
        #hl = plt.legend(fontsize=16)
        hl = plt.legend(fontsize=16, bbox_to_anchor=(0.435, 0.35))
        plt.show()

        # zoom into max
        if log10scale is not None:
            z_array = intensity_ensemble.reshape(-1, xbins, ybins)
            z_slice = np.mean(z_array[:, :, xidx], axis=0).squeeze()
        h1a_xslicezoommax = plt.figure()
        yidx_max = np.argmax(z_slice)
        n = 3
        plt.plot(y[yidx_max - n:yidx_max + n], z_slice[yidx_max - n:yidx_max + n], 'k', linewidth=3,
                 label='GP-ETAS (this study)')
        y1 = np.quantile(z_array[:, :, xidx], 1. - quantile, axis=0).squeeze()
        y2 = np.quantile(z_array[:, :, xidx], quantile, axis=0).squeeze()
        plt.fill_between(x=y[yidx_max - n:yidx_max + n], y1=y1[yidx_max - n:yidx_max + n],
                         y2=y2[yidx_max - n:yidx_max + n],
                         facecolor='gray', alpha=0.5)



        if intensity_1_grid is not None:
            if log10scale is not None:
                z_grid = intensity_1_grid.reshape(xbins, -1).T
                z_slice_2 = z_grid[xidx, :].squeeze()
            plt.plot(y[yidx_max - n:yidx_max + n],
                     z_slice_2[yidx_max - n:yidx_max + n], '--r', linewidth=1.5, label='ETAS classical')

        plt.ylabel('$\\lambda(x,t^\\ast)$')
        plt.xlabel('$x_2$, (Lat.)')
        plt.show()

        # where are the slices
        h3_where = gpetas.plotting.plot_intensity_2d(intensity_grid=
                                                     np.log10(np.mean(intensity_ensemble, axis=0)),
                                                     cb_label='$\\log_{10} \ {\\rm E}[\\lambda(x,t^\\ast)]$',
                                                     X_grid=X_grid,
                                                     cb_format='%.1f',
                                                     clim=clim_where)
        # if intensity_1_grid is not None:
        #    h3_where = gpetas.plotting.plot_intensity_2d(intensity_grid=np.log10(intensity_1_grid),
        #                                             X_grid=X_grid)
        ax = h3_where.gca()
        ax.axvline(x=(xidx[0]+1.) * dx + X_borders[0, 0], color='w', linestyle='--')

        if points is not None:
            if size_points is None:
                size_points = 20
            plt.scatter(points[:, 0], points[:, 1], s=size_points, c='red')  # s=10

        return h1_xslice, h1a_xslicezoommax, h3_where


def plot_priors_offspring(save_obj_GS):
    h1 = None
    if hasattr(save_obj_GS['setup_obj'], 'prior_theta_dist'):
        prior_theta_params = save_obj_GS['setup_obj'].prior_theta_params
        prior_theta_dist = save_obj_GS['setup_obj'].prior_theta_dist
        # uniform priors
        if prior_theta_dist == 'uniform':
            if save_obj_GS['setup_obj'].spatial_offspring == 'R':
                params = ('$K$', '$c$', '$p$', '$\\alpha_m$', '$d$', '$\\gamma$', '$q$')
                h1 = plt.figure(figsize=(15, 7))
                for i in range(len(prior_theta_params[:, 0])):
                    plt.subplot(2, 4, i + 1)
                    x = np.linspace(0.8 * prior_theta_params[i, 0], 1.2 * prior_theta_params[i, 1], 1000)
                    u = sc.stats.uniform(loc=prior_theta_params[i, 0],
                                         scale=prior_theta_params[i, 1] - prior_theta_params[i, 0])
                    plt.plot(x, u.pdf(x), linewidth=5, label=params[i])
                    plt.gca().get_yaxis().set_ticks([])
                    plt.legend()
                    plt.text(0.4, 0.4, '$U(%.1f,%.1f)$' % (prior_theta_params[i, 0], prior_theta_params[i, 1]),
                             horizontalalignment='center', verticalalignment='center',
                             transform=plt.gca().transAxes)
        if prior_theta_dist == 'gamma':
            if save_obj_GS['setup_obj'].spatial_offspring == 'R':
                params = ('$K$', '$c$', '$p$', '$\\alpha_m$', '$d$', '$\\gamma$', '$q$')
                h1 = plt.figure(figsize=(15, 7))
                for i in range(len(prior_theta_params[:, 0])):
                    plt.subplot(2, 4, i + 1)
                    loc = 0.
                    if i == 2:
                        loc = 1.
                    if i == 6:
                        loc = 1.
                    mean_prior, c_prior = prior_theta_params[i, :]
                    alpha_prior = 1. / c_prior ** 2.
                    beta_prior = alpha_prior / mean_prior
                    x = np.linspace(sc.stats.gamma.ppf(0.01, a=alpha_prior, loc=loc, scale=1. / beta_prior),
                                    sc.stats.gamma.ppf(0.99, a=alpha_prior, loc=loc, scale=1. / beta_prior), 100)
                    p_x = sc.stats.gamma.pdf(x, a=alpha_prior, scale=1. / beta_prior, loc=0)
                    plt.plot(x, p_x, label=params[i], linewidth=5)
                    plt.gca().get_yaxis().set_ticks([])
                    plt.legend()
                    ylim = plt.gca().get_ylim()
                    xlim = plt.gca().get_xlim()
                    if not (i == 2 or i == 6):
                        plt.text(xlim[1], 0.5 * ylim[1],
                                 '%s $\\sim \\Gamma(%.1f,%.1f)$\n $m_{\\Gamma}=%.1f,c_{\\Gamma}=%.1f$' % (
                                     params[i], alpha_prior, beta_prior, mean_prior, c_prior),
                                 horizontalalignment='right',
                                 fontsize=16)
                    if i == 2 or i == 6:
                        plt.text(xlim[1], 0.5 * ylim[1],
                                 '%s-1 $\\sim \\Gamma(%.1f,%.1f)$\n $m_{\\Gamma}=%.1f,c_{\\Gamma}=%.1f$' % (
                                     params[i], alpha_prior, beta_prior, mean_prior, c_prior),
                                 horizontalalignment='right',
                                 fontsize=16)

    return h1


def plot_l_ltest(save_obj_GS, mle_obj=None, mle_obj_silverman=None, t1=None, t2=None, idx_samples=None,
                 method_posterior_GP=None, table_yes=None, fout_table_dir=None):
    '''
    Plots test likelihood for test data.
    :param save_obj_GS:
    :type save_obj_GS:
    :param mle_obj:
    :type mle_obj:
    :param mle_obj_silverman:
    :type mle_obj_silverman:
    :param t1:
    :type t1:
    :param t2:
    :type t2:
    :param idx_samples:
    :type idx_samples:
    :param method_posterior_GP:
    :type method_posterior_GP:
    :param table_yes:
    :type table_yes:
    :return:
    :rtype:
    '''
    if t1 is None:
        t1 = save_obj_GS['data_obj'].domain.T_borders_testing[0]
    if t2 is None:
        t2 = save_obj_GS['data_obj'].domain.T_borders_testing[1]

    # eval l
    testing_periods = np.zeros([2, 2]) * np.nan
    testing_periods[0, :] = save_obj_GS['data_obj'].domain.T_borders_training
    testing_periods[1, :] = np.array([t1, t2])

    # gpetas
    # method_posterior_GP = 'sparse'
    if idx_samples is None:
        idx_samples = np.arange(0, len(save_obj_GS['lambda_bar']))
    l_values = gpetas.loglike.test_likelihood_GS(save_obj_GS=save_obj_GS,
                                                 testing_periods=testing_periods,
                                                 data_obj=save_obj_GS['data_obj'],
                                                 idx_samples=idx_samples,
                                                 method_posterior_GP=method_posterior_GP,
                                                 method_integral=None)
    print('===============Numbers==============================================')
    print('gpetas:', l_values.l_test_GPetas_log_E_L, 'Events:', l_values.Ntest_arr)

    # mle
    if mle_obj is not None:
        data_obj = save_obj_GS['data_obj']  # mle_obj.data_obj
        mu_xi_at_all_data = mle_obj.eval_kde_xprime(data_obj.data_all.positions)
        l_mle_training = gpetas.loglike.eval_lnl(data_obj=data_obj,
                                                 mu_xi_at_all_data=mu_xi_at_all_data,
                                                 integral_mu_x_unit_time=np.sum(mle_obj.p_i_vec) / np.diff(
                                                     mle_obj.data_obj.domain.T_borders_training),
                                                 theta_phi__Kcpadgq=mle_obj.theta_mle_Kcpadgq,
                                                 m0=mle_obj.data_obj.domain.m0,
                                                 X_borders_eval_l=mle_obj.data_obj.domain.X_borders,
                                                 T_borders_eval_l=testing_periods[0, :],
                                                 spatial_kernel='R')

        l_mle_testing = gpetas.loglike.eval_lnl(data_obj=data_obj,
                                                mu_xi_at_all_data=mu_xi_at_all_data,
                                                integral_mu_x_unit_time=np.sum(mle_obj.p_i_vec) / np.diff(
                                                    mle_obj.data_obj.domain.T_borders_training),
                                                theta_phi__Kcpadgq=mle_obj.theta_mle_Kcpadgq,
                                                m0=mle_obj.data_obj.domain.m0,
                                                X_borders_eval_l=mle_obj.data_obj.domain.X_borders,
                                                T_borders_eval_l=testing_periods[1, :],
                                                spatial_kernel='R')
        print('MLE:', l_mle_training.lnl_value, l_mle_testing.lnl_value, 'Events:',
              l_mle_training.N_lnl_eval, l_mle_testing.N_lnl_eval)

    if mle_obj_silverman is not None:
        data_obj = save_obj_GS['data_obj']  # mle_obj.data_obj
        mu_xi_at_all_data = mle_obj_silverman.eval_kde_xprime(data_obj.data_all.positions)
        l_mle_training_silverman = gpetas.loglike.eval_lnl(data_obj=data_obj,
                                                           mu_xi_at_all_data=mu_xi_at_all_data,
                                                           integral_mu_x_unit_time=np.sum(
                                                               mle_obj_silverman.p_i_vec) / np.diff(
                                                               mle_obj_silverman.data_obj.domain.T_borders_training),
                                                           theta_phi__Kcpadgq=mle_obj_silverman.theta_mle_Kcpadgq,
                                                           m0=mle_obj_silverman.data_obj.domain.m0,
                                                           X_borders_eval_l=mle_obj_silverman.data_obj.domain.X_borders,
                                                           T_borders_eval_l=testing_periods[0, :],
                                                           spatial_kernel='R')

        l_mle_testing_silverman = gpetas.loglike.eval_lnl(data_obj=data_obj,
                                                          mu_xi_at_all_data=mu_xi_at_all_data,
                                                          integral_mu_x_unit_time=np.sum(
                                                              mle_obj_silverman.p_i_vec) / np.diff(
                                                              mle_obj_silverman.data_obj.domain.T_borders_training),
                                                          theta_phi__Kcpadgq=mle_obj_silverman.theta_mle_Kcpadgq,
                                                          m0=mle_obj_silverman.data_obj.domain.m0,
                                                          X_borders_eval_l=mle_obj_silverman.data_obj.domain.X_borders,
                                                          T_borders_eval_l=testing_periods[1, :],
                                                          spatial_kernel='R')
        print('MLE:', l_mle_training_silverman.lnl_value, l_mle_testing_silverman.lnl_value, 'Events:',
              l_mle_training_silverman.N_lnl_eval, l_mle_testing_silverman.N_lnl_eval)

    h1 = plt.figure(figsize=(10, 7.5))
    plt.subplot(211)
    plt.plot(l_values.lnl_samples_mat[0, :], '.k', label='$\\ell_{\\rm \\mathcal{D},post}$')
    plt.axhline(y=l_values.l_test_GPetas_log_E_L[0], color='k', linestyle='--',
                label='log E[$L_{\\rm \\mathcal{D},post}$]')
    if mle_obj is not None:
        plt.axhline(y=l_mle_training.lnl_value, color='b', label='$\\ell_{\\rm \\mathcal{D},mle}$')
    if mle_obj_silverman is not None:
        plt.axhline(y=l_mle_training_silverman.lnl_value, color='b', linestyle='--',
                    label='$\\ell_{\\rm \\mathcal{D},mle_{\\rm Silverman}}$')
    plt.ylabel('$\\ell_{\\rm training}$')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.subplot(212)
    plt.plot(l_values.lnl_samples_mat[1, :], '.k', label='$\\ell_{\\rm test,post}$')
    plt.axhline(y=l_values.l_test_GPetas_log_E_L[1], color='k', linestyle='--', label='log E[$L_{\\rm test,post}$]')
    if mle_obj is not None:
        plt.axhline(y=l_mle_testing.lnl_value, color='b', label='$\\ell_{\\rm test,mle}$')
    if mle_obj_silverman is not None:
        plt.axhline(y=l_mle_testing_silverman.lnl_value, color='b', linestyle='--',
                    label='$\\ell_{\\rm test,mle_{\\rm Silverman}}$')
    plt.xlabel('index samples')
    plt.ylabel('$\\ell_{\\rm test}$')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # plt.show()

    h2 = plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(l_values.lnl_samples_mat[0, :], density=True, color='k')
    if mle_obj is not None:
        plt.axvline(x=l_mle_training.lnl_value, color='b')
    if mle_obj_silverman is not None:
        plt.axvline(x=l_mle_training_silverman.lnl_value, color='b', linestyle='--')
    plt.axvline(x=l_values.l_test_GPetas_log_E_L[0], color='k', linestyle='--')
    plt.xlabel('$\\ell$')
    plt.ylabel('density')
    plt.subplot(1, 2, 2)
    plt.hist(l_values.lnl_samples_mat[1, :], density=True, color='k', label='post. samples')
    if mle_obj is not None:
        plt.axvline(x=l_mle_testing.lnl_value, color='b', label='mle')
    if mle_obj_silverman is not None:
        plt.axvline(x=l_mle_testing_silverman.lnl_value, color='b', linestyle='--',
                    label='$\\rm mle_{\\rm Silverman}$')
    plt.axvline(x=l_values.l_test_GPetas_log_E_L[1], color='k', linestyle='--', label='log E[$L_{\\rm post}$]')
    plt.xlabel('$\\ell_{\\rm test}$')
    ax = plt.gca()
    ax.yaxis.tick_right()
    plt.ylabel('density')
    plt.legend(bbox_to_anchor=(1.5, 1), loc=2, borderaxespad=0.)
    # plt.show()

    if table_yes is not None:
        gpetas.some_fun.write_table_l_test_real_data(testing_periods, l_values.Ntest_arr,
                                                     l_test_GP=l_values.l_test_GPetas_log_E_L,
                                                     l_test_kde_default=np.array(
                                                         [l_mle_training.lnl_value, l_mle_testing.lnl_value]),
                                                     l_test_kde_silverman=None,
                                                     fout_dir=fout_table_dir, idx_samples=idx_samples)

    return h1, h2


def plot_1D_estimation(save_obj_GS=None, sample_idx_vec=None, mle_obj=None, mle_obj_silverman=None,
                       quantile=0.05, resolution=None, xlim=None, figsize=(10, 5),
                       pos_textND=None, pos_textNDstar=None, label=None):
    '''
    Computes and plots 1D estimation of ETAS, in time, given history H_t until t
    :param label:
    :type label:
    :param save_obj_GS:
    :type save_obj_GS:
    :param sample_idx_vec:
    :type sample_idx_vec:
    :param mle_obj:
    :type mle_obj:
    :param mle_obj_silverman:
    :type mle_obj_silverman:
    :param quantile:
    :type quantile:
    :param resolution:
    :type resolution:
    :param xlim:
    :type xlim:
    :param figsize:
    :type figsize:
    :param pos_textND:
    :type pos_textND:
    :param pos_textNDstar:
    :type pos_textNDstar:
    :return:
    :rtype:
    '''
    if resolution is None:
        resolution = 1000
    if label is None:
        label = 'GP-E'
    hf1 = None
    if mle_obj is not None:
        data_obj = mle_obj.data_obj
        if xlim is None:
            t_end = data_obj.domain.T_borders_all[1]
        else:
            t_end = xlim[1]
        Lambda_t_mle, t_eval_vec_mle = gpetas.some_fun.Lambda_t_1D(t_end=t_end, mle_obj=mle_obj, resolution=resolution)

    if mle_obj_silverman is not None:
        data_obj = mle_obj_silverman.data_obj
        if xlim is None:
            t_end = data_obj.domain.T_borders_testing[1]
        else:
            t_end = xlim[1]
        Lambda_t_mle_silverman, t_eval_vec_mle_silverman = gpetas.some_fun.Lambda_t_1D(t_end=t_end,
                                                                                       mle_obj=mle_obj_silverman,
                                                                                       resolution=resolution)

    if save_obj_GS is not None:
        data_obj = save_obj_GS['data_obj']
        if xlim is None:
            t_end = data_obj.domain.T_borders_all[1]
        else:
            t_end = xlim[1]
        Lambda_t_gpe, t_eval_vec_gpe = gpetas.some_fun.Lambda_t_1D(t_end=t_end, save_obj_GS=save_obj_GS,
                                                                   sample_idx_vec=sample_idx_vec,
                                                                   resolution=resolution)
        x = np.empty([len(data_obj.data_all.times), 2])
        x[:, 0] = data_obj.data_all.times
        x[:, 1] = data_obj.data_all.magnitudes
        x_label = None
        training_start = data_obj.domain.T_borders_training[0]
        training_end = data_obj.domain.T_borders_training[1]
        idx_test = data_obj.data_all.times > data_obj.domain.T_borders_training[1]
        if pos_textND is None:
            pos_textND = [np.min(x[:, 0]), x.shape[0]]
        if pos_textNDstar is None:
            pos_textNDstar = [training_end, 0]

        hf1 = plt.figure(figsize=figsize)
        plt.tight_layout()
        if training_end is not None:
            plt.axvline(x=training_end, color='r')
        plt.locator_params(nbins=5)
        if xlim is not None:
            plt.locator_params(nbins=3)
        # Lambda(1)
        plt.plot(t_eval_vec_gpe, np.quantile(a=Lambda_t_gpe, q=0.5, axis=1), '-k', label=label, linewidth=3)
        plt.fill_between(t_eval_vec_gpe, y1=np.quantile(a=Lambda_t_gpe, q=quantile, axis=1),
                         y2=np.quantile(a=Lambda_t_gpe, q=1 - quantile, axis=1), color='lightgrey',
                         label='$q_{%.3f,%.3f}$' % (quantile, 1 - quantile))
        # mle: Lambda
        if mle_obj is not None:
            plt.plot(t_eval_vec_mle, Lambda_t_mle, 'b--', label='E', linewidth=3)
        if mle_obj_silverman is not None:
            plt.plot(t_eval_vec_mle_silverman, Lambda_t_mle_silverman, 'b:', label='E-S', linewidth=3)

        # data
        plt.step(np.concatenate([[0.], x[:, 0]]), np.arange(0, x[:, 0].shape[0] + 1, 1), 'm', linewidth=3,
                 label='observed')

        # labels and text
        ylim = plt.gca().get_ylim()
        if xlim is not None:
            plt.xlim([xlim[0], xlim[1]])
        else:
            plt.xlim(data_obj.domain.T_borders_all)
            if np.sum(idx_test) > 0:
                plt.text(0.975 * data_obj.domain.T_borders_training[1], 0.7 * ylim[1],
                         '$N_{\mathcal{D}^{\\ast}} =$ %s\n$m_{\mathcal{D}^{\\ast}}\in$[%.2f,%.2f]'
                         % (x[idx_test].shape[0], np.min(x[idx_test, 1]), np.max(x[idx_test, 1])),
                         horizontalalignment='right',
                         verticalalignment='top', fontsize=20)
            plt.ylabel('counts')
            if x_label is not None:
                plt.xlabel(x_label)
            else:
                plt.xlabel('time, days')
            plt.text(0, 0.975 * ylim[1],
                     '$N_{\mathcal{D}}=$ %s\n$m_{\mathcal{D}}\in[%.2f,%.2f]$'
                     % (x[data_obj.idx_training].shape[0], np.min(x[data_obj.idx_training, 1]),
                        np.max(x[data_obj.idx_training, 1])),
                     horizontalalignment='left',
                     verticalalignment='top', fontsize=20)
        plt.gca().tick_params(direction='out', left=True, right=True, top=True, bottom=True, labelright=False,
                              labelleft=True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=20)
    return hf1


def plot_autocorr_hyperparams(save_obj_GS, idx_samples=None, **kwargs):
    Ksamples = len(save_obj_GS['lambda_bar'])
    if idx_samples is None:
        idx_samples = np.arange(0, Ksamples, 1)
    x = np.empty([Ksamples, 3])
    x[:, 0] = np.array(save_obj_GS['cov_params_nu1'])
    x[:, 1] = np.array(save_obj_GS['cov_params_nu2'])
    x[:, 2] = np.array(save_obj_GS['cov_params_theta']).flatten()
    h1 = plot_autocorr_data(x, varnames=['$\\nu_1$', '$\\nu_2$', '$\\nu_0$'], idx_samples=idx_samples, **kwargs)
    return h1


def plot_autocorr_data(data, idx_samples=None, varnames=None, xlim=None,
                       maxlag=None, label_ESS=None, alpha=None,
                       show_u_level=None, **kwargs_plot):
    # plot definitions
    pSIZE = 20
    plt.rc('font', size=pSIZE)
    plt.rc('axes', titlesize=pSIZE)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    all_samples, numvars = data.shape

    if idx_samples is None:
        idx_samples = np.arange(0, all_samples)
    if maxlag is None:
        maxlag = np.min([100, len(idx_samples)])
    if varnames is None:
        if numvars == 7:
            varnames = ['$K_0$', '$c$', '$p$', '$\\alpha_m$', '$D$', '$\\gamma$', 'q', '$\\bar{\\lambda}$']
            colors[7] = '#000000'
        if numvars == 5:
            varnames = ['$K_0$', '$c$', '$p$', '$\\alpha_m$', '$D_{\\rm gauss}$', '$\\bar{\\lambda}$']
            colors[5] = '#000000'
    x = np.copy(data[idx_samples, :])
    if xlim is None:
        xlim = [-1, maxlag]
    if alpha is None:
        alpha = 0.05
    linewidth = 2
    h1 = plt.figure(figsize=(15, 5))
    h1.subplots_adjust(hspace=0.2, wspace=0.2)
    for i in range(numvars):
        r, u_level = gpetas.some_fun.autocorr(x[:, i], alpha=alpha)
        ax = plt.subplot(2, (numvars + 1) // 2, i + 1)
        plt.vlines(x=np.arange(0, maxlag), ymin=0, ymax=r[0:maxlag], lw=linewidth, color=colors[i])
        plt.axhline(y=0, color='k')
        if show_u_level is not None:
            plt.axhline(y=u_level, color='gray', linestyle='--')
            plt.axhline(y=-u_level, color='gray', linestyle='--')
        plt.ylim([-1, 1])
        plt.xlim([xlim[0], xlim[1]])
        plt.ylabel('correlation')
        if i < numvars // 2:
            ax.xaxis.set_ticklabels([])
        if i >= numvars // 2:
            plt.xlabel('lag')
            # plt.xlabel('lag $\\times 10^3$')
        if (i != 0) and (i != (numvars + 1) // 2):
            ax.yaxis.set_ticklabels([])
            plt.ylabel('')
        plt.locator_params(tight=True, nbins=3)

        if label_ESS is not None:
            Neff = len(x) / (1 + 2 * np.max(np.cumsum(r[1:])))
            maxlag_ESS = np.argmax(np.cumsum(r[1:]))
            ax.annotate('$N_{\\rm eff}^{(%i)}\\approx$%i (%.1f)' % (maxlag_ESS, Neff, Neff / len(x)),
                        (0.75, 0.05), xycoords='axes fraction', ha='center', va='bottom', fontsize=12)
            plt.axvline(x=maxlag_ESS, color='gray', linestyle='--')
        ax.annotate(varnames[i], (0.85, 0.95), xycoords='axes fraction', ha='center', va='top')
    return h1


def plot_hyperparams(save_obj_GS):
    yu = 1.1 * np.max([np.max(save_obj_GS['cov_params_nu1']), np.max(save_obj_GS['cov_params_nu2'])])
    yl = 0.9 * np.min([np.min(save_obj_GS['cov_params_nu1']), np.min(save_obj_GS['cov_params_nu2'])])

    h1 = plt.figure(figsize=(10, 10))
    h1.subplots_adjust(hspace=0.35, wspace=0.1)
    plt.subplot(2, 7, (1, 3))
    plt.plot(save_obj_GS['cov_params_nu1'], 'k')
    plt.xlabel('samples')
    plt.ylabel('$\\nu_1$')
    plt.ylim([yl, yu])
    plt.locator_params(tight=True, nbins=3)
    plt.subplot(2, 7, (8, 10))
    plt.plot(save_obj_GS['cov_params_nu2'], 'k')
    plt.xlabel('samples')
    plt.ylabel('$\\nu_2$')
    plt.ylim([yl, yu])
    plt.locator_params(tight=True, nbins=3)
    plt.subplot(2, 7, (5, 7))
    plt.plot(save_obj_GS['cov_params_theta'], color='darkblue')
    ax = plt.gca()
    ax.yaxis.set_label_position("right")
    ax.tick_params(direction='out', left=False, right=True, top=False, bottom=True, labelright=True, labelleft=False)
    plt.xlabel('samples')
    plt.ylabel('$\\nu_0$')
    plt.locator_params(tight=True, nbins=3)

    ax = plt.subplot(2, 7, 12)
    plt.boxplot(np.array(save_obj_GS['cov_params_nu1']), 'k')
    plt.ylim([yl, yu])
    ax.set_xticklabels(['$\\nu_1$'])
    ax = plt.subplot(2, 7, 13)
    plt.boxplot(np.array(save_obj_GS['cov_params_nu2']), 'k')
    ax.set_yticklabels('')
    ax.set_xticklabels(['$\\nu_2$'])
    plt.ylim([yl, yu])
    ax = plt.subplot(2, 7, 14)
    plt.boxplot(np.array(save_obj_GS['cov_params_theta']), 'k')
    ax.set_xticklabels(['$\\nu_0$'])
    ax.yaxis.tick_right()
    return (h1)


def plot_acceptance_offspring(save_obj_GS):
    """
    Multi proposal MH-MCMC acceptance rate
    :param save_obj_GS:
    :type save_obj_GS:
    :return:
    :rtype:
    """
    h1 = plt.figure(figsize=(10, 10))
    ax = plt.subplot(2, 5, (1, 3))
    plt.hist(np.array(save_obj_GS['acc_offspring_per_iter']), density=False, color='k')
    plt.axvline(np.mean(save_obj_GS['acc_offspring_per_iter']))
    plt.annotate(
        ' $\\bar r_{\\rm acc}=$ %.2f \n $r_{\\rm total}$=%.4f' % (np.mean(save_obj_GS['acc_offspring_per_iter']),
                                                                  np.sum(np.array(
                                                                      save_obj_GS['acc_offspring_per_iter']) > 0) / len(
                                                                      save_obj_GS['lambda_bar'])),
        (0.58, 0.9), xycoords='axes fraction', ha='left', va='top')
    plt.xlabel('acceptance rate per GS iteration')
    plt.ylabel('number of samples')
    plt.subplot(2, 5, 5)
    plt.boxplot(np.array(save_obj_GS['acc_offspring_per_iter']))
    plt.xlabel('acceptance rate per GS iteration')
    plt.subplot(2, 5, (6, 10))
    plt.plot(np.array(save_obj_GS['acc_offspring_per_iter']), '--.', color='gray')
    plt.xlabel('GS iteration')
    plt.ylabel('acceptance rate')
    return h1


def plot_autocorr(save_obj_GS, idx_samples=None, varnames=None, xlim=None,
                  maxlag=None, label_ESS=None, alpha=None,
                  **kwargs_plot):
    # plot definitions
    pSIZE = 20
    plt.rc('font', size=pSIZE)
    plt.rc('axes', titlesize=pSIZE)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    if idx_samples is None:
        idx_samples = np.arange(0, len(save_obj_GS['lambda_bar']))
    Ksamples = len(idx_samples)
    if maxlag is None:
        maxlag = np.min([100, len(idx_samples)])
    dim_theta = len(save_obj_GS['theta_tilde'][0])
    if varnames is None:
        if dim_theta == 7:
            varnames = ['$K_0$', '$c$', '$p$', '$\\alpha_m$', '$D$', '$\\gamma$', 'q', '$\\bar{\\lambda}$']
            colors[7] = '#000000'
        if dim_theta == 5:
            varnames = ['$K_0$', '$c$', '$p$', '$\\alpha_m$', '$D_{\\rm gauss}$', '$\\bar{\\lambda}$']
            colors[5] = '#000000'
    numvars = dim_theta + 1
    x = np.empty([Ksamples, numvars])
    x[:, :-1] = np.array(save_obj_GS['theta_tilde'])[idx_samples, :]
    x[:, -1] = (np.array(save_obj_GS['lambda_bar'])[idx_samples]).flatten()
    if xlim is None:
        xlim = [-1, maxlag]
    if alpha is None:
        alpha = 0.05
    linewidth = 2

    h1 = plt.figure(figsize=(15, 5))
    h1.subplots_adjust(hspace=0.2, wspace=0.2)
    for i in range(numvars):
        r, u_level = gpetas.some_fun.autocorr(x[:, i], alpha=alpha)
        ax = plt.subplot(2, numvars // 2, i + 1)
        plt.vlines(x=np.arange(0, maxlag), ymin=0, ymax=r[0:maxlag], lw=linewidth, color=colors[i])
        plt.axhline(y=0, color='k')
        plt.axhline(y=u_level, color='gray', linestyle='--')
        plt.axhline(y=-u_level, color='gray', linestyle='--')
        plt.ylim([-1, 1])
        plt.xlim([xlim[0], xlim[1]])
        plt.ylabel('correlation')
        if i < numvars // 2:
            ax.xaxis.set_ticklabels([])
        if i >= numvars // 2:
            plt.xlabel('lag')
            # plt.xlabel('lag $\\times 10^3$')
        if (i != 0) and (i != numvars // 2):
            ax.yaxis.set_ticklabels([])
            plt.ylabel('')
        plt.locator_params(tight=True, nbins=3)

        if label_ESS is not None:
            Neff = len(x) / (1 + 2 * np.max(np.cumsum(r[1:])))
            maxlag_ESS = np.argmax(np.cumsum(r[1:]))
            ax.annotate('$N_{\\rm eff}^{(%i)}\\approx$%i (%.1f)' % (maxlag_ESS, Neff, Neff / len(x)),
                        (0.75, 0.05), xycoords='axes fraction', ha='center', va='bottom', fontsize=12)
            plt.axvline(x=maxlag_ESS, color='gray', linestyle='--')

        ax.annotate(varnames[i], (0.85, 0.95), xycoords='axes fraction', ha='center', va='top')
    return h1


def plot_offspring_mcmc_chains(save_obj_GS, gm_obj=None, mle_obj=None, mle_obj_silverman=None,
                               theta_true_Kcpadgq=None):
    # plot definitions
    pSIZE = 20
    plt.rc('font', size=pSIZE)
    plt.rc('axes', titlesize=pSIZE)
    bins = 20

    data_obj = save_obj_GS['data_obj']
    Ksamples = len(save_obj_GS['lambda_bar'])
    dim_theta = len(save_obj_GS['theta_tilde'][0])
    if dim_theta == 5:
        leg_str = ['$K_0$', '$c$', '$p$', '$\\alpha_m$', '$D_{gauss}$']
    if dim_theta == 7:
        leg_str = ['$K_0$', '$c$', '$p$', '$\\alpha_m$', '$D$', '$\\gamma$', 'q']
    if gm_obj is not None:
        theta_true_Kcpadgq = gm_obj.theta_Kcpadgq

    h1 = plt.figure(figsize=(18, 5))
    ax = plt.subplot(1, 2, 1)
    plt.semilogy(save_obj_GS['theta_tilde'], '.-')
    plt.xlim([0, Ksamples])
    plt.xlabel('samples')
    plt.ylabel('values')
    plt.legend(leg_str, bbox_to_anchor=(1.05, 1.0), loc='upper left')
    if theta_true_Kcpadgq is not None:
        for i in range(len(theta_true_Kcpadgq)):
            plt.axhline(y=theta_true_Kcpadgq[i], color='k', linestyle='--')

    ax = plt.subplot(1, 3, 3)
    plt.plot(save_obj_GS['l_tilde'], '.', color='gray', label='l($\\tilde{\\theta}$)')
    plt.xlabel('samples')
    plt.ylabel('$\\ln p(\\mathcal{D}|Z,\\theta_{\\varphi}$)')

    # hist of marginals
    if dim_theta == 5:
        leg_str = ['$K_0$', '$c$', '$p$', '$\\alpha_m$', '$D_{gauss}$']
    if dim_theta == 7:
        leg_str = ['$K_0$', '$c$', '$p$', '$\\alpha_m$', '$D$', '$\\gamma$', 'q']
    theta_tilde_array = np.copy(np.array(save_obj_GS['theta_tilde']))
    # idx_1000 = [0, 1, 4]  # K,c,d
    plot_dummy = np.ones([Ksamples, dim_theta])
    # plot_dummy[:, idx_1000] = plot_dummy[:, idx_1000] * 1000
    h2 = plt.figure(figsize=(20, 11))
    for i in range(dim_theta):
        ax = plt.subplot(2, int((dim_theta + 1) / 2), i + 1)
        plt.locator_params(tight=True, nbins=3)
        plt.hist(theta_tilde_array[:, i] * plot_dummy[:, i], color='gray', density=True, bins=bins)
        plt.axvline(x=np.median(theta_tilde_array[:, i] * plot_dummy[:, i]), color='k', label='GP-E median')
        if theta_true_Kcpadgq is not None:
            plt.axvline(theta_true_Kcpadgq[i] * plot_dummy[0, i], color='m', label='gM true')
        if mle_obj is not None:
            plt.axvline(mle_obj.theta_mle_Kcpadgq[i] * plot_dummy[0, i], color='k', linestyle='--', label='E')
        if mle_obj_silverman is not None:
            plt.axvline(mle_obj_silverman.theta_mle_Kcpadgq[i] * plot_dummy[0, i], color='k', linestyle=':',
                        label='E-S')
        plt.xlabel(leg_str[i])
        ax.yaxis.set_ticklabels([])
        # prior
        if hasattr(save_obj_GS['setup_obj'], 'prior_theta_dist'):
            xlim = plt.gca().get_xlim()
            if save_obj_GS['setup_obj'].prior_theta_dist == 'uniform':
                prior_theta_params = save_obj_GS['setup_obj'].prior_theta_params
                x = np.linspace(xlim[0], xlim[1], 1000)
                u = sc.stats.uniform(loc=prior_theta_params[i, 0],
                                     scale=prior_theta_params[i, 1] - prior_theta_params[i, 0])
                plt.plot(x, u.pdf(x), color='b', linestyle='-', label='prior', zorder=-5, linewidth=0.5)
            if save_obj_GS['setup_obj'].prior_theta_dist == 'gamma':
                prior_theta_params = save_obj_GS['setup_obj'].prior_theta_params
                mean_prior, c_prior = prior_theta_params[i, :]
                alpha_prior = 1. / c_prior ** 2.
                beta_prior = alpha_prior / mean_prior
                x = np.linspace(xlim[0], xlim[1], 1000)
                p_x = sc.stats.gamma.pdf(x, a=alpha_prior, scale=1. / beta_prior, loc=0)
                plt.plot(x, p_x, color='b', linestyle='-', label='prior', zorder=-5, linewidth=0.5)
        else:
            xlim = plt.gca().get_xlim()
            plt.plot(xlim, [1 / 10., 1 / 10.], color='b', linestyle='-', label='prior', linewidth=0.5)
        # xlabel, ylabels
        if i == 0:
            plt.ylabel('density')
            plt.ticklabel_format(style='sci', axis='x', scilimits=(-3, -3), useMathText=True)  # idx1000
        if i == 1:
            plt.ticklabel_format(style='sci', axis='x', scilimits=(-3, -3), useMathText=True)  # idx1000
        if i == 4:
            plt.ylabel('density')
            plt.ticklabel_format(style='sci', axis='x', scilimits=(-3, -3), useMathText=True)  # idx1000
        if i == 6:
            plt.legend(fontsize=20, bbox_to_anchor=(1.05, 1.0), loc='upper left')

    return h1, h2


def plot_scatter(save_obj_GS, mle_obj=None, mle_obj_silverman=None, tracked_values=None, cmap='seismic', r_value=0.1,
                 median_yes=None, **kwargs_plot):
    if tracked_values is None:
        data = np.array(save_obj_GS['theta_tilde'])
    else:
        data = np.array(save_obj_GS['tracked_data']['theta_tilde'])
    numdata, numvars = data.shape
    if numvars == 7:
        varnames = ['$K_0$', '$c$', '$p$', '$\\alpha_m$', '$D$', '$\\gamma$', 'q']
    if numvars == 5:
        varnames = ['$K_0$', '$c$', '$p$', '$\\alpha_m$', '$D_{gauss}$']
    if cmap == 'seismic':
        normcolor = matplotlib.colors.TwoSlopeNorm(vcenter=0, vmin=-5, vmax=5)
    else:
        normcolor = matplotlib.colors.Normalize(vmin=0, vmax=1.)

    h1 = plt.figure(figsize=(10, 10))
    h1.subplots_adjust(hspace=0.05, wspace=0.05)
    for i in range(numvars):
        for j in range(numvars):
            ax = plt.subplot(numvars, numvars, (i * numvars + j + 1))
            ax.plot(data[:, j], data[:, i], linestyle='', marker='.', color='k', markersize=0.5, **kwargs_plot)
            if mle_obj is not None:
                plt.plot(mle_obj.theta_mle_Kcpadgq[j], mle_obj.theta_mle_Kcpadgq[i], 'dr')
            if mle_obj_silverman is not None:
                plt.plot(mle_obj_silverman.theta_mle_Kcpadgq[j], mle_obj_silverman.theta_mle_Kcpadgq[i], 'ob')
            if median_yes is not None:
                plt.plot(np.median(data, axis=0)[j], np.median(data, axis=0)[i], 'sm')

            if np.corrcoef(data[:, i], data[:, j])[0, 1] >= r_value:
                ax.annotate('%.2f' % np.corrcoef(data[:, i], data[:, j])[0, 1], (0.99, 0.9), xycoords='axes fraction',
                            ha='right', va='center', fontsize=12, color='r')

            if np.corrcoef(data[:, i], data[:, j])[0, 1] <= -r_value:
                ax.annotate('%.2f' % np.corrcoef(data[:, i], data[:, j])[0, 1], (0.99, 0.9), xycoords='axes fraction',
                            ha='right', va='center', fontsize=12, color='b')

            cmap_used = matplotlib.cm.get_cmap(cmap)
            ax.set_facecolor(cmap_used(normcolor(np.abs(np.corrcoef(data[:, i], data[:, j])[0, 1]))))
            if np.abs(np.corrcoef(data[:, i], data[:, j])[0, 1]) <= r_value:
                ax.set_facecolor('w')

            if i == j:
                ax.clear()
                ax.set_facecolor('w')
                ax.annotate(varnames[i], (0.5, 0.5), xycoords='axes fraction',
                            ha='center', va='center')

            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            if j == 0 and i > 0:
                ax.yaxis.set_visible(True)
            if j < numvars - 1 and i == numvars - 1:
                ax.xaxis.set_visible(True)
            plt.locator_params(axis='y', nbins=1)
            plt.locator_params(axis='x', nbins=1)
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.tick_params(axis='both', which='minor', labelsize=12)
            nx = 2
            ny = 1
            for index, label in enumerate(ax.xaxis.get_ticklabels()):
                if index % nx != 0:
                    label.set_visible(False)
            for index, label in enumerate(ax.yaxis.get_ticklabels()):
                if index % ny != 0:
                    label.set_visible(False)
    return h1


def plot_scatter_data(data, varnames=None, cmap='seismic', r_value=0.1, **kwargs_plot):
    numdata, numvars = data.shape
    if varnames is None:
        varnames = ['$K_0$', '$c$', '$p$', '$\\alpha_m$', '$D$', '$\\gamma$', 'q']
    if cmap == 'seismic':
        normcolor = matplotlib.colors.TwoSlopeNorm(vcenter=0, vmin=-5, vmax=5)
    else:
        normcolor = matplotlib.colors.Normalize(vmin=0, vmax=1.)

    h1 = plt.figure(figsize=(10, 10))
    h1.subplots_adjust(hspace=0.05, wspace=0.05)
    for i in range(numvars):
        for j in range(numvars):
            ax = plt.subplot(numvars, numvars, (i * numvars + j + 1))
            ax.plot(data[:, j], data[:, i], linestyle='', marker='.', color='k', markersize=0.5, **kwargs_plot)

            if np.corrcoef(data[:, i], data[:, j])[0, 1] >= r_value:
                ax.annotate('%.2f' % np.corrcoef(data[:, i], data[:, j])[0, 1], (0.99, 0.9), xycoords='axes fraction',
                            ha='right', va='center', fontsize=12, color='r')

            if np.corrcoef(data[:, i], data[:, j])[0, 1] <= -r_value:
                ax.annotate('%.2f' % np.corrcoef(data[:, i], data[:, j])[0, 1], (0.99, 0.9), xycoords='axes fraction',
                            ha='right', va='center', fontsize=12, color='b')

            cmap_used = matplotlib.cm.get_cmap(cmap)
            ax.set_facecolor(cmap_used(normcolor(np.abs(np.corrcoef(data[:, i], data[:, j])[0, 1]))))
            if np.abs(np.corrcoef(data[:, i], data[:, j])[0, 1]) <= r_value:
                ax.set_facecolor('w')

            if i == j:
                ax.clear()
                ax.set_facecolor('w')
                ax.annotate(varnames[i], (0.5, 0.5), xycoords='axes fraction',
                            ha='center', va='center')

            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            if j == 0 and i > 0:
                ax.yaxis.set_visible(True)
            if j < numvars - 1 and i == numvars - 1:
                ax.xaxis.set_visible(True)
            plt.locator_params(axis='y', nbins=1)
            plt.locator_params(axis='x', nbins=1)
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.tick_params(axis='both', which='minor', labelsize=12)
            nx = 2
            ny = 1
            for index, label in enumerate(ax.xaxis.get_ticklabels()):
                if index % nx != 0:
                    label.set_visible(False)
            for index, label in enumerate(ax.yaxis.get_ticklabels()):
                if index % ny != 0:
                    label.set_visible(False)
    return h1


def plot_number_of_bg_events(save_obj_GS, mle_obj=None, mle_obj_silverman=None, gm_obj=None, kde_hist=None, pred=None,
                             quantile=None):
    plt.rcParams.update({'font.size': 40})
    data_obj = save_obj_GS['data_obj']
    Nall = len(data_obj.data_all.times[data_obj.idx_training])
    Ksamples = len(save_obj_GS['lambda_bar'])
    bins = 20
    data_obj = save_obj_GS['data_obj']
    abs_T_training = np.diff(data_obj.domain.T_borders_training)
    abs_X = np.prod(np.diff(data_obj.domain.X_borders))
    L = len(save_obj_GS['mu_grid'][0])

    N0_gpetas_mu = np.sum(save_obj_GS['mu_grid'], axis=1) * abs_X / L * abs_T_training
    N0_gpetas = np.array(save_obj_GS['N_S0']).squeeze()
    mean_N0 = np.mean(N0_gpetas_mu)

    h1 = plt.figure()
    # GP-E
    plt.hist(N0_gpetas_mu, density=True, color='k', label='GP-E histogram', bins=bins)
    plt.axvline(x=mean_N0, color='r', label='GP-E:   %i $\\pm$%i \n (%.2f$\\pm$%.3f)'
                                            % (mean_N0, np.std(N0_gpetas_mu), mean_N0 / Nall,
                                               np.std(N0_gpetas_mu) / Nall))
    if quantile is not None:
        plt.axvline(x=np.quantile(np.array(N0_gpetas_mu), 1. - quantile), color='r', linestyle=':',
                    label='%.3f,%.3f quantiles' % (1 - quantile, quantile))
        plt.axvline(x=np.quantile(np.array(N0_gpetas_mu), quantile), color='r', linestyle=':')
    # other
    if mle_obj is not None:
        N0_mle_kde = np.sum(mle_obj.p_i_vec)
        plt.axvline(x=N0_mle_kde, color='green', linestyle='--',
                    label='E:         %i\n (%.2f)' % (N0_mle_kde, N0_mle_kde / Nall))
        if kde_hist is not None:
            N0_hist_kde_mle = np.random.poisson(lam=N0_mle_kde, size=Ksamples)
            plt.hist(N0_hist_kde_mle, density=True, color='g',
                     label='E: HPP sampled \n %.1f $\\pm$%.1f' % (np.mean(N0_hist_kde_mle), np.std(N0_hist_kde_mle)),
                     alpha=0.75, bins=bins)
    if mle_obj_silverman is not None:
        N0_mle_kde = np.sum(mle_obj_silverman.p_i_vec)
        plt.axvline(x=N0_mle_kde, color='dodgerblue', linestyle=':',
                    label='E-S:         %i\n (%.2f)' % (N0_mle_kde, N0_mle_kde / Nall))
        if kde_hist is not None:
            N0_hist_kde_mle = np.random.poisson(lam=N0_mle_kde, size=Ksamples)
            plt.hist(N0_hist_kde_mle, density=True, color='dodgerblue',
                     label='E-S: HPP sampled \n %.1f $\\pm$%.1f' % (np.mean(N0_hist_kde_mle), np.std(N0_hist_kde_mle)),
                     alpha=0.5, bins=bins)
    if gm_obj is not None:
        plt.axvline(x=gm_obj.integral_mu_x_unit_time * np.diff(gm_obj.domain.T_borders_all),
                    color='m', linestyle='-',
                    label='gM truth %i\n (%.2f)' % (
                        gm_obj.integral_mu_x_unit_time * np.diff(gm_obj.domain.T_borders_all),
                        len(gm_obj.S0_events[:, 0]) / Nall))

    if pred is not None:
        Ndraws = 1
        N0_distribution = np.empty([Ksamples, Ndraws])
        for k in range(Ksamples):
            N0_distribution[k] = np.random.poisson(lam=N0_gpetas_mu[k], size=Ndraws)
        plt.hist(N0_distribution.reshape(-1), density=True, color='k',
                 label='GP-E: HPP sampled \n %.1f $\\pm$%.1f' % (
                     np.mean(N0_distribution.reshape(-1)), np.std(N0_distribution.reshape(-1))),
                 alpha=1, bins=bins, histtype='step')

    plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.xlabel('background events $\\in \\mathcal{X}$')
    plt.ylabel('density')
    # plt.show()
    return h1


def plot_hist_lambda_bar(save_obj_GS, lambda_bar_true=None, quantile=None):
    plt.rcParams.update({'font.size': 40})
    if quantile is None:
        quantile = 0.01
    lambda_bar_array = np.array(save_obj_GS['lambda_bar']).reshape(-1)
    upper_bound_true = lambda_bar_true

    hf = plt.figure(figsize=(10, 10))
    plt.hist(lambda_bar_array, density=True, color='k', rasterized=True)
    plt.axvline(x=np.median(lambda_bar_array), color='b', linewidth=3, label='median GP-ETAS')
    if upper_bound_true is not None:
        plt.axvline(x=upper_bound_true, color='m', label='ground truth', linewidth=3)
    plt.axvline(x=np.quantile(lambda_bar_array, 1. - quantile), color='darkgray', linestyle='--',
                label='%.2f,%.2f quantiles' % (1 - quantile, quantile))
    plt.axvline(x=np.quantile(lambda_bar_array, quantile), color='darkgray', linestyle='--')
    plt.legend(fontsize=35, loc='upper right', framealpha=1)
    xmin, xmax, ymin, ymax = plt.axis()
    y_up = ymax * 2
    plt.ylim([0, y_up])
    plt.xlabel('$\\bar\lambda$')
    plt.ylabel('\ndensity\n\n')
    ax = plt.gca()
    ax.tick_params(axis='both', direction='out')
    ax.yaxis.set_label_position("right")
    ax.tick_params(direction='out', left=True, right=True, top=True, bottom=True, labelright=True, labelleft=False)
    # plt.text(xmin * 0.5, y_up, '(f)', verticalalignment='top')
    ax.annotate('(f)',
                (-0.1, 0.95), xycoords='axes fraction', ha='center', va='bottom', fontsize=40)
    # plt.show()
    return hf


def plot_lambda_bar(save_obj):
    """
    Lambda bar, latent Poisson process
    Analysis Background part 2: SAMPLER
    :param save_obj:
    :return:
    """
    absT = np.diff(save_obj['data_obj'].domain.T_borders_training)
    absX = np.prod(np.diff(save_obj['data_obj'].domain.X_borders))
    print('from orig. data_obj', '|T_training|', absT, '|X|', absX)

    # lambda_bar chain
    lambda_bar_array = np.array(save_obj['lambda_bar']).reshape(-1)
    h1 = plt.figure()
    plt.plot(lambda_bar_array, 'k')
    plt.ylabel('$\\bar\lambda$')
    plt.xlabel('samples')
    plt.axhline(y=np.median(lambda_bar_array), color='r')
    # plt.show(block=False)
    return h1


def plot_slices_and_upper_bound(save_obj_GS, gm_obj=None,
                                mle_obj=None, mle_obj_silverman=None, quantile=0.05, case_name=None,
                                xidx_vec=None, yidx_vec=None, logscale=None):
    # plot definitions
    pSIZE = 20
    plt.rc('font', size=pSIZE)
    plt.rc('axes', titlesize=pSIZE)

    # binning
    xbins = int(np.sqrt(len(save_obj_GS['X_grid'])))
    ybins = int(np.sqrt(len(save_obj_GS['X_grid'])))
    data_obj = save_obj_GS['data_obj']
    dx = np.diff(data_obj.domain.X_borders[0, :]) / xbins
    dy = np.diff(data_obj.domain.X_borders[1, :]) / ybins
    X_grid = save_obj_GS['X_grid']
    x = np.unique(X_grid[:, 0].reshape(xbins, -1))
    y = np.unique(X_grid[:, 1].reshape(xbins, -1))
    zlimits = [0, np.max(save_obj_GS['mu_grid'])]

    # slices
    if yidx_vec is None:
        yidx_vec = [4, 19, 29, 44]
    if xidx_vec is None:
        xidx_vec = [9, 19, 24, 34]
    if case_name == 'case_01':
        yidx_vec = [4, 19, 29, 44]
        xidx_vec = [9, 19, 24, 34]
    if case_name == 'case_02':
        yidx_vec = [4, 14, 24, 39]
        xidx_vec = [10, 14, 24, 34]
    if case_name == 'case_03':
        # yidx_1_vec = [11, 24, 37, 48]
        # xidx_1_vec = [9, 24, 40, 48]
        xidx_vec = [9, 24, 34, 44]
        yidx_vec = [14, 24, 38, 41]

    # print('-------figure y starts--------------')
    h1_slices = plt.figure(figsize=(20, 15.5))
    for i in range(len(yidx_vec)):
        yidx = yidx_vec[i]
        ax = plt.subplot(4, 4, i + 1)

        # truth
        if gm_obj is not None:
            mu_grid_true = gm_obj.mu_grid.reshape(xbins, -1).T
            plt.plot(x, mu_grid_true[:, yidx], 'm', linewidth=3, label='ground truth')

        # mle: ETAS classical
        if mle_obj is not None:
            mu_grid_kde = mle_obj.mu_grid.reshape(xbins, -1).T
            plt.plot(x, mu_grid_kde[:, yidx], '--k', linewidth=1.5, label='ETAS classical')
            zlimits = [0, np.max(mle_obj.mu_grid)]
        if mle_obj_silverman is not None:
            mu_grid_kde_silverman = mle_obj_silverman.mu_grid.reshape(xbins, -1).T
            plt.plot(x, mu_grid_kde_silverman[:, yidx], ':k', linewidth=1.5,
                     label='ETAS silverman')

        # gpetas median and quantiles
        mu_array_slice = np.array(save_obj_GS['mu_grid']).reshape(-1, xbins, xbins)
        plt.plot(x, np.median(mu_array_slice[:, yidx, :], axis=0), 'k', linewidth=3,
                 label='GP-ETAS (this study)')
        plt.fill_between(x=x, y1=np.median(mu_array_slice[:, yidx, :], axis=0),
                         y2=np.quantile(mu_array_slice[:, yidx, :], 1. - quantile, axis=0),
                         facecolor='gray', alpha=0.5)
        plt.fill_between(x=x, y1=np.median(mu_array_slice[:, yidx, :], axis=0),
                         y2=np.quantile(mu_array_slice[:, yidx, :], quantile, axis=0),
                         facecolor='gray', alpha=0.5)

        # upper bound parameter lambda_bar
        plt.plot(x, np.median(save_obj_GS['lambda_bar']) * np.ones(len(x)), color='b', linestyle='-', linewidth=0.5,
                 label='median $\\bar\lambda$')

        if logscale is not None:
            plt.yscale('log')
            if mle_obj is not None:
                zlimits = [np.min(mle_obj.mu_grid), np.max(mle_obj.mu_grid)]
            else:
                zlimits = [np.min(save_obj_GS['mu_grid']), np.max(save_obj_GS['mu_grid'])]
        plt.ylim(zlimits)
        plt.xlim(data_obj.domain.X_borders[0, :])
        plt.xlabel('x')

        if (i) > 0:
            ax.set_yticklabels([])
        if i == 0:
            plt.legend(fontsize=16)
            if logscale is not None:
                plt.ylabel('$\\log \ \\mu(x)$')
            else:
                plt.ylabel('$\\mu(x)$')

        xpos_text = data_obj.domain.X_borders[0, 0] + 0.05 * np.diff(data_obj.domain.X_borders[0, :])
        if i == 0:
            if logscale is None:
                plt.text(xpos_text, zlimits[1] * 0.3, 'y=%.1f' % ((yidx + 1) * dy + data_obj.domain.X_borders[1, 0]),
                         verticalalignment='top')
            if logscale is not None:
                plt.text(data_obj.domain.X_borders[0, 1], zlimits[1] * 0.3,
                         'y=%.1f' % ((yidx + 1) * dy + data_obj.domain.X_borders[1, 0]),
                         verticalalignment='top', horizontalalignment='right')
        if i > 0:
            plt.text(xpos_text, zlimits[1] * 0.95, 'y=%.1f' % ((yidx + 1) * dy + data_obj.domain.X_borders[1, 0]),
                     verticalalignment='top')

        plt.tight_layout()

    # print('-------figure x starts--------------')
    for i in range(len(xidx_vec)):
        xidx = xidx_vec[i]
        ax = plt.subplot(4, 4, i + 1 + 4)

        # truth
        if gm_obj is not None:
            mu_grid_true = gm_obj.mu_grid.reshape(xbins, -1).T
            plt.plot(y, mu_grid_true[xidx, :], 'm', linewidth=3, label='ground truth')

        # mle: ETAS classical
        if mle_obj is not None:
            mu_grid_kde = mle_obj.mu_grid.reshape(xbins, -1).T
            plt.plot(y, mu_grid_kde[xidx, :], '--k', linewidth=1.5, label='ETAS classical')
            zlimits = [0, np.max(mle_obj.mu_grid)]
        if mle_obj_silverman is not None:
            mu_grid_kde_silverman = mle_obj_silverman.mu_grid.reshape(xbins, -1).T
            plt.plot(y, mu_grid_kde_silverman[xidx, :], ':k', linewidth=1.5,
                     label='ETAS silverman')
        # gpetas median and quantiles
        mu_array_slice = np.array(save_obj_GS['mu_grid']).reshape(-1, xbins, xbins)
        plt.plot(y, np.median(mu_array_slice[:, :, xidx], axis=0), 'k', linewidth=3,
                 label='GP-ETAS (this study)')
        plt.fill_between(x=y, y1=np.median(mu_array_slice[:, :, xidx], axis=0),
                         y2=np.quantile(mu_array_slice[:, :, xidx], 1. - quantile, axis=0),
                         facecolor='gray', alpha=0.5)
        plt.fill_between(x=y, y1=np.median(mu_array_slice[:, :, xidx], axis=0),
                         y2=np.quantile(mu_array_slice[:, :, xidx], quantile, axis=0),
                         facecolor='gray', alpha=0.5)

        # upper bound parameter lambda_bar
        plt.plot(y, np.median(save_obj_GS['lambda_bar']) * np.ones(len(x)), color='b', linestyle='-', linewidth=0.5,
                 label='median $\\bar\lambda$')

        if logscale is not None:
            plt.yscale('log')
            if mle_obj is not None:
                zlimits = [np.min(mle_obj.mu_grid), np.max(mle_obj.mu_grid)]
            else:
                zlimits = [np.min(save_obj_GS['mu_grid']), np.max(save_obj_GS['mu_grid'])]
        plt.ylim(zlimits)
        plt.xlim(data_obj.domain.X_borders[1, :])
        plt.xlabel('y')

        if (i) > 0:
            ax.set_yticklabels([])
        if i == 0:
            if logscale is not None:
                plt.ylabel('$\\log \ \\mu(x)$')
            else:
                plt.ylabel('$\\mu(x)$')

        xpos_text = data_obj.domain.X_borders[1, 0] + 0.05 * np.diff(data_obj.domain.X_borders[1, :])
        if i == 0:
            if logscale is None:
                plt.text(xpos_text, zlimits[1] * 0.95, 'x=%.1f' % ((xidx + 1) * dy + data_obj.domain.X_borders[0, 0]),
                         verticalalignment='top')
            if logscale is not None:
                plt.text(data_obj.domain.X_borders[0, 1], zlimits[1] * 0.3,
                         'x=%.1f' % ((xidx + 1) * dy + data_obj.domain.X_borders[1, 0]),
                         verticalalignment='top', horizontalalignment='right')
        if (i > 0):
            plt.text(xpos_text, zlimits[1] * 0.95, 'x=%.1f' % ((xidx + 1) * dy + data_obj.domain.X_borders[0, 0]),
                     verticalalignment='top')

        plt.tight_layout()
    # plt.show()

    # where are the slices
    h2_where = gpetas.plotting.plot_intensity_2d(intensity_grid=np.median(save_obj_GS['mu_grid'], axis=0),
                                                 X_grid=X_grid)
    if mle_obj is not None:
        h2_where = gpetas.plotting.plot_intensity_2d(intensity_grid=mle_obj.mu_grid,
                                                     X_grid=X_grid)
    ax = h2_where.gca()
    for i in range(len(xidx_vec)):
        xidx = xidx_vec[i]
        yidx = yidx_vec[i]
        ax.axhline(y=(yidx + 1) * dy + data_obj.domain.X_borders[1, 0], color='r', linestyle='--')
        ax.axvline(x=(xidx + 1) * dx + data_obj.domain.X_borders[0, 0], color='r', linestyle='--')

    return h1_slices, h2_where


def plot_intensity_2d(intensity_grid, X_grid=None,
                      data_obj=None, points=None, data_testing_points=None, data_training_points=None, cmap_dots=None,
                      cmap=None, clim=None, cb_label=None, cb_ticks=None, cb_format=None,
                      contour_lines=None, cl_color=None,
                      fig_label=None,size_points=None):
    '''

    :param intensity_grid:
    :type intensity_grid:
    :param X_grid:
    :type X_grid:
    :param data_obj:
    :type data_obj:
    :param points:
    :type points:
    :param data_testing_points:
    :type data_testing_points:
    :param data_training_points:
    :type data_training_points:
    :param cmap_dots:
    :type cmap_dots:
    :param cmap:
    :type cmap:
    :param clim:
    :type clim:
    :param cb_label:
    :type cb_label:
    :param cb_ticks:
    :type cb_ticks:
    :param cb_format:
    :type cb_format:
    :param contour_lines:
    :type contour_lines:
    :param cl_color:
    :type cl_color:
    :param fig_label:
    :type fig_label:
    :return:
    :rtype:
    '''
    # plot definitions
    pSIZE = 30
    plt.rc('font', size=pSIZE)
    plt.rc('axes', titlesize=pSIZE)
    if cmap is None:
        cmap = 'viridis'
    if cmap_dots is None:
        cmap_dots = 'gray'

    # grid binning
    xbins = int(np.sqrt(len(intensity_grid)))

    # short plot variables
    z = intensity_grid.reshape(xbins, -1)
    if X_grid is not None:
        x = X_grid[:, 0].reshape(xbins, -1)
        y = X_grid[:, 1].reshape(xbins, -1)

    # pcolor version
    h1 = plt.figure(figsize=(10, 10))
    if X_grid is None:
        hp = plt.pcolor(intensity_grid.reshape(xbins, -1), cmap=cmap, rasterized=True, shading='auto')
    else:
        hp = plt.pcolor(x, y, z, cmap=cmap, rasterized=True, shading='auto')

    ax1 = plt.gca()

    # clim and colorbar
    if cb_format is None:
        cb_format = '% .4f'
    if clim is not None:
        hp.set_clim(clim[0], clim[1])
        if clim[1] < 1e-4:
            cb = plt.colorbar(hp, shrink=0.25, ticks=[clim[0], clim[1]], format='% .2E')
        else:
            cb = plt.colorbar(hp, shrink=0.25, ticks=[clim[0], clim[1]], format=cb_format)
        if cb_label is None:
            cb.set_label('$\mu(x)$')
            if np.max(intensity_grid) > clim[1]:
                cb.ax.set_yticklabels(['0', '>%.4f' % (clim[1])])
            if clim[1] < 1e-4:
                cb.ax.set_yticklabels(['%.2E' % (clim[0]), '%.2E' % (clim[1])])
        if cb_label is not None:
            cb.set_label(cb_label)
    else:
        hp.set_clim(np.min(intensity_grid), np.max(intensity_grid))
        if np.abs(np.max(intensity_grid)) < 1e-4:
            cb = plt.colorbar(hp, shrink=0.25, ticks=[np.min(intensity_grid), np.max(intensity_grid)], format='%.2E')
        else:
            cb = plt.colorbar(hp, shrink=0.25, ticks=[np.min(intensity_grid), np.max(intensity_grid)], format=cb_format)
        if cb_label is None:
            cb.set_label('$\mu(x)$')
        if cb_label is not None:
            cb.set_label(cb_label)
    if cb_ticks is not None:
        cb.remove()
        plt.draw()
        cb = plt.colorbar(hp, shrink=0.25, ticks=cb_ticks, format=cb_format)
        if cb_label is None:
            cb.set_label('$\mu(x)$')
        if cb_label is not None:
            cb.set_label(cb_label)

    # points
    if points is not None:
        if size_points is None:
            size_points = 20 #10
        plt.scatter(points[:, 0], points[:, 1], s=size_points, c='red')  # s=10
    if data_testing_points is not None:
        idx_testing = np.where((data_obj.data_all.times >= data_obj.domain.T_borders_testing[0]))
        points = data_obj.data_all.positions[idx_testing]
        points_time = data_obj.data_all.times[idx_testing]
        im = plt.scatter(points[:, 0], points[:, 1], s=15, c=points_time, cmap=cmap_dots, vmin=0,
                         vmax=data_obj.domain.T_borders_all[1])
    if data_training_points is not None:
        idx_training = np.where((data_obj.data_all.times >= data_obj.domain.T_borders_training[0]) & (
                data_obj.data_all.times <= data_obj.domain.T_borders_training[1]))
        points = data_obj.data_all.positions[idx_training]
        points_time = data_obj.data_all.times[idx_training]
        im = plt.scatter(points[:, 0], points[:, 1], s=15, c=points_time, cmap=cmap_dots, vmin=0,
                         vmax=data_obj.domain.T_borders_training[1])

    # contourlines
    if contour_lines is not None:
        if len(str(contour_lines)) == 1:
            # contour_lines = np.array([0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.18])
            contour_lines = np.linspace(np.min(z), np.max(z), 10)
        if cl_color is None:
            cl_color = 'k'
        h = plt.contour(x, y, z, contour_lines.T, colors=cl_color, linestyles=':')
        if np.max(contour_lines) < 0.0001:
            plt.clabel(h, fontsize=9, inline=1, fmt='%2.2E')
        else:
            plt.clabel(h, fontsize=9, inline=1, fmt='%2.4f')

    # axis label
    ax1.set_xlabel('$x_1$')
    ax1.set_ylabel('$x_2$')
    ax1.set_aspect('equal')
    if X_grid is not None:
        ax1.set_xlim([np.min(X_grid[:, 0]), np.max(X_grid[:, 0])])
        ax1.set_ylim([np.min(X_grid[:, 1]), np.max(X_grid[:, 1])])

    if fig_label is not None:
        letter_xpos = - 0.25 * np.diff(data_obj.domain.X_borders[0, :])
        plt.text(letter_xpos, np.max(X_grid[:, 1]), fig_label, verticalalignment='top')

    plt.locator_params(tight=True, nbins=3)
    # plt.show()
    return h1


def plot_setting(data_obj=None, save_obj_GS=None, test_data_blue=None, gm_obj=None, show_datasets='Yes',
                 show_domain=None, pos_xy_text=None, pos_xy_text_star=None, show_training_data_only=None,
                  plot_lon_lat=None, scale_markersize=None):
    """
    :param data_obj: type data from get_data in data_utils
    :param show_domain: shows X domain borders, default = None
    :return: figure handles
    """
    if data_obj is None:
        data_obj = save_obj_GS['data_obj']

    # plot definitions
    pSIZE = 30
    plt.rc('font', size=pSIZE)
    plt.rc('axes', titlesize=pSIZE)

    hf1a = None
    color_string = 'grey'
    idx_test = data_obj.data_all.times > data_obj.domain.T_borders_training[1]

    hf1 = plt.figure(figsize=(10, 10))
    plt.tight_layout()
    if plot_lon_lat is not None:
        if scale_markersize is None:
            plt.plot(data_obj.data_all.positions_lon_lat[:, 0], data_obj.data_all.positions_lon_lat[:, 1], 'k.', markersize=5.)
        else:
            m_size = 20 ** (-1.75 + 0.75 * data_obj.data_all.magnitudes)
            plt.scatter(data_obj.data_all.positions_lon_lat[:, 0], data_obj.data_all.positions_lon_lat[:, 1],
                    marker='o', facecolors='none', edgecolors='k',
                    s=m_size)
    else:
        if scale_markersize is None:
            plt.plot(data_obj.data_all.positions[:, 0], data_obj.data_all.positions[:, 1], 'k.', markersize=5)
        else:
            m_size = 20 ** (-1.75 + 0.75 * data_obj.data_all.magnitudes)
            plt.scatter(data_obj.data_all.positions[:, 0], data_obj.data_all.positions[:, 1],
                        marker='o', facecolors='none', edgecolors='k',
                        s=m_size)

    ax = plt.gca()
    ax.tick_params(direction='out', left=True, right=True, top=True, bottom=True)
    if test_data_blue == 1:
        plt.plot(data_obj.data_all.positions[idx_test, 0], data_obj.data_all.positions[idx_test, 1], 'b.', markersize=5)
    if show_domain is not None:
        plt.axvline(data_obj.domain.X_borders[0, 0], color=color_string)
        plt.axvline(data_obj.domain.X_borders[0, 1], color=color_string)
        plt.axhline(data_obj.domain.X_borders[1, 0], color=color_string)
        plt.axhline(data_obj.domain.X_borders[1, 1], color=color_string)
    xticks = np.round(np.linspace(data_obj.domain.X_borders[0, 0], data_obj.domain.X_borders[0, 1], 3), 3)
    yticks = np.round(np.linspace(data_obj.domain.X_borders[1, 0], data_obj.domain.X_borders[1, 1], 3), 3)
    if plot_lon_lat is not None:
        xticks = np.round(np.linspace(data_obj.domain.X_borders_lon_lat[0, 0], data_obj.domain.X_borders_lon_lat[0, 1], 3), 3)
        yticks = np.round(np.linspace(data_obj.domain.X_borders_lon_lat[1, 0], data_obj.domain.X_borders_lon_lat[1, 1], 3), 3)
    plt.xticks(xticks)
    plt.yticks(yticks)
    ax.set_yticklabels(('', yticks[1], yticks[2]))
    ax.set_xticklabels(('', xticks[1], xticks[2]))
    if plot_lon_lat is not None:
        plt.xlabel('$x_1$,  (Lon.) $N_{\mathcal{D} \cup \mathcal{D}^\\ast=%i}$' % (len(data_obj.data_all.positions[:, 0])))
        plt.ylabel('$x_2$,  (Lat.)')
    else:
        plt.xlabel(
            '$x_1$, $N_{\mathcal{D} \cup \mathcal{D}^\\ast=%i}$' % (len(data_obj.data_all.positions[:, 0])))
        plt.ylabel('$x_2$')
    ax = plt.gca()
    ax.tick_params(direction='out', left=True, right=True, top=True, bottom=True)
    plt.axis('square')
    if plot_lon_lat is not None:
        plt.xlim(data_obj.domain.X_borders_lon_lat[0, :])
        plt.ylim(data_obj.domain.X_borders_lon_lat[1, :])
    else:
        plt.xlim(data_obj.domain.X_borders[0, :])
        plt.ylim(data_obj.domain.X_borders[1, :])

    if data_obj.domain.X_borders_UTM_km is not None:
        hf1a = plt.figure(figsize=(10, 10))
        plt.tight_layout()
        plt.plot(data_obj.data_all.positions_UTM_km[:, 0], data_obj.data_all.positions_UTM_km[:, 1], 'k.', markersize=5)
        ax = plt.gca()
        if show_domain == 1:
            plt.axvline(data_obj.domain.X_borders_UTM_km[0, 0], color=color_string)
            plt.axvline(data_obj.domain.X_borders_UTM_km[0, 1], color=color_string)
            plt.axhline(data_obj.domain.X_borders_UTM_km[1, 0], color=color_string)
            plt.axhline(data_obj.domain.X_borders_UTM_km[1, 1], color=color_string)
        plt.xticks(np.linspace(data_obj.domain.X_borders_UTM_km[0, 0], data_obj.domain.X_borders_UTM_km[0, 1], 3))
        plt.yticks(np.linspace(data_obj.domain.X_borders_UTM_km[1, 0], data_obj.domain.X_borders_UTM_km[1, 1], 3))
        plt.xlabel('$x_1$,  (km)')
        plt.ylabel('$x_2$,  (km)')
        ax = plt.gca()
        ax.tick_params(direction='out', left=True, right=True, top=True, bottom=True)
        plt.axis('square')

    # plotting earthquake sequence with magnitudes over time
    x = np.empty([len(data_obj.data_all.times), 2])
    x[:, 0] = data_obj.data_all.times
    x[:, 1] = data_obj.data_all.magnitudes
    x_label = None
    training_start = data_obj.domain.T_borders_training[0]
    training_end = data_obj.domain.T_borders_training[1]

    hf2 = plt.figure(figsize=(10, 10))
    plt.locator_params(nbins=4)
    plt.subplots_adjust(hspace=0.1)
    ax = plt.subplot(2, 1, 1)
    plt.plot(x[:, 0], x[:, 1], '.k', markersize=5)
    plt.ylim((np.min(x[:, 1]) - 0.5, np.max(x[:, 1]) + 0.5))
    plt.xlim(data_obj.domain.T_borders_all)
    plt.ylabel('magnitude')
    if training_end is not None:
        plt.axvline(x=training_end, color='r')
    ax.tick_params(direction='out', left=True, right=True, top=True, bottom=True, labelright=True, labelleft=False)
    ax.set_xticklabels([])
    ax.yaxis.set_label_position("right")

    ax = plt.subplot(2, 1, 2)
    plt.locator_params(nbins=5)
    plt.step(np.concatenate([[0.], x[:, 0]]), np.arange(0, x[:, 0].shape[0] + 1, 1), 'k', linewidth=3)
    plt.xlim(data_obj.domain.T_borders_all)
    plt.ylabel('counts')
    if x_label is not None:
        plt.xlabel(x_label)
    else:
        plt.xlabel('time, days')
    # plt.text(np.min(x[:, 0]), x.shape[0],
    if pos_xy_text is None:
        plt.text(0.05, 0.975,
             '$N_{\mathcal{D}}=$ %s\n$m_{\mathcal{D}}\in[%.2f,%.2f]$'
             % (x[data_obj.idx_training].shape[0], np.min(x[data_obj.idx_training, 1]),
                np.max(x[data_obj.idx_training, 1])),
             horizontalalignment='left',
             verticalalignment='top', fontsize=20, transform=ax.transAxes)
    else:
        plt.text(pos_xy_text[0], pos_xy_text[1],
                 '$N_{\mathcal{D}}=$ %s\n$m_{\mathcal{D}}\in[%.2f,%.2f]$'
                 % (x[data_obj.idx_training].shape[0], np.min(x[data_obj.idx_training, 1]),
                    np.max(x[data_obj.idx_training, 1])),
                 horizontalalignment='left',
                 verticalalignment='top', fontsize=20, transform=ax.transAxes)
    if np.sum(idx_test) > 0:
        # plt.text(training_end, 0,
        if pos_xy_text_star is None:
            plt.text(0.95 * training_end, 0.65 * x.shape[0],
                     '$N_{\mathcal{D}^{\\ast}} =$ %s\n$m_{\mathcal{D}^{\\ast}}\in$[%.2f,%.2f]'
                     % (x[idx_test].shape[0], np.min(x[idx_test, 1]), np.max(x[idx_test, 1])),
                     horizontalalignment='right',
                     verticalalignment='top', fontsize=20)
        else:
            plt.text(pos_xy_text_star[0], pos_xy_text_star[1],
                     '$N_{\mathcal{D}^{\\ast}} =$ %s\n$m_{\mathcal{D}^{\\ast}}\in$[%.2f,%.2f]'
                     % (x[idx_test].shape[0], np.min(x[idx_test, 1]), np.max(x[idx_test, 1])),
                     horizontalalignment='right',
                     verticalalignment='top', fontsize=20, transform=ax.transAxes)

    if training_end is not None:
        plt.axvline(x=training_end, color='r')
    ax.tick_params(direction='out', left=True, right=True, top=True, bottom=True, labelright=True, labelleft=False)
    ax.yaxis.set_label_position("right")
    # plt.show()
    hf3 = None
    hf4 = None
    if show_datasets is not None:
        # training data
        hf3 = plt.figure(figsize=(10, 10))
        plt.tight_layout()
        idx = np.copy(data_obj.idx_training)
        if plot_lon_lat is not None:
            plt.plot(data_obj.data_all.positions_lon_lat[idx, 0], data_obj.data_all.positions_lon_lat[idx, 1], 'k.', markersize=5)
        else:
            plt.plot(data_obj.data_all.positions[idx, 0], data_obj.data_all.positions[idx, 1], 'k.', markersize=5)
        ax = plt.gca()
        ax.tick_params(direction='out', left=True, right=True, top=True, bottom=True)
        if show_domain is not None:
            plt.axvline(data_obj.domain.X_borders[0, 0], color=color_string)
            plt.axvline(data_obj.domain.X_borders[0, 1], color=color_string)
            plt.axhline(data_obj.domain.X_borders[1, 0], color=color_string)
            plt.axhline(data_obj.domain.X_borders[1, 1], color=color_string)
        xticks = np.round(np.linspace(data_obj.domain.X_borders[0, 0], data_obj.domain.X_borders[0, 1], 3), 3)
        yticks = np.round(np.linspace(data_obj.domain.X_borders[1, 0], data_obj.domain.X_borders[1, 1], 3), 3)
        if plot_lon_lat is not None:
            xticks = np.round(np.linspace(data_obj.domain.X_borders_lon_lat[0, 0], data_obj.domain.X_borders_lon_lat[0, 1], 3), 3)
            yticks = np.round(np.linspace(data_obj.domain.X_borders_lon_lat[1, 0], data_obj.domain.X_borders_lon_lat[1, 1], 3), 3)
        plt.xticks(xticks)
        plt.yticks(yticks)
        ax.set_yticklabels(('', yticks[1], yticks[2]))
        ax.set_xticklabels(('', xticks[1], xticks[2]))
        if plot_lon_lat is not None:
            plt.xlabel('$x_1$,  (Lon.)  $N_{\mathcal{D}}=%i$' % (x[data_obj.idx_training].shape[0]))
            plt.ylabel('$x_2$,  (Lat.)')
        else:
            plt.xlabel('$x_1$,  $N_{\mathcal{D}}=%i$' % (x[data_obj.idx_training].shape[0]))
            plt.ylabel('$x_2$')
        ax = plt.gca()
        ax.tick_params(direction='out', left=True, right=True, top=True, bottom=True)
        plt.axis('square')
        plt.xlim(data_obj.domain.X_borders[0, :])
        plt.ylim(data_obj.domain.X_borders[1, :])
        if plot_lon_lat is not None:
            plt.xlim(data_obj.domain.X_borders_lon_lat[0, :])
            plt.ylim(data_obj.domain.X_borders_lon_lat[1, :])

        hf4 = plt.figure(figsize=(10, 10))
        plt.tight_layout()
        idx = np.copy(idx_test)
        if plot_lon_lat is not None:
            plt.plot(data_obj.data_all.positions_lon_lat[idx, 0], data_obj.data_all.positions_lon_lat[idx, 1], 'k.', markersize=5)
        else:
            plt.plot(data_obj.data_all.positions[idx, 0], data_obj.data_all.positions[idx, 1], 'k.', markersize=5)
        ax = plt.gca()
        ax.tick_params(direction='out', left=True, right=True, top=True, bottom=True)
        if show_domain is not None:
            plt.axvline(data_obj.domain.X_borders[0, 0], color=color_string)
            plt.axvline(data_obj.domain.X_borders[0, 1], color=color_string)
            plt.axhline(data_obj.domain.X_borders[1, 0], color=color_string)
            plt.axhline(data_obj.domain.X_borders[1, 1], color=color_string)
        xticks = np.round(np.linspace(data_obj.domain.X_borders[0, 0], data_obj.domain.X_borders[0, 1], 3), 3)
        yticks = np.round(np.linspace(data_obj.domain.X_borders[1, 0], data_obj.domain.X_borders[1, 1], 3), 3)
        if plot_lon_lat is not None:
            xticks = np.round(np.linspace(data_obj.domain.X_borders_lon_lat[0, 0], data_obj.domain.X_borders_lon_lat[0, 1], 3), 3)
            yticks = np.round(np.linspace(data_obj.domain.X_borders_lon_lat[1, 0], data_obj.domain.X_borders_lon_lat[1, 1], 3), 3)
        plt.xticks(xticks)
        plt.yticks(yticks)
        ax.set_yticklabels(('', yticks[1], yticks[2]))
        ax.set_xticklabels(('', xticks[1], xticks[2]))
        if plot_lon_lat is not None:
            plt.xlabel('$x_1$,  (Lon.)  $N_{\mathcal{D}^\\ast}=%i$' % (x[idx].shape[0]))
            plt.ylabel('$x_2$,  (Lat.)')
        else:
            plt.xlabel('$x_1$,  $N_{\mathcal{D}^\\ast}=%i$' % (x[idx].shape[0]))
            plt.ylabel('$x_2$')
        ax = plt.gca()
        ax.tick_params(direction='out', left=True, right=True, top=True, bottom=True)
        plt.axis('square')
        plt.xlim(data_obj.domain.X_borders[0, :])
        plt.ylim(data_obj.domain.X_borders[1, :])
        if plot_lon_lat is not None:
            plt.xlim(data_obj.domain.X_borders_lon_lat[0, :])
            plt.ylim(data_obj.domain.X_borders_lon_lat[1, :])

    hf5 = None
    if show_training_data_only is not None:
        x = np.empty([len(data_obj.data_all.times), 2])
        x[:, 0] = data_obj.data_all.times
        x[:, 1] = data_obj.data_all.magnitudes
        x_label = None
        training_start = data_obj.domain.T_borders_training[0]
        training_end = data_obj.domain.T_borders_training[1]

        hf5 = plt.figure(figsize=(10, 10))
        plt.locator_params(nbins=4)
        plt.subplots_adjust(hspace=0.1)
        ax = plt.subplot(2, 1, 1)
        plt.plot(x[:, 0], x[:, 1], '.k', markersize=5)
        plt.ylim((np.min(x[:, 1]) - 0.5, np.max(x[:, 1]) + 0.5))
        plt.xlim(data_obj.domain.T_borders_training)
        plt.ylabel('magnitude')
        if training_end is not None:
            plt.axvline(x=training_end, color='r')
        ax.tick_params(direction='out', left=True, right=True, top=True, bottom=True, labelright=True, labelleft=False)
        ax.set_xticklabels([])
        ax.yaxis.set_label_position("right")

        ax = plt.subplot(2, 1, 2)
        plt.locator_params(nbins=5)
        plt.step(np.concatenate([[0.], x[:, 0]]), np.arange(0, x[:, 0].shape[0] + 1, 1), 'k', linewidth=3)
        plt.xlim(data_obj.domain.T_borders_training)
        plt.ylim([0, np.max(data_obj.idx_training) * 1.1])
        plt.ylabel('counts')
        if x_label is not None:
            plt.xlabel(x_label)
        else:
            plt.xlabel('time, days')
        # plt.text(np.min(x[:, 0]), x.shape[0],
        plt.text(0.05, 0.975,
                 '$N_{\mathcal{D}}=$ %s\n$m_{\mathcal{D}}\in[%.2f,%.2f]$'
                 % (x[data_obj.idx_training].shape[0], np.min(x[data_obj.idx_training, 1]),
                    np.max(x[data_obj.idx_training, 1])),
                 horizontalalignment='left',
                 verticalalignment='top', fontsize=20, transform=ax.transAxes)

        if training_end is not None:
            plt.axvline(x=training_end, color='r')
        ax.tick_params(direction='out', left=True, right=True, top=True, bottom=True, labelright=True, labelleft=False)
        ax.yaxis.set_label_position("right")

    return hf1, hf2, hf1a, hf3, hf4, hf5
