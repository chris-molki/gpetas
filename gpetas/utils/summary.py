import numpy as np
import os
import gpetas.utils.plotting
import matplotlib.pyplot as plt


class summary_gpetas():
    " Gives plots and summary tables of GP-ETAS inference"

    def __init__(self, save_obj_GS, gm_obj=None, mle_obj=None, mle_obj_silverman=None, fout_dir=None,
                 case_name=None,ltest_plot='yes'):

        # data structures
        data_obj = save_obj_GS['data_obj']
        if case_name is None:
            case_name = "case_XX"

        # init output directories
        if fout_dir is not None:
            out_dir = fout_dir
        else:
            out_dir = "output"
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        # (1) setup data
        if gm_obj is not None:
            hf1, hf2, h3 = gm_obj.plot_paper_figure02_data()
            hf1.savefig(out_dir + '/%s_data_01.pdf' % gm_obj.case_name, bbox_inches='tight')
            hf2.savefig(out_dir + '/%s_data_02.pdf' % gm_obj.case_name, bbox_inches='tight')
        if gm_obj is None:
            h1, h2, hf1a, h4a, h4b = gpetas.utils.plotting.plot_setting(data_obj=data_obj,show_datasets='yes')
            h1.savefig(out_dir + '/%s_data_01_rd.pdf' % data_obj.case_name, bbox_inches='tight')
            h2.savefig(out_dir + '/%s_data_02.pdf' % data_obj.case_name, bbox_inches='tight')
            h4a.savefig(out_dir + '/%s_data_03.pdf' % data_obj.case_name, bbox_inches='tight')
            h4b.savefig(out_dir + '/%s_data_04.pdf' % data_obj.case_name, bbox_inches='tight')

        # (2) background
        h1 = gpetas.plotting.plot_intensity_2d(np.median(save_obj_GS['mu_grid'], axis=0),
                                               save_obj_GS['X_grid'])
        h1.savefig(out_dir + '/%s_F02_mu_gpetas_median.pdf' % data_obj.case_name, bbox_inches='tight')
        plt.close(h1)
        h1 = gpetas.plotting.plot_intensity_2d(np.std(save_obj_GS['mu_grid'], axis=0),
                                               save_obj_GS['X_grid'], cmap='binary',
                                               cb_label='stand. deviation')
        h1.savefig(out_dir + '/%s_F02d_mu_gpetas_sigma.pdf' % data_obj.case_name, bbox_inches='tight')
        plt.close(h1)
        h1 = gpetas.plotting.plot_intensity_2d(
            np.std(save_obj_GS['mu_grid'], axis=0) / np.mean(save_obj_GS['mu_grid'], axis=0),
            save_obj_GS['X_grid'], cmap='binary',
            cb_label='coeff. of var')
        h1.savefig(out_dir + '/%s_F02e_mu_gpetas_coeff_var.pdf' % data_obj.case_name, bbox_inches='tight')
        plt.close(h1)
        if mle_obj is not None:
            mu_min = np.min([np.min(np.median(save_obj_GS['mu_grid'], axis=0)), np.min(mle_obj.mu_grid)])
            mu_max = np.max([np.max(np.median(save_obj_GS['mu_grid'], axis=0)), np.max(mle_obj.mu_grid)])
            h1 = gpetas.plotting.plot_intensity_2d(np.median(save_obj_GS['mu_grid'], axis=0),
                                                   save_obj_GS['X_grid'], clim=[mu_min, mu_max],
                                                   cmap='binary')
            h1.savefig(out_dir + '/%s_F02a_mu_gpetas_median_binary.pdf' % data_obj.case_name, bbox_inches='tight')
            plt.close(h1)
            h1 = gpetas.plotting.plot_intensity_2d(np.log10(np.median(save_obj_GS['mu_grid'], axis=0)),
                                                   save_obj_GS['X_grid'],
                                                   clim=[np.floor(np.log10(mu_min)), np.ceil(np.log10(mu_max))],
                                                   cmap='viridis')
            h1.savefig(out_dir + '/%s_F02_log10a_mu_gpetas_median.pdf' % data_obj.case_name, bbox_inches='tight')
            plt.close(h1)
            h1 = gpetas.plotting.plot_intensity_2d(mle_obj.mu_grid,
                                                   save_obj_GS['X_grid'], clim=[mu_min, mu_max],
                                                   cmap='binary')
            h1.savefig(out_dir + '/%s_F02b_mu_mle_default.pdf' % data_obj.case_name, bbox_inches='tight')
            plt.close(h1)
            h1 = gpetas.plotting.plot_intensity_2d(np.log10(mle_obj.mu_grid),
                                                   save_obj_GS['X_grid'],
                                                   clim=[np.floor(np.log10(mu_min)), np.ceil(np.log10(mu_max))],
                                                   cmap='viridis')
            h1.savefig(out_dir + '/%s_F02_log10b_mu_mle_default_log10.pdf' % data_obj.case_name, bbox_inches='tight')
            plt.close(h1)
            diff_max = np.max(np.abs(np.median(save_obj_GS['mu_grid'], axis=0) - mle_obj.mu_grid))
            h1 = gpetas.plotting.plot_intensity_2d(mle_obj.mu_grid - np.median(save_obj_GS['mu_grid'], axis=0),
                                                   save_obj_GS['X_grid'], clim=[-diff_max, diff_max],
                                                   cmap='seismic', cb_label='$\\mu_{\\rm E}-\\mu_{\\rm GPE}$')
            h1.savefig(out_dir + '/%s_mu_diff_gpetas_mle_default.pdf' % data_obj.case_name, bbox_inches='tight')
            plt.close(h1)
            if mle_obj_silverman is not None:
                h1 = gpetas.plotting.plot_intensity_2d(mle_obj_silverman.mu_grid,
                                                       save_obj_GS['X_grid'], clim=[mu_min, mu_max],
                                                       cmap='binary')
                h1.savefig(out_dir + '/%s_F02c_mu_mle_silverman.pdf' % data_obj.case_name, bbox_inches='tight')
                plt.close(h1)
            if gm_obj is not None:
                h1 = gpetas.plotting.plot_intensity_2d(gm_obj.mu_grid, save_obj_GS['X_grid'],
                                                       cmap='binary', clim=[mu_min, mu_max],
                                                       points=gm_obj.S0_events[:, 2:4])
                h1.savefig(out_dir + '/%s_mu_gm.pdf' % data_obj.case_name, bbox_inches='tight')
                plt.close(h1)
                quantile = 0.05
                h1 = gpetas.plotting.plot_intensity_2d((np.quantile(save_obj_GS['mu_grid'], q=1 - quantile, axis=0) -
                                                        np.quantile(save_obj_GS['mu_grid'], q=quantile, axis=0)) / 2,
                                                       save_obj_GS['X_grid'], cmap='binary',
                                                       cb_label='$(q95-q5)/2$',
                                                       points=gm_obj.S0_events[:, 2:4])
                h1.savefig(out_dir + '/%s_mu_gpetas_q95q5.pdf' % data_obj.case_name, bbox_inches='tight')
                plt.close(h1)
                h1 = gpetas.plotting.plot_intensity_2d(mle_obj.mu_grid - gm_obj.mu_grid,
                                                       save_obj_GS['X_grid'], clim=[-diff_max, diff_max],
                                                       cmap='seismic', cb_label='$\\mu_{\\rm E}-\\mu_{\\rm true}$')
                h1.savefig(out_dir + '/%s_mu_diff_gm_mle_default.pdf' % data_obj.case_name, bbox_inches='tight')
                plt.close(h1)
                h1 = gpetas.plotting.plot_intensity_2d(np.median(save_obj_GS['mu_grid'], axis=0) - gm_obj.mu_grid,
                                                       save_obj_GS['X_grid'], clim=[-diff_max, diff_max],
                                                       cmap='seismic', cb_label='$\\mu_{\\rm GPE}-\\mu_{\\rm true}$')
                h1.savefig(out_dir + '/%s_mu_diff_gm_gpetas.pdf' % data_obj.case_name, bbox_inches='tight')
                plt.close(h1)
                h1 = gpetas.plotting.plot_intensity_2d(
                    np.log10(np.abs(mle_obj.mu_grid - np.median(save_obj_GS['mu_grid'], axis=0)) / gm_obj.mu_grid),
                    save_obj_GS['X_grid'],
                    points=data_obj.data_all.positions,
                    cmap='viridis', cb_label='$\\mu_{\\rm GPE}-\\mu_{\\rm true}$')
                h1.savefig(out_dir + '/%s_mu_diff_mle_gpetas_normed_by_gm.pdf' % data_obj.case_name,
                           bbox_inches='tight')
                plt.close(h1)

        # (3) Slices bg
        h1, h2 = gpetas.plotting.plot_slices_and_upper_bound(save_obj_GS,
                                                             gm_obj=gm_obj,
                                                             mle_obj=mle_obj,
                                                             mle_obj_silverman=mle_obj_silverman,
                                                             quantile=0.05,
                                                             case_name=case_name,
                                                             xidx_vec=None,
                                                             yidx_vec=None,
                                                             logscale=None)
        h1.savefig(out_dir + '/%s_F03_mu_gpetas_slices.pdf' % data_obj.case_name, bbox_inches='tight')
        plt.close(h1)
        h2.savefig(out_dir + '/%s_F03a_mu_gpetas_slices_location.pdf' % data_obj.case_name, bbox_inches='tight')
        plt.close(h2)

        # (4) lambda bar histogram
        if gm_obj is not None:
            lambda_bar_true = np.copy(gm_obj.lambda_bar_true)
        else:
            lambda_bar_true = None
        h1 = gpetas.plotting.plot_hist_lambda_bar(save_obj_GS, lambda_bar_true=lambda_bar_true, quantile=None)
        h1.savefig(out_dir + '/%s_F04_lambda_bar_hist.pdf' % data_obj.case_name, bbox_inches='tight')
        plt.close(h1)
        h1 = gpetas.plotting.plot_lambda_bar(save_obj_GS)
        h1.savefig(out_dir + '/%s_F04a_lambda_bar_chain.pdf' % data_obj.case_name, bbox_inches='tight')
        plt.close(h1)

        # (5) total bg events of X histogram
        h1 = gpetas.plotting.plot_number_of_bg_events(save_obj_GS, mle_obj, mle_obj_silverman, gm_obj, quantile=0.001)
        h1.savefig(out_dir + '/%s_F05_total_number_of_bg_01.pdf' % data_obj.case_name, bbox_inches='tight')
        plt.close(h1)
        h1 = gpetas.plotting.plot_number_of_bg_events(save_obj_GS, mle_obj, mle_obj_silverman, gm_obj, kde_hist=1, pred=1)
        h1.savefig(out_dir + '/%s_F05a_total_number_of_bg_02_poisson_pred.pdf' % data_obj.case_name, bbox_inches='tight')
        plt.close(h1)

        # (6) offspring chains
        h1,h2 = gpetas.plotting.plot_offspring_mcmc_chains(save_obj_GS, gm_obj=gm_obj, mle_obj=mle_obj,
                                                           mle_obj_silverman=mle_obj_silverman,
                                                           theta_true_Kcpadgq=None)
        h1.savefig(out_dir + '/%s_F06_offspring_mcmc_chains_01.pdf' % data_obj.case_name, bbox_inches='tight')
        plt.close(h1)
        h2.savefig(out_dir + '/%s_F06_offspring_mcmc_marginals_02.pdf' % data_obj.case_name, bbox_inches='tight')
        plt.close(h2)
        h1 = gpetas.plotting.plot_scatter_data(np.array(save_obj_GS['theta_tilde']), varnames=None, cmap='seismic', r_value=0.1)
        h1.savefig(out_dir + '/%s_F06a_offspring_mcmc_scatter_plot_01.pdf' % data_obj.case_name, bbox_inches='tight')
        plt.close(h1)
        h1 = gpetas.plotting.plot_scatter(save_obj_GS, mle_obj=mle_obj, mle_obj_silverman=mle_obj_silverman, median_yes=1)
        h1.savefig(out_dir + '/%s_F06_offspring_mcmc_scatter_plot_02.pdf' % data_obj.case_name, bbox_inches='tight')
        plt.close(h1)

        # (6a) autocorrelations of the MCMC chains
        h1 = gpetas.plotting.plot_autocorr(save_obj_GS, idx_samples=None, varnames=None, xlim=None,
                      maxlag=None, label_ESS=None, alpha=None)
        h1.savefig(out_dir + '/%s_F06a_offspring_mcmc_autocorr_plot.pdf' % data_obj.case_name, bbox_inches='tight')
        plt.close(h1)
        if len(save_obj_GS['lambda_bar']) >= 500:
            h1 = gpetas.plotting.plot_autocorr(save_obj_GS, idx_samples=np.arange(0,len(save_obj_GS['lambda_bar']),50), varnames=None, xlim=None,
                                               maxlag=None, label_ESS=None, alpha=None)
            h1.savefig(out_dir + '/%s_F06a_offspring_mcmc_autocorr_thinned_50.pdf' % data_obj.case_name, bbox_inches='tight')
            plt.close(h1)

        # (7) acceptance rate of MH-MCMC of the bg
        h1 = gpetas.plotting.plot_acceptance_offspring(save_obj_GS)
        h1.savefig(out_dir + '/%s_F07_offspring_acceptance_rate.pdf' % data_obj.case_name, bbox_inches='tight')
        plt.close(h1)

        # (8) hyper parameters GP
        h1 = gpetas.plotting.plot_hyperparams(save_obj_GS)
        h1.savefig(out_dir + '/%s_F08_mu_hyperparams.pdf' % data_obj.case_name, bbox_inches='tight')
        plt.close(h1)
        h1 = gpetas.plotting.plot_autocorr_hyperparams(save_obj_GS, idx_samples=None, show_u_level=1)
        h1.savefig(out_dir + '/%s_F08a_mu_hyperparams_autocorr_plot.pdf' % data_obj.case_name, bbox_inches='tight')
        plt.close(h1)
        if len(save_obj_GS['lambda_bar']) >= 500:
            h1 = gpetas.plotting.plot_autocorr_hyperparams(save_obj_GS,
                            idx_samples=np.arange(0,len(save_obj_GS['lambda_bar']),50), show_u_level=1)
            h1.savefig(out_dir + '/%s_mu_hyperparams_autocorr_thinned_50.pdf' % data_obj.case_name, bbox_inches='tight')
            plt.close(h1)

        # (9) gpetas_intensity(X_grid,t)

        # (10) ltest plot
        h1, h2 = gpetas.plotting.plot_l_ltest(save_obj_GS=save_obj_GS,mle_obj=mle_obj,mle_obj_silverman=mle_obj_silverman,
                                              method_posterior_GP='nearest')
        h1.savefig(out_dir + '/%s_F10_l_ltest01.pdf' % data_obj.case_name, bbox_inches='tight')
        h2.savefig(out_dir + '/%s_F10b_l_ltest02.pdf' % data_obj.case_name, bbox_inches='tight')

        # (11) 1D ETAS intensity
        h1 = gpetas.plotting.plot_1D_estimation(save_obj_GS=save_obj_GS, mle_obj=mle_obj,mle_obj_silverman=mle_obj_silverman)
        h1.savefig(out_dir + '/%s_F11_1D_intensity.pdf' % data_obj.case_name, bbox_inches='tight')


        # (50) Priors offspring
        h1 = gpetas.plotting.plot_priors_offspring(save_obj_GS)
        h1.savefig(out_dir + '/%s_F50_priors_offspring.pdf' % data_obj.case_name, bbox_inches='tight')