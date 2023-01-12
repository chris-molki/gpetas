import numpy as np
import os


# writing routines for synthetic cases
def write_results_to_tex_file(t_start=None, t_end=None, fname_dat=None):
    # load text data and write latex file
    aux = np.loadtxt(fname_dat, skiprows=1)
    aux_line = np.vstack((aux, np.mean(aux, axis=0), np.std(aux, axis=0)))
    N_unseen = len(aux[:, 0])

    fid = open(fname_dat[:-4] + '_big_table.tex', 'w')
    fid.write("\\begin{table}[h!]\n")
    fid.write("\\begin{adjustwidth}{-.9in}{-.9in}\n")
    fid.write("\centering\n")
    fid.write("\\tiny\n")
    fid.write(
        "\caption{Predictive log expected likelihood of %i unseen realizations of the Hawkes process with $t_{\\rm start}=$%.1f and $t_{\\rm end}=$%.1f.}\n" % (
            N_unseen, t_start, t_end))
    fid.write("\\begin{tabular}{l%s}\n" % ((N_unseen + 2) * 'c'))
    fid.write("\hline\n")
    str_line = " " + N_unseen * ' & run %i' + " & mean & std \\\ \n"
    fid.write(str_line % (tuple(np.arange(N_unseen) + 1)))
    fid.write("\hline \n")

    str_line = "generative model (gM)" + (N_unseen + 2) * ' & %.1f' + " \\\ \n"
    fid.write(str_line % (tuple(aux_line[:, 1])))
    str_line = "ETAS classical (KDE) (E)" + (N_unseen + 2) * ' & %.1f' + " \\\ \n"
    fid.write(str_line % (tuple(aux_line[:, 2])))
    str_line = "ETAS KDE Silverman (E-S)" + (N_unseen + 2) * ' & %.1f' + " \\\ \n"
    fid.write(str_line % (tuple(aux_line[:, 3])))
    str_line = "GP-ETAS log(E[L]) (GP-E)" + (N_unseen + 2) * ' & \\textbf{%.1f}' + " \\\ \n"
    fid.write(str_line % (tuple(aux_line[:, 4])))
    fid.write("\hline \n")

    str_line = "$N$ all" + (N_unseen + 2) * ' & %i' + " \\\ \n"
    fid.write(str_line % (tuple(aux_line[:, 5])))
    str_line = "$N_{0}$ all" + (N_unseen + 2) * ' & %i' + " \\\ \n"
    fid.write(str_line % (tuple(aux_line[:, 6])))
    str_line = "$N_{\\rm triggered}$ all" + (N_unseen + 2) * ' & %i' + " \\\ \n"
    fid.write(str_line % (tuple(aux_line[:, 7])))
    fid.write("\hline \n")

    str_line = "$N \in [t_1,t_2]$" + (N_unseen + 2) * ' & %i' + " \\\ \n"
    fid.write(str_line % (tuple(aux_line[:, 8])))
    str_line = "$N_0 \in [t_1,t_2]$" + (N_unseen + 2) * ' & %i' + " \\\ \n"
    fid.write(str_line % (tuple(aux_line[:, 9])))
    str_line = "$N_{\\rm triggered} \in [t_1,t_2]$" + (N_unseen + 2) * ' & %i' + " \\\ \n"
    fid.write(str_line % (tuple(aux_line[:, 10])))
    fid.write("\hline \n")

    str_line = "$N_{0}/N$" + (N_unseen + 2) * ' & %.2f' + " \\\ \n"
    fid.write(str_line % (tuple(aux_line[:, 6] / aux_line[:, 5])))
    str_line = "$N_{triggered}/N_0$" + (N_unseen + 2) * ' & %.2f' + " \\\ \n"
    fid.write(str_line % (tuple(aux_line[:, 7] / aux_line[:, 6])))
    fid.write("\hline \n")

    str_line = "$\\int\\int \\lambda$ :gm" + (N_unseen + 2) * ' & %.1f' + " \\\ \n"
    fid.write(str_line % (tuple(aux_line[:, 11])))
    str_line = "$\\int\\int \\mu$ :gm" + (N_unseen + 2) * ' & %.1f' + " \\\ \n"
    fid.write(str_line % (tuple(aux_line[:, 12])))
    str_line = "$\\int\\int \\sum \\varphi_{ij}$ :gm" + (N_unseen + 2) * ' & %.1f' + " \\\ \n"
    fid.write(str_line % (tuple(aux_line[:, 13])))
    str_line = "$\\sum_i \\ln(\\mu(x_i)+\\sum \\varphi_{ij})$ :gm" + (N_unseen + 2) * ' & %.1f' + " \\\ \n"
    fid.write(str_line % (tuple(aux_line[:, 14])))
    fid.write("\hline \n")

    str_line = "$\\int\\int \\lambda$ :classical(KDE)" + (N_unseen + 2) * ' & %.1f' + " \\\ \n"
    fid.write(str_line % (tuple(aux_line[:, 15])))
    str_line = "$\\int\\int \\mu$ :classical(KDE)" + (N_unseen + 2) * ' & %.1f' + " \\\ \n"
    fid.write(str_line % (tuple(aux_line[:, 16])))
    str_line = "$\\int\\int \\sum \\varphi_{ij}$ :classical(KDE)" + (N_unseen + 2) * ' & %.1f' + " \\\ \n"
    fid.write(str_line % (tuple(aux_line[:, 17])))
    str_line = "$\\sum_i \\ln(\\mu(x_i)+\\sum \\varphi_{ij})$ :classical(KDE)" + (N_unseen + 2) * ' & %.1f' + " \\\ \n"
    fid.write(str_line % (tuple(aux_line[:, 18])))
    fid.write("\hline \n")

    str_line = "$\\int\\int \\lambda$ :silverman(KDE)" + (N_unseen + 2) * ' & %.1f' + " \\\ \n"
    fid.write(str_line % (tuple(aux_line[:, 19])))
    str_line = "$\\int\\int \\mu$ :silverman(KDE)" + (N_unseen + 2) * ' & %.1f' + " \\\ \n"
    fid.write(str_line % (tuple(aux_line[:, 20])))
    str_line = "$\\int\\int \\sum \\varphi_{ij}$ :silverman(KDE)" + (N_unseen + 2) * ' & %.1f' + " \\\ \n"
    fid.write(str_line % (tuple(aux_line[:, 21])))
    str_line = "$\\sum_i \\ln(\\mu(x_i)+\\sum \\varphi_{ij})$ :silverman(KDE)" + (N_unseen + 2) * ' & %.1f' + " \\\ \n"
    fid.write(str_line % (tuple(aux_line[:, 22])))
    fid.write("\hline \n")

    str_line = "seed" + (N_unseen) * ' & %i' + " & --- \\\ \n"
    fid.write(str_line % (tuple(aux[:, 0])))
    fid.write("\hline \n")
    fid.write("\end{tabular} \n")
    fid.write("\end{adjustwidth} \n")
    fid.write("\end{table}\n")
    fid.close()

    fid = open(fname_dat[:-4] + '_mean.tex', 'w')
    fid.write("\\begin{table}\n")
    fid.write("\\begin{adjustwidth}{-.9in}{-.9in}\n")
    fid.write("\centering\n")
    fid.write("\\small\n")
    fid.write(
        "\caption{Mean predictive log expected likelihood of %i unseen realizations of the Hawkes process with $t_{start}=$%.1f and $t_{end}=$%.1f.}\n" % (
            N_unseen, t_start, t_end))
    fid.write("\\begin{tabular}{c%s}\n" % ((4) * 'c'))
    fid.write("\hline\n")
    str_line = "gM & E &  E-S & GP-E log(E[L]) \\\ \n"
    fid.write(str_line)
    fid.write("\hline \n")
    fid.write("%.1f & %.1f & %.1f & \\textbf{%.1f} \\\ \n" % (
        aux_line[-2, 1], aux_line[-2, 2], aux_line[-2, 3], aux_line[-2, 4]))
    fid.write("\hline \n")
    fid.write("\end{tabular} \n")
    fid.write("\end{adjustwidth} \n")
    fid.write("\end{table}\n")
    fid.close()
    return


def write_table2_theta(save_obj_GS, gm_obj, mle_obj, mle_obj_silverman, fout_dir=None, idx_samples=None):
    """

    :param save_obj_GS:
    :type save_obj_GS:
    :param gm_obj:
    :type gm_obj:
    :param mle_obj:
    :type mle_obj:
    :param mle_obj_silverman:
    :type mle_obj_silverman:
    :param fout_dir:
    :type fout_dir:
    :param idx_samples:
    :type idx_samples:
    :return:
    :rtype:
    """
    if fout_dir is None:
        fout_dir = "output/tables"
    if not os.path.isdir(fout_dir):
        os.mkdir(fout_dir)
    if idx_samples is None:
        idx_samples = np.arange(0, len(save_obj_GS['lambda_bar']))
    Ksamples = len(idx_samples)

    # get theta_tilde according to idx_samples
    theta_tilde_array = np.empty([Ksamples, len(save_obj_GS['theta_tilde'][0])])
    for i in range(len(idx_samples)):
        k = idx_samples[i]
        theta_tilde_array[i, :] = save_obj_GS['theta_tilde'][k]

    # some quantities
    quantile = 0.05
    theta_high = np.quantile(theta_tilde_array, q=1 - quantile, axis=0)
    theta_median = np.quantile(theta_tilde_array, q=0.5, axis=0)
    theta_low = np.quantile(theta_tilde_array, q=quantile, axis=0)
    L1, L2, L2a, L3, L4, L5 = np.empty(6) * np.nan

    if gm_obj is not None:
        K, c, p, alpha, D, gamma, q = gm_obj.theta_Kcpadgq
        L1 = "gM  & %.4f & %.4f & %.2f & %.3f & %.3f  & %.2f   & %.2f \\\ " % (K, c, p, alpha, D, gamma, q)
        if mle_obj is not None:
            K, c, p, alpha, D, gamma, q = mle_obj.theta_mle_Kcpadgq
            L2 = "E & %.4f & %.4f & %.2f & %.3f & %.3f  & %.2f   & %.2f \\\ " % (K, c, p, alpha, D, gamma, q)
        if mle_obj_silverman is not None:
            K, c, p, alpha, D, gamma, q = mle_obj_silverman.theta_mle_Kcpadgq
            L2a = "E-S  & %.4f & %.4f & %.2f & %.3f & %.3f  & %.2f   & %.2f \\\ " % (K, c, p, alpha, D, gamma, q)
        K, c, p, alpha, D, gamma, q = theta_high
        L3 = " & %.4f & %.4f & %.2f & %.3f & %.3f  & %.2f   & %.2f \\\ " % (
            K, c, p, alpha, D, gamma, q)
        K, c, p, alpha, D, gamma, q = theta_median
        L4 = "GP-E  & %.4f & %.4f & %.2f & %.3f & %.3f  & %.2f   & %.2f \\\ " % (K, c, p, alpha, D, gamma, q)
        K, c, p, alpha, D, gamma, q = theta_low
        L5 = "  & %.4f & %.4f & %.2f & %.3f & %.3f  & %.2f   & %.2f \\\ " % (K, c, p, alpha, D, gamma, q)
    if gm_obj is None:
        if mle_obj is not None:
            K, c, p, alpha, D, gamma, q = mle_obj.theta_mle_Kcpadgq
            L2 = "E & %.4f & %.4f & %.2f & %.3f & %.4f  & %.2f   & %.2f \\\ " % (K, c, p, alpha, D, gamma, q)
        if mle_obj_silverman is not None:
            K, c, p, alpha, D, gamma, q = mle_obj_silverman.theta_mle_Kcpadgq
            L2a = "E-S  & %.4f & %.4f & %.2f & %.3f & %.4f  & %.2f   & %.2f \\\ " % (K, c, p, alpha, D, gamma, q)
        K, c, p, alpha, D, gamma, q = theta_high
        L3 = " & %.4f & %.4f & %.2f & %.3f & %.4f  & %.2f   & %.2f \\\ " % (
            K, c, p, alpha, D, gamma, q)
        K, c, p, alpha, D, gamma, q = theta_median
        L4 = "GP-E  & %.4f & %.4f & %.2f & %.3f & %.4f  & %.2f   & %.2f \\\ " % (K, c, p, alpha, D, gamma, q)
        K, c, p, alpha, D, gamma, q = theta_low
        L5 = "  & %.4f & %.4f & %.2f & %.3f & %.4f  & %.2f   & %.2f \\\ " % (K, c, p, alpha, D, gamma, q)

    fid = open(fout_dir + '/table002_theta.tex', 'w')
    fid.write("\\begin{table}[h!]\n")
    fid.write("\centering\n")
    fid.write("\small\n")
    if gm_obj is not None:
        fid.write(
            "\caption{Case %i: Parameter values $\\thetas_{\\varphi}$ of the triggering function. Based on $K=$%i samples.}" % (
                np.int(gm_obj.case_name[-2:]), Ksamples))
    else:
        fid.write(
            "\caption{Real data: Parameter values $\\thetas_{\\varphi}$ of the triggering function. Based on $K=$%i samples.}" % (
                Ksamples))
    fid.write("\n")
    fid.write("\\begin{tabular}{lccccccc}")
    fid.write("\n")
    fid.write("\hline")
    fid.write("\n")
    fid.write("model & $K_0$ & $c$ & $p$ & $\\alpha$ & $d$ & $\gamma$ & $q$  \\\ \hline")
    fid.write("\n")
    if gm_obj is not None:
        fid.write(L1)
        fid.write("\hline\n")
    if mle_obj is not None:
        fid.write(L2)
        fid.write("\hline\n")
    if mle_obj_silverman is not None:
        fid.write(L2a)
        fid.write("\hline\n")
    fid.write(L4)
    fid.write("\n")
    fid.write(L5)
    fid.write("\n")
    fid.write(L3)
    fid.write("\hline\n")
    fid.write("\n")
    fid.write("\end{tabular}\n")
    fid.write("\end{table}\n")
    fid.close()

    return


def write_table3_l2_background(save_obj_GS, gm_obj, mle_obj, mle_obj_silverman, fout_dir=None, idx_samples=None):

    if mle_obj is None:
        exit()
    if mle_obj_silverman is None:
        exit()

    # output directory
    if fout_dir is None:
        fout_dir = "output/tables"
    if not os.path.isdir(fout_dir):
        os.mkdir(fout_dir)
    if idx_samples is None:
        idx_samples = np.arange(0, len(save_obj_GS['lambda_bar']))
    Ksamples = len(idx_samples)

    # get theta_tilde according to idx_samples
    mu_grid_array = np.empty([Ksamples, len(save_obj_GS['mu_grid'][0])])
    for i in range(len(idx_samples)):
        k = idx_samples[i]
        mu_grid_array[i, :] = save_obj_GS['mu_grid'][k]

    # some quantities
    residuals_SGCP = gm_obj.mu_grid - np.median(mu_grid_array, axis=0)
    residuals_KDE_default = gm_obj.mu_grid - mle_obj.mu_grid
    residuals_KDE_silverman = gm_obj.mu_grid - mle_obj_silverman.mu_grid
    L = len(gm_obj.mu_grid)
    absX = np.prod(np.diff(gm_obj.domain.X_borders))
    l2_SGCP = np.sqrt(absX / L * np.sum(residuals_SGCP ** 2.))
    l2_kde = np.sqrt(absX / L * np.sum(residuals_KDE_default ** 2.))
    l2_kde_silverman = np.sqrt(absX / L * np.sum(residuals_KDE_silverman ** 2.))

    fid = open(fout_dir + '/table003_l2_background.tex', 'w')
    fid.write("\\begin{table}[h!]\n")
    fid.write("\centering\n")
    fid.write("\small\n")
    fid.write(
        "\caption{Case %0i: Comparison of estimated background intensity. Approximated $\ell_2$ norm (per unit time [1/day]) to the true function over a rectangular grid with $L=50^2$ grid cells; $\ell_2\\approx \\ \sqrt{\\frac{|X|}{L}\sum_{l=1}^L (\mu^*_l-\mu_l)^2}$, $\mu_l$ is the value of the true functions of the $l$th grid cell, $\mu^*_l$ the estimates from the classical KDE-ETAS model or the mean value of posterior predictive distribution of the GP-ETAS background model. Based on $K=$ %i samples.}\n" % (
            np.int(gm_obj.case_name[-2:]), Ksamples))
    fid.write("\\begin{tabular}{lccc}\n")
    fid.write("\hline\n")
    fid.write(
        "criterium  & ETAS classical KDE & ETAS Silverman KDE & GP-ETAS $\\ell_2(\\mu-{\\rm median}[\\hat\\mu_{\\rm GPetas}])$ \\\ \hline\n")
    fid.write("$\ell_2$   & %.4f    & %.4f      & %.4f \\\ \n" % (l2_kde, l2_kde_silverman, l2_SGCP))
    fid.write("normalized & %.2f    & %.2f      & %.0f \\\ \n" % (
        l2_kde / l2_SGCP, l2_kde_silverman / l2_SGCP, l2_SGCP / l2_SGCP))
    fid.write("\hline\n")
    fid.write("\end{tabular} \n")
    fid.write("\end{table}\n")
    fid.close()

    return
