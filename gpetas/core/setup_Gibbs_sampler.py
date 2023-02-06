import numpy as np
import os
import gpetas
import pickle


class setup_sampler():
    def __init__(self, data_obj, utm_yes=None, spatial_offspring='R', theta_start_Kcpadgq=None,
                 sigma_proposal_offspring_params=None, ngrid_per_dim=50, cov_params=None, sigma_proposal_hypers=None,
                 mu_nu0=None, X_grid=None, outdir=None, prior_theta_dist=None, prior_theta_params=None,
                 stable_theta_sampling=None, time_origin=None, case_name='case_01', burnin=None, Ksamples=None,
                 num_iterations=None, thinning=None, MH_proposals_offspring=None, MH_cov_empirical_yes=None,
                 kth_sample_obj=None, corresponding_mle=None):
        """
        :param data_obj:
        :type data_obj:
        :param utm_yes:
        :type utm_yes:
        :param spatial_offspring:
        :type spatial_offspring:
        :param theta_start_Kcpadgq:
        :type theta_start_Kcpadgq:
        :param sigma_proposal_offspring_params:
        :type sigma_proposal_offspring_params:
        :param ngrid_per_dim:
        :type ngrid_per_dim:
        :param cov_params:
        :type cov_params:
        :param sigma_proposal_hypers:
        :type sigma_proposal_hypers:
        :param mu_nu0:
        :type mu_nu0:
        :param X_grid:
        :type X_grid:
        :param outdir:
        :type outdir:
        :param prior_theta_dist:
        :type prior_theta_dist:
        :param prior_theta_params:
        :type prior_theta_params:
        :param stable_theta_sampling:
        :type stable_theta_sampling:
        :param time_origin:
        :type time_origin:
        :param case_name:
        :type case_name:
        :param burnin:
        :type burnin:
        :param num_iterations:
        :type num_iterations:
        :param thinning:
        :type thinning:
        :param MH_proposals_offspring:
        :type MH_proposals_offspring:
        :param MH_cov_empirical_yes:
        :type MH_cov_empirical_yes:
        :param kth_sample_obj:
        :type kth_sample_obj:
        :param corresponding_mle: dict containing keywords for setup_obj_mle
        :type corresponding_mle: dictionary as corresponding_mle={'type':'default','Nnearest': 20}
        """

        """
        inititialization as in the paper
        :type data_obj_all: all data object including training and test data
        :type data_obj_training: only training data object
        self.spatial_offspring = 'R' # rupture length power law 'R'
        self.spatial_offspring = 'P' # classical power law 'P'
        self.spatial_offspring = 'G' # Gaussian decay 'G'
        """
        # spatial coordinates
        self.utm_yes = utm_yes
        if utm_yes == 1:
            data_obj.domain.X_borders = np.copy(data_obj.domain.X_borders_UTM_km)
            data_obj.data_all.positions = np.copy(data_obj.data_all.positions_UTM_km)
            print('Note: all positions are in UTM now!')

        # get training data
        T1, T2 = data_obj.domain.T_borders_training
        idx = np.where((data_obj.data_all.times >= T1) & (data_obj.data_all.times <= T2))
        data_obj_training = data_structure()
        data_obj_training.times = data_obj.data_all.times[idx]
        data_obj_training.magnitudes = data_obj.data_all.magnitudes[idx]
        data_obj_training.positions = data_obj.data_all.positions[np.squeeze(idx), :]
        data_obj_training.positions[:, 0] = data_obj_training.positions[:, 0] - data_obj.domain.X_borders[0, 0]
        data_obj_training.positions[:, 1] = data_obj_training.positions[:, 1] - data_obj.domain.X_borders[1, 0]
        self.N_training = len(np.squeeze(idx))

        # start: bg rate at all data points
        absX = np.prod(np.diff(data_obj.domain.X_borders))
        absT_training = np.diff(data_obj.domain.T_borders_training)
        self.mu0_start = (len(data_obj_training.times) / 2. / absX / absT_training).item()
        self.mu0_grid = None

        # ETAS offspring parameters
        # start: mark distribution params:
        m0 = data_obj.domain.m0
        if np.min(data_obj_training.magnitudes) < m0:
            m0 = np.min(data_obj_training.magnitudes)
            print('Warning: m0 is set to m0=', m0)
        self.m0 = m0
        self.m_beta = 1. / np.mean(data_obj_training.magnitudes - self.m0)  # or default np.log(10.)
        # start: ETAS offspring time params: theta_tilde = [0.01, 0.01, 1.5, 2.3, 0.1, 0.5, 2., self.m_beta, self.m0]
        if theta_start_Kcpadgq is None:
            # prior_theta_uniform = np.array([[1e-7, 10], [1e-7, 10], [1e-7, 10], [1e-7, 10], [1e-7, 10], [1e-7, 10], [1., 10]])
            # self.theta_start_Kcpadgqbm0 = np.array([0.01, 0.01, 1.2, 2.3, 0.05, 0.5, 2., self.m_beta, self.m0])
            # self.theta_start_Kcpadgqbm0[:7]=np.random.uniform(prior_theta_uniform[:,0],prior_theta_uniform[:,1])
            # self.theta_start_Kcpadgqbm0 = np.array([0.01, 0.01, 1.2, 2.3, 0.05, 0.5, 2., self.m_beta, self.m0])
            self.theta_start_Kcpadgqbm0 = np.array(
                [0.01 / 4., 0.01, 1.2, 2.3 - 0.5, 0.05, 0.5, 2., self.m_beta, self.m0])
            # fac = 3.#np.random.uniform(1.5, 2.5)
            # print('factor start=',fac)
            # self.theta_start_Kcpadgqbm0[:6] = fac*self.theta_start_Kcpadgqbm0[:6]
        if theta_start_Kcpadgq is not None:
            self.theta_start_Kcpadgqbm0 = np.zeros(9)
            self.theta_start_Kcpadgqbm0[:-2] = theta_start_Kcpadgq
            self.theta_start_Kcpadgqbm0[-2] = self.m_beta
            self.theta_start_Kcpadgqbm0[-1] = self.m0
        if self.utm_yes is not None:
            self.theta_start_Kcpadgqbm0[4] = self.theta_start_Kcpadgqbm0[4] * 111.
        self.spatial_offspring = spatial_offspring
        if sigma_proposal_offspring_params is None:
            sigma_proposal_offspring_params = 0.0001
        self.sigma_proposal_offspring_params = sigma_proposal_offspring_params

        # background sampler
        self.cov_params = cov_params
        if self.cov_params is None:
            nu_lengthscale_start = gpetas.some_fun.silverman_scott_rule_d(data_obj_training.positions)
            self.cov_params = [np.array([5.]), np.array([nu_lengthscale_start, nu_lengthscale_start])]
        self.cov_params_start = self.cov_params

        # hyper-prior background sampler:
        self.mu_upper_bound = None  # automated via counts in a 2d histogram
        self.std_factor = 1.  # coefficient of variation sigma/mean of a Gamma distribution
        self.mu_nu0 = mu_nu0  # mean of exponential distribution for the amplitude of cov-func
        self.mu_length_scale = None  # determines the mean of the length scale means c*deltaX_dim
        if sigma_proposal_hypers is None:
            sigma_proposal_hypers = .05
        self.sigma_proposal_hypers = sigma_proposal_hypers  # width of hyperparam proposal distribution in log units
        X_borders_NN = data_obj.domain.X_borders - np.array(
            [[data_obj.domain.X_borders[0, 0], data_obj.domain.X_borders[0, 0]],
             [data_obj.domain.X_borders[1, 0], data_obj.domain.X_borders[1, 0]]])
        self.X_grid = X_grid
        self.X_grid_NN = gpetas.some_fun.make_X_grid(X_borders_NN, nbins=ngrid_per_dim)
        if self.X_grid is None:
            self.X_grid = gpetas.some_fun.make_X_grid(data_obj.domain.X_borders, nbins=ngrid_per_dim)

        # prior theta offspring
        self.prior_theta_dist = prior_theta_dist
        self.prior_theta_params = prior_theta_params
        self.stable_theta_sampling = stable_theta_sampling

        # parameters related to the sampler and data
        self.data_obj = data_obj
        self.time_origin = time_origin
        self.case_name = case_name
        self.burnin = burnin
        self.Ksamples = Ksamples
        self.num_iterations = num_iterations
        self.thinning = thinning
        self.MH_proposals_offspring = MH_proposals_offspring
        self.MH_cov_empirical_yes = MH_cov_empirical_yes
        self.kth_sample_obj = kth_sample_obj

        # create subdirectory for output
        if outdir is None:
            outdir = './inference_results'
        if os.path.isdir(outdir):
            print('Output subdirectory exists')
        else:
            os.mkdir(outdir)
            print('Output subdirectory has been created.')
        self.outdir = outdir

        # write to file
        fname_setup_obj = outdir + "/setup_obj_%s.all" % (case_name)
        file = open(fname_setup_obj, "wb")  # remember to open the file in binary mode
        pickle.dump(self, file)
        file.close()
        print('setup_obj has been created and saved:', fname_setup_obj)

        if corresponding_mle is not None:
            self.setup_obj_mle = setup_obj_mle_from_setup_obj_GS(self, **corresponding_mle)
            corresponding_mle['silverman'] = 'yes'
            corresponding_mle['h_min_degree'] = gpetas.some_fun.silverman_scott_rule_d(data_obj.data_all.positions)
            self.setup_obj_mle_silverman = setup_obj_mle_from_setup_obj_GS(self, **corresponding_mle)




# get kth-sample
class kth_sample():
    def __init__(self, mu_grid, lambda_bar, cov_params, theta_phi, spatial_offspring, X_grid_NN=None,
                 save_to_file=None):
        self.mu_grid = mu_grid
        self.lambda_bar = lambda_bar
        self.cov_params = cov_params
        self.theta_phi = theta_phi
        self.spatial_offspring = spatial_offspring
        self.X_grid_NN = X_grid_NN
        if save_to_file is not None:
            file = open("save_obj_kth_sample.obj", "wb")
            pickle.dump(self, file)
            file.close()


def get_last_sample(save_obj_GS, k=-1):
    mu_grid = save_obj_GS['mu_grid'][k]
    lambda_bar = save_obj_GS['lambda_bar'][k]
    cov_params = save_obj_GS['cov_params'][k]
    theta_phi = save_obj_GS['theta_tilde'][k]
    spatial_offspring = save_obj_GS['setup_obj'].spatial_offspring
    X_grid_NN = save_obj_GS['X_grid_NN']
    kth_sample_obj = kth_sample(mu_grid, lambda_bar, cov_params, theta_phi, spatial_offspring, X_grid_NN)
    return kth_sample_obj


def get_kth_sample_from_mle(mle_obj):
    # lambda_bar should be limited due to computational reasons
    lambda_bar = np.max(mle_obj.mu_grid)
    mu_grid = mle_obj.mu_grid
    theta_phi = mle_obj.theta_mle_Kcpadgq
    cov_params = [np.array([20.]), np.array([np.mean(mle_obj.h_i_vec), np.mean(mle_obj.h_i_vec)])]
    spatial_offspring = mle_obj.setup_obj.spatial_offspring
    X_grid_NN = mle_obj.X_grid_NN
    kth_sample_obj = kth_sample(mu_grid, lambda_bar, cov_params, theta_phi, spatial_offspring, X_grid_NN)
    return kth_sample_obj


# subclasses
class data_structure():
    def __init__(self):
        self.times = None
        self.magnitudes = None
        self.positions = None


# setup_obj_mle
def setup_obj_mle_from_setup_obj_GS(setup_obj, **kwargs):
    case_name = setup_obj.case_name
    Nnearest = 15
    if 'Nnearest' in kwargs:
        Nnearest = kwargs['Nnearest']
    h_min_degree = 0.05
    if 'h_min_degree' in kwargs:
        h_min_degree = kwargs['h_min_degree']
    # if case_name == 'gron':
    #    h_min_degree = 100.
    data_obj = setup_obj.data_obj
    theta_start_Kcpadgqbm0 = setup_obj.theta_start_Kcpadgqbm0
    if 'theta_start_Kcpadgq' in kwargs:
        theta_start_Kcpadgqbm0[:7] = kwargs['theta_start_Kcpadgq']
    spatial_offspring = setup_obj.spatial_offspring
    bins = int(np.sqrt(setup_obj.X_grid.shape[0]))
    X_grid = setup_obj.X_grid
    stable_theta = setup_obj.stable_theta_sampling
    silverman = None
    if 'silverman' in kwargs:
        silverman = kwargs['silverman']

    # mle default
    outdir = setup_obj.outdir

    # (2.1) default: h_min = small and fixed from before
    setup_obj_mle = gpetas.mle_KDE_ETAS.setup_mle(data_obj=data_obj,
                                                  theta_start_Kcpadgqbm0=theta_start_Kcpadgqbm0,
                                                  spatial_offspring=spatial_offspring,
                                                  Nnearest=Nnearest,
                                                  h_min_degree=h_min_degree,
                                                  spatial_units='degree',
                                                  utm_yes=None,
                                                  bins=bins,
                                                  X_grid=X_grid,
                                                  outdir=outdir,
                                                  stable_theta=stable_theta,
                                                  case_name=case_name,
                                                  silverman=silverman)
    return setup_obj_mle
