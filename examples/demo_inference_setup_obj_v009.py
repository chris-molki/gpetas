import gpetas
import numpy as np
import os
import sys
import argparse
import pprint




''' start main '''
if __name__ == '__main__':
    """
    Sampling and MLE inference of real data: case_name data GS_setup.bin
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--fsetup", type=str, help="file name of the setup file which includes the data_obj",required=True)
    parser.add_argument("--mle_yes", type=str, help="any string, in order to NOT compute mle")
    parser.add_argument("--mle_yes_silverman", type=str, help="any string, in order to NOT compute mle with hmin Silverman")
    parser.add_argument("--mle_Nnearest", type=int, help="MLE: number of nearest neighbours considered for the individual bandwidth of kde background intensity")
    parser.add_argument("--mle_hmin", type=float, help="MLE: minimum bandwith hmin of kde background intensity")

    args = parser.parse_args()

    # input setup file including the data
    fname_setup = args.fsetup

    # Gibbs sampling based on setup file
    setup_obj = np.load(fname_setup,allow_pickle=True)

    # get inputs
    case_name = setup_obj.case_name
    data_obj = setup_obj.data_obj
    burnin=setup_obj.burnin
    Ksamples=setup_obj.Ksamples
    num_iterations=setup_obj.num_iterations
    thinning=setup_obj.thinning
    MH_proposals_offspring=setup_obj.MH_proposals_offspring
    MH_cov_empirical_yes=setup_obj.MH_cov_empirical_yes
    kth_sample_obj=setup_obj.kth_sample_obj
    prior_theta_dist=setup_obj.prior_theta_dist
    prior_theta_params=setup_obj.prior_theta_params
    stable_theta_sampling=setup_obj.stable_theta_sampling


    # init Gibbs sampler
    GS = gpetas.Gibbs_sampler.GS_ETAS(data_obj=data_obj, setup_obj=setup_obj,
                                      burnin=burnin, num_samples=num_iterations,
                                      thinning=thinning,
                                      MH_proposals_offspring=MH_proposals_offspring,
                                      MH_cov_empirical_yes=MH_cov_empirical_yes,
                                      kth_sample_obj=kth_sample_obj)


    # some info to the screen
    import pprint
    print('---------------------------------------')
    print('case name                =',case_name)
    print('---------------------------------------')
    #print('dim                      =', GS.dim)
    print('---------------------------------------')
    print('domain from data_obj')
    pprint.pprint(vars(data_obj.__dict__['domain']))
    print('---------------------------------------')
    print('burnin                   = ',burnin)
    print('Ksamples                 = ',Ksamples)
    print('thinning                 = ',thinning)
    print('Ntotal iter              = ',num_iterations)
    print('MH_proposals_offspring   = ',MH_proposals_offspring)
    print('MH_cov_empirical_yes     = ',MH_cov_empirical_yes)
    print('kth_sample_obj           = ',kth_sample_obj)
    print('---------------------------------------')
    print('theta prior:                           ')
    print('prior_theta_dist         =',prior_theta_dist)
    print('prior_theta_params       =',prior_theta_params)
    print('stable_theta_sampling    =',stable_theta_sampling)



    # START Gibbs sampler based on settings in setup_obj
    #(3.2) sampling
    GS.sample()


    # mle inference

    # from setup_obj directly
    if args.mle_yes is None:
        if hasattr(setup_obj,'setup_obj_mle'):
            gpetas.mle_KDE_ETAS.mle_units(data_obj=setup_obj.data_obj, setup_obj=setup_obj.setup_obj_mle)
            args.mle_yes = 'no'
    if args.mle_yes_silverman is None:
        if hasattr(setup_obj,'setup_obj_mle_silverman'):
            gpetas.mle_KDE_ETAS.mle_units(data_obj=setup_obj.data_obj,setup_obj=setup_obj.setup_obj_mle_silverman,
                                          fout_name="mle_silverman_hmin_")
            args.mle_yes_silverman = 'no'

    # if setup_obj has no attribute setub_obj_mle
    case_name = setup_obj.case_name
    Nnearest=15
    h_min_degree=0.05
    if case_name == 'gron':
        h_min_degree = 100.
    data_obj = setup_obj.data_obj
    theta_start_Kcpadgqbm0 = setup_obj.theta_start_Kcpadgqbm0
    spatial_offspring = setup_obj.spatial_offspring
    bins = int(np.sqrt(setup_obj.X_grid.shape[0]))
    X_grid = setup_obj.X_grid
    stable_theta = setup_obj.stable_theta_sampling

    # mle default
    outdir = setup_obj.outdir
    if args.mle_yes is None:
        #(2.1) default: h_min = small and fixed from before
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
                    case_name=case_name)
        print(setup_obj_mle.outdir)
        mle_obj = gpetas.mle_KDE_ETAS.mle_units(data_obj=data_obj, setup_obj=setup_obj_mle)

    # mle silverman
    if args.mle_yes_silverman is None:
        #(2.2) silverman: h_min = via silverman rule
        h_min_degree = gpetas.some_fun.silverman_scott_rule_d(data_obj.data_all.positions)
        setup_obj_mle_silverman = gpetas.mle_KDE_ETAS.setup_mle(data_obj=data_obj,
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
                    silverman='yes')
        print(setup_obj_mle_silverman.outdir)
        mle_obj_silverman = gpetas.mle_KDE_ETAS.mle_units(data_obj=data_obj, setup_obj=setup_obj_mle_silverman,fout_name="mle_silverman_hmin_")


    # (4) plotting results
    sampler_plot_summary_yes = 'yes'
    if sampler_plot_summary_yes is not None:
        # extract from path to GSsave.bin file
        root_dir = './output'
        fname_in_GS_save = root_dir + "/inference_results/GS_save_data_%s.bin"%case_name
        path_name=os.path.dirname(fname_in_GS_save)
        output_dir = path_name#os.path.basename(path_name)
        print(output_dir)
        output_dir_tables = output_dir + "/%s_tables"%case_name
        if not os.path.isdir(output_dir_tables):
            os.mkdir(output_dir_tables)
        output_dir_figures = output_dir + "/%s_figures"%case_name
        if not os.path.isdir(output_dir_figures):
            os.mkdir(output_dir_figures)

        # generative model
        gm_obj = None
        if case_name=='case_01':
            gm_obj = gpetas.generative_model.generate_synthetic_data()
            gm_obj.sim_case_01()
        if case_name=='case_02':
            gm_obj = gpetas.generative_model.generate_synthetic_data()
            gm_obj.sim_case_02()
        if case_name=='case_03':
            gm_obj = gpetas.generative_model.generate_synthetic_data()
            gm_obj.sim_case_03_gaborlike()

        # load relevant results
        data_obj = None
        save_obj_GS = None
        mle_obj = None
        mle_obj_silverman = None
        if os.path.exists(output_dir+'/data_obj_%s.all'%case_name):
            data_obj = np.load(output_dir + '/data_obj_%s.all' % case_name, allow_pickle=True)
        if os.path.exists(output_dir + '/GS_save_data_%s.bin' % case_name):
            save_obj_GS = np.load(output_dir + '/GS_save_data_%s.bin' % case_name, allow_pickle=True)
        if os.path.exists(output_dir+'/mle_default_hmin_%s.all'%case_name):
            mle_obj = np.load(output_dir+'/mle_default_hmin_%s.all'%case_name,allow_pickle=True)
        if os.path.exists(output_dir + '/mle_silverman_hmin_%s.all' % case_name):
            mle_obj_silverman = np.load(output_dir + '/mle_silverman_hmin_%s.all' % case_name, allow_pickle=True)

        # idx_samples
        idx_samples = np.arange(0,len(save_obj_GS['lambda_bar']))


        # sampler results
        gpetas.summary.summary_gpetas(save_obj_GS=save_obj_GS,gm_obj=gm_obj,
                                      mle_obj=mle_obj,mle_obj_silverman=mle_obj_silverman,
                                      fout_dir=output_dir_figures,case_name=case_name)

        # tables
        gpetas.synthetic_cases_aux_functions.write_table2_theta(save_obj_GS, gm_obj, mle_obj, mle_obj_silverman,
                                                                fout_dir=output_dir_tables,
                                                                idx_samples=idx_samples)
        if gm_obj is not None:
            gpetas.synthetic_cases_aux_functions.write_table3_l2_background(save_obj_GS, gm_obj, mle_obj,
                                                                        mle_obj_silverman, fout_dir=output_dir_tables,
                                                                        idx_samples=idx_samples)

        print('done')




    print('done')
