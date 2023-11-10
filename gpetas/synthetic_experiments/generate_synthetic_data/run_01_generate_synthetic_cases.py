# imports
import gpetas
import numpy as np

if __name__ == '__main__':
    # generative model case01
    gm_obj = gpetas.generative_model.generate_synthetic_data()
    gm_obj.sim_case_01(save_yes=1)

    # generative model case02
    gm_obj = gpetas.generative_model.generate_synthetic_data(
        T_borders_all=np.array([0., 4000.]), T_borders_training=np.array([0., 4000.]),
        T_borders_testing=np.array([0., 4000.]))
    gm_obj.sim_case_02(save_yes=1)

    # generative model case03
    gm_obj = gpetas.generative_model.generate_synthetic_data()
    gm_obj.sim_case_03_gaborlike(save_yes=1)
