from ml_collections import config_dict


def get_MPC_config():
    config = config_dict.ConfigDict()

    config.PLANNING_HORIZON = 20

    config.INIT_VAR_DIVISOR = 4  # TODO what is this and where should it be?

    config.BASE_NSAMPS = 15
    config.GAMMA = 0.99
    config.N_ELITES = 10

    config.BETA = 3.0

    return config