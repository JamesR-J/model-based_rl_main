from ml_collections import config_dict


def get_MPC_config():
    config = config_dict.ConfigDict()

    config.PLANNING_HORIZON = 25

    config.INIT_VAR_DIVISOR = 4  # TODO what is this and where should it be?

    config.BASE_NSAMPS = 15
    config.GAMMA = 0.99
    config.DISCOUNT_FACTOR = 0.99  # TODO what is the diff to the above?
    config.N_ELITES = 10
    config.iCEM_ITERS = 3
    config.XI = 4.0  # TODO not sure what this is tbh

    config.BETA = 3.0

    return config