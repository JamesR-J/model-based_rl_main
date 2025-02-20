from ml_collections import config_dict


def get_MPC_config():
    config = config_dict.ConfigDict()

    config.PLANNING_HORIZON = 25

    config.INIT_VAR_DIVISOR = 4  # TODO what is this and where should it be?

    config.BASE_NSAMPS = 20  # 25  # TODO not sure what sets this value
    config.GAMMA = 1.25  # This is something else not discount value
    config.DISCOUNT_FACTOR = 1.0
    config.N_ELITES = 3
    config.iCEM_ITERS = 3
    config.XI = 0.3

    config.ACTIONS_PER_PLAN = 6  # How many actions to keep out of each total plan iteration, must be less than planning horizon

    config.BETA = 3.0

    """
    Current iCEM adaptions
    Use constant batch size
    Also use constant sample size for iCEM
    Check saved plan and best plan are the same thing, is there anyway the best plan may not be saved?
    """

    return config