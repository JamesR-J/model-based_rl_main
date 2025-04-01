from ml_collections import config_dict


def get_PILCO_config():
    config = config_dict.ConfigDict()

    config.PLANNING_HORIZON = 25

    config.GAMMA = 1.25  # This is something else not discount value
    config.DISCOUNT_FACTOR = 1.0

    config.LR = 1e-3
    config.POLICY_LR = 0.1
    config.PRETRAIN_RESTARTS = 5
    config.TRAIN_GP_NUM_ITERS = 10#0#0
    config.GP_LR = 0.01
    config.NUM_INDUCING_POINTS = 100

    config.MAX_GRAD_NORM = 0.5

    config.ROLLOUT_SAMPLING = False

    """
    Current iCEM adaptions
    Use constant batch size
    Also use constant sample size for iCEM
    Check saved plan and best plan are the same thing, is there anyway the best plan may not be saved?
    """

    return config