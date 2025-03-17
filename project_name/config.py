from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()
    config.SEED = 42

    # config.ENV_NAME = "pilcocartpole-v0"
    config.ENV_NAME = "bacpendulum-v0"
    config.NORMALISE_ENV = True
    config.GENERATIVE_ENV = True
    config.TELEPORT = True  # aka teleporting in the original thing
    config.PRETRAIN_HYPERPARAMS = False  # True
    config.NUM_INIT_DATA = 1000  # 50

    config.SAVE_FIGURES = True

    config.TEST_SET_SIZE = 100#0
    config.NUM_EVAL_TRIALS = 5
    config.EVAL_FREQ = 10

    config.NUM_ITERS = 51#0

    # config.WANDB = "disabled"
    config.WANDB = "online"

    config.DISABLE_JIT = False
    # config.DISABLE_JIT = True

    config.WANDB_ENTITY = "jamesr-j"  # change this to your wandb username

    config.ROLLOUT_SAMPLING = True  # TODO understand this, I think it only matters for MPC based things?

    # config.AGENT_TYPE = "MPC"
    # config.AGENT_TYPE = "TIP"
    config.AGENT_TYPE = "PETS"

    config.AGENT_CONFIG = {}

    return config


"""
Suffixes
B - Batch size, probably when using replay buffer
E - Number of Episodes
L - Episode Length/NUM_INNER_STEPS/Actions Per Plan
S - Seq length if using trajectory buffer/Planning Horizon
N - Number of Envs
O - Observation Dim
A - Action Dim
C - Action Choices (mostly for discrete actions basically)
Z - More dimensions when in a list
U - Ensemble num
I - Number of elite tops for iCEM
R - Number of iCEM iterations
P - Plus
M - Minus
"""
