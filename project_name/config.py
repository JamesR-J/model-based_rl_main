from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()
    config.SEED = 42

    # config.ENV_NAME = "pilcocartpole-v0"
    config.ENV_NAME = "bacpendulum-v0"
    config.NORMALISE_ENV = False
    config.GENERATIVE_ENV = True

    config.PRETRAIN_HYPERPARAMS = False
    config.PRETRAIN_NUM_DATA = 10#00

    config.LEARN_REWARD = False
    # config.LEARN_REWARD = True

    # config.SAVE_FIGURES = False
    config.SAVE_FIGURES = True

    config.TEST_SET_SIZE = 100#0

    config.NUM_EVAL_TRIALS = 5
    config.EVAL_FREQ = 10

    config.NUM_ITERS = 250 + 1

    config.WANDB = "disabled"
    # config.WANDB = "online"

    config.DISABLE_JIT = False
    # config.DISABLE_JIT = True

    config.WANDB_ENTITY = "jamesr-j"  # change this to your wandb username

    config.AGENT_TYPE = "MPC"
    # config.AGENT_TYPE = "PILCO"
    # config.AGENT_TYPE = "TIP"
    # config.AGENT_TYPE = "PETS"

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
