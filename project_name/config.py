from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()
    config.SEED = 42

    # config.ENV_NAME = "pilcocartpole-v0"
    config.ENV_NAME = "bacpendulum-v0"
    config.NORMALISE_ENV = False  # TODO sort this out its so bad
    config.GENERATIVE_ENV = True
    config.TELEPORT = False  # aka teleporting in the original thing

    config.SAVE_FIGURES = True

    config.NUM_INIT_DATA = 1  # 50
    config.TEST_SET_SIZE = 100#0
    config.NUM_EVAL_TRIALS = 5
    config.EVAL_FREQ = 10

    config.NUM_ITERS = 51#0

    config.TOTAL_TIMESTEPS = 2300000
    config.NUM_DEVICES = 1

    # config.WANDB = "disabled"
    config.WANDB = "online"

    config.DISABLE_JIT = False
    # config.DISABLE_JIT = True

    config.WANDB_ENTITY = "jamesr-j"  # change this to your wandb username

    config.AGENT_TYPE = "MPC"

    config.AGENT_CONFIG = {}

    return config

# TODO need to clarify, for discrete there is only 1 d but A number of actions, for continuous there are Ad actions with a max and min scale


"""
BELNOAZ LoL
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

further maybes
M - Number of Meta Episodes
"""
