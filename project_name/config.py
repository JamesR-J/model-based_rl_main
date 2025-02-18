from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()
    config.SEED = 42

    config.NORMALISE_ENV = True

    config.TOTAL_TIMESTEPS = 2300000
    config.NUM_DEVICES = 1

    # config.WANDB = "disabled"
    config.WANDB = "online"

    config.DISABLE_JIT = False
    # config.DISABLE_JIT = True

    config.WANDB_ENTITY = "jamesr-j"  # change this to your wandb username

    config.AGENT_TYPE = "ERSAC"

    config.AGENT_CONFIG = {}

    return config

# TODO need to clarify, for discrete there is only 1 d but A number of actions, for continuous there are Ad actions with a max and min scale


"""
BELNOAZ LoL
B - Batch size, probably when using replay buffer
E - Number of Episodes
L - Episode Length/NUM_INNER_STEPS
S - Seq length if using trajectory buffer
N - Number of Envs
O - Observation Dim
A - Action Dim
C - Action Choices (mostly for discrete actions basically)
Z - More dimensions when in a list
U - Ensemble num
P - Plus
M - Minus

further maybes
M - Number of Meta Episodes
"""
