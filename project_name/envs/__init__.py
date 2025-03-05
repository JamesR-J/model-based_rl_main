import logging
from gymnasium.envs.registration import register
from project_name.envs.pilco_cartpole import CartPoleSwingUpEnv, pilco_cartpole_reward
from project_name.envs.wet_chicken import WetChicken, wet_chicken_reward


# register(
#     id="bacpendulum-v0",
#     entry_point=PendulumEnv,
# )
register(
    id="pilcocartpole-v0",
    entry_point=CartPoleSwingUpEnv,
)
reward_functions = {
    # "bacpendulum-v0": pendulum_reward,
    "pilcocartpole-v0": pilco_cartpole_reward,
}

register(id="wetchicken-v0", entry_point=WetChicken)
reward_functions["wetchicken-v0"] = wet_chicken_reward

# # mujoco stuff
# try:
#     from barl.envs.reacher import BACReacherEnv, reacher_reward, tf_reacher_reward
#
#     register(
#         id="bacreacher-v0",
#         entry_point=BACReacherEnv,
#     )
#     reward_functions["bacreacher-v0"] = reacher_reward
# except Exception as e:
#     logging.warning("mujoco not found, skipping those envs")
#     logging.warning(e)
# try:
#     from barl.envs.beta_tracking_env import BetaTrackingGymEnv, beta_tracking_rew
#     from barl.envs.tracking_env import TrackingGymEnv, tracking_rew
#     from barl.envs.betan_env import BetanRotationEnv, betan_rotation_reward
#
#     register(
#         id="betatracking-v0",
#         entry_point=BetaTrackingGymEnv,
#     )
#     reward_functions["betatracking-v0"] = beta_tracking_rew
# except:
#     logging.info("old fusion dependencies not found, skipping")
# try:
#     from fusion_control.envs.gym_env import BetaTrackingGymEnv as NewBetaTrackingGymEnv
#     from fusion_control.envs.gym_env import (
#         BetaRotationTrackingGymEnv,
#         BetaRotationTrackingMultiGymEnv,
#     )
#     from fusion_control.envs.rewards import TrackingReward
#
#     register(
#         id="newbetatracking-v0",
#         entry_point=NewBetaTrackingGymEnv,
#     )
#     _beta_env = NewBetaTrackingGymEnv()
#     reward_functions["newbetatracking-v0"] = _beta_env.get_reward
#     register(
#         id="newbetarotation-v0",
#         entry_point=BetaRotationTrackingGymEnv,
#     )
#     _beta_rotation_env = BetaRotationTrackingGymEnv()
#     reward_functions["newbetarotation-v0"] = _beta_rotation_env.get_reward
#     register(
#         id="multibetarotation-v0",
#         entry_point=BetaRotationTrackingMultiGymEnv,
#     )
#     _multi_beta_rotation_env = BetaRotationTrackingMultiGymEnv()
#     reward_functions["multibetarotation-v0"] = _multi_beta_rotation_env.get_reward
# except:
#     logging.warning("new fusion dependencies not found, skipping")
