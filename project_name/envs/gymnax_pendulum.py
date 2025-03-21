import numpy as np
from os import path
import jax.numpy as jnp
import jax.random as jrandom
from gymnax.environments import environment
from gymnax.environments import spaces
from flax import struct
from typing import Any, Dict, Optional, Tuple, Union
import chex
from jax import lax


@struct.dataclass
class EnvState(environment.EnvState):
    theta: jnp.ndarray
    theta_dot: jnp.ndarray
    time: int


@struct.dataclass
class EnvParams(environment.EnvParams):
    max_speed: float = 8.0
    max_torque: float = 2.0
    dt: float = 0.05
    gravity: float = 10.0
    mass: float = 1.0
    length: float = 1.0
    horizon: int = 200

class GymnaxPendulum(environment.Environment[EnvState, EnvParams]):

    def __init__(self):
        super().__init__()
        self.obs_dim = 2

        self.periodic_dim = jnp.array((1, 0))  # TODO is this the best way?

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters for CartPole-v1
        return EnvParams()

    def step_env(self,
                 key: chex.PRNGKey,
                 state: EnvState,
                 action: Union[int, float, chex.Array],
                 params: EnvParams) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        u = jnp.clip(action, -params.max_torque, params.max_torque)[0]

        newthdot = (state.theta_dot + (-3 * params.gravity / (2 * params.length) * jnp.sin(state.theta + jnp.pi) +
                                       3.0 / (params.mass * params.length ** 2) * u) * params.dt)
        unnorm_newth = state.theta + newthdot * params.dt
        newth = self._angle_normalise(unnorm_newth)
        newthdot = jnp.clip(newthdot, -params.max_speed, params.max_speed)

        costs = self._angle_normalise(newth) ** 2 + 0.1 * newthdot ** 2 + 0.001 * (u ** 2)

        delta_s = jnp.array((unnorm_newth, newthdot)) - self.get_obs(state)

        state = EnvState(theta=newth, theta_dot=newthdot, time=state.time+1)

        done = False

        return (lax.stop_gradient(self.get_obs(state)),
                lax.stop_gradient(state),
                jnp.array(-costs),
                done,
                {"delta_obs": delta_s})

    def generative_step_env(self, key, obs, action, params):
        state = EnvState(theta=obs[0], theta_dot=obs[1], time=0)
        return self.step_env(key, state, action, params)

    def _angle_normalise(self, x):
        return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi

    def reward_function(self, x, next_obs, params: EnvParams):
        th = next_obs[..., 0]
        thdot = next_obs[..., 1]
        u = x[..., 2]
        costs = self._angle_normalise(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)
        return -costs

    def reset_env(self, key: chex.PRNGKey, params: EnvParams) -> Tuple[chex.Array, EnvState]:
        high = jnp.array([jnp.pi, 1])
        init_state = jrandom.uniform(key, shape=(2,), minval=-high, maxval=high)
        state = EnvState(theta=init_state[0],
                         theta_dot=init_state[1],
                         time=0)

        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params=None, key=None) -> chex.Array:
        return jnp.array([state.theta, state.theta_dot])

    @property
    def name(self) -> str:
        """Environment name."""
        return "Pendulum-v1"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 1

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        """Action space of the environment."""
        return spaces.Box(-params.max_torque, params.max_torque, shape=(1,))

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        high = jnp.array([jnp.pi, params.max_speed])
        return spaces.Box(-high, high, (2,), dtype=jnp.float32)

    # TODO add in state space
