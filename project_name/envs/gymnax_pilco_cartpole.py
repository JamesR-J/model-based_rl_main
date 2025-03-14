"""
Based off: https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/classic_control/cartpole.py

Converted to be like Pilco cartpole

"""

from typing import Any, Dict, Optional, Tuple, Union


import chex
from flax import struct
import jax
from jax import lax
import jax.numpy as jnp
from gymnax.environments import environment
from gymnax.environments import spaces


@struct.dataclass
class EnvState(environment.EnvState):
    x: jnp.ndarray
    x_dot: jnp.ndarray
    theta: jnp.ndarray
    theta_dot: jnp.ndarray
    time: int


@struct.dataclass
class EnvParams(environment.EnvParams):
    gravity: float = 9.82
    masscart: float = 0.5  # 1.0
    masspole: float = 0.5  # 0.1
    total_mass: float = 0.5 + 0.5  # (masscart + masspole)  # TODO can we make this automated rather than hard written calcs
    length: float = 0.6  # 0.5
    mass_pole_length: float = 0.3  # (masspole * length)
    force_mag: float = 10.0
    dt: float = 0.1  # seconds between state updates
    b: float = 0.1  # friction coefficient
    theta_threshold_radians: float = 12 * 2 * jnp.pi / 360
    x_threshold: float = 2.4
    horizon: int = 25


class GymnaxPilcoCartPole(environment.Environment[EnvState, EnvParams]):
    def __init__(self):
        super().__init__()
        self.obs_dim = 4

        self.periodic_dim = jnp.array((0, 0, 1, 0))  # TODO is this the best way?

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters for CartPole-v1
        return EnvParams()

    def step_env(self,
                 key: chex.PRNGKey,
                 state: EnvState,
                 action: Union[int, float, chex.Array],
                 params: EnvParams) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
                """Performs step transitions in the environment."""
                action = jnp.clip(action, -1, 1)[0] * params.force_mag

                costheta = jnp.cos(state.theta)
                sintheta = jnp.sin(state.theta)

                xdot_update = (-2 * params.mass_pole_length * (state.theta_dot ** 2) * sintheta + 3 * params.masspole * params.gravity
                                * sintheta * costheta + 4 * action - 4 * params.b * state.x_dot) / (4 * params.total_mass - 3
                                                                                                    * params.masspole * costheta ** 2)

                thetadot_update = (-3 * params.mass_pole_length * (state.theta_dot ** 2) * sintheta * costheta + 6 * params.total_mass
                                   * params.gravity * sintheta + 6 * (action - params.b * state.x_dot) * costheta) / (4 * params.length * params.total_mass - 3 * params.mass_pole_length * costheta ** 2)

                x = state.x + state.x_dot * params.dt
                unnorm_theta = state.theta + state.theta_dot * params.dt
                theta = self._angle_normalise(unnorm_theta)
                x_dot = state.x_dot + xdot_update * params.dt
                theta_dot = state.theta_dot + thetadot_update * params.dt

                # compute costs - saturation cost
                goal = jnp.array([0.0, params.length])
                pole_x = params.length * jnp.sin(theta)
                pole_y = params.length * jnp.cos(theta)
                position = jnp.array([state.x + pole_x, pole_y])
                squared_distance = jnp.sum((position - goal) ** 2)
                squared_sigma = 0.25 ** 2
                costs = 1 - jnp.exp(-0.5 * squared_distance / squared_sigma)

                delta_s = jnp.array((x, x_dot, unnorm_theta, theta_dot)) - self.get_obs(state)

                # Update state dict and evaluate termination conditions
                state = EnvState(x=x,
                                 x_dot=x_dot,
                                 theta=theta,
                                 theta_dot=theta_dot,
                                 time=state.time + 1)

                # done = self.is_terminal(state, params)
                done = False  # TODO apparently always false

                return (lax.stop_gradient(self.get_obs(state)),
                        lax.stop_gradient(state),
                        jnp.array(-costs),
                        done,
                        {"discount": self.discount(state, params),
                         "delta_obs": delta_s})

    def generative_step_env(self, key, obs, action, params):
        state = EnvState(x=obs[0], x_dot=obs[1], theta=obs[2], theta_dot=obs[3], time=0)
        return self.step_env(key, state, action, params)

    def _angle_normalise(self, x):
        return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi

    def _get_pole_pos(self, x, params: EnvParams):
        xpos = x[..., 0]
        theta = x[..., 2]
        pole_x = params.length * jnp.sin(theta)
        pole_y = params.length * jnp.cos(theta)
        position = jnp.array([xpos + pole_x, pole_y]).T
        return position

    def reward_function(self, x, next_obs, params: EnvParams):
        position = self._get_pole_pos(next_obs, params)
        goal = jnp.array([0.0, params.length])
        squared_distance = jnp.sum((position - goal) ** 2, axis=-1)
        squared_sigma = 0.25 ** 2
        costs = 1 - jnp.exp(-0.5 * squared_distance / squared_sigma)
        return -costs

    def reset_env(self, key: chex.PRNGKey, params: EnvParams) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        loc = jnp.array([0.0, 0.0, jnp.pi, 0.0])
        scale = jnp.array([0.02, 0.02, 0.02, 0.02])
        init_state = jax.random.normal(key, shape=(4,)) * scale + loc
        state = EnvState(x=init_state[0],
                         x_dot=init_state[1],
                         theta=self._angle_normalise(init_state[2]),
                         theta_dot=init_state[3],
                         time=0)
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params=None, key=None) -> chex.Array:
        """Applies observation function to state."""
        # TODO if self.use_trig then it is the below
        # return jnp.array([state.x, state.x_dot, jnp.sin(state.theta), jnp.cos(state.theta), state.theta_dot])
        # Otherwise
        return jnp.array([state.x, state.x_dot, state.theta, state.theta_dot])


    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Check whether state is terminal."""
        # Check termination criteria
        done1 = jnp.logical_or(state.x < -params.x_threshold, state.x > params.x_threshold)
        done2 = jnp.logical_or(state.theta < -params.theta_threshold_radians, state.theta > params.theta_threshold_radians)

        # Check number of steps in episode termination condition
        done_steps = state.time >= params.max_steps_in_episode
        done = jnp.logical_or(jnp.logical_or(done1, done2), done_steps)
        return done

    @property
    def name(self) -> str:
        """Environment name."""
        return "CartPole-v1"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 2

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        """Action space of the environment."""
        return spaces.Box(-1, 1, shape=(1,))

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        high = jnp.array([10.0, 10.0, 3.14159, 25.0])
        return spaces.Box(-high, high, (4,), dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        high = jnp.array(
            [
                params.x_threshold * 2,
                jnp.finfo(jnp.float32).max,
                params.theta_threshold_radians * 2,
                jnp.finfo(jnp.float32).max,
            ]
        )
        return spaces.Dict(
            {
                "x": spaces.Box(-high[0], high[0], (), jnp.float32),
                "x_dot": spaces.Box(-high[1], high[1], (), jnp.float32),
                "theta": spaces.Box(-high[2], high[2], (), jnp.float32),
                "theta_dot": spaces.Box(-high[3], high[3], (), jnp.float32),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )