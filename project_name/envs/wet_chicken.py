"""
Based off the following: https://github.com/LAVA-LAB/improved_spi/blob/main/wetChicken.py
"""

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


ACTION_TRANSLATOR = {
    'Drift': np.zeros(2),
    'Neutral': np.array([-1, 0]),
    'Max': np.array([-2, 0]),
    'Left': np.array([0, -1]),
    'Right': np.array([0, 1])
}


@struct.dataclass
class EnvState(environment.EnvState):
    x: jnp.ndarray
    y: jnp.ndarray
    time: int


@struct.dataclass
class EnvParams(environment.EnvParams):
    length: float = 5.0
    width: float = 5.0
    max_turbulence: float = 3.5
    max_velocity: float = 3.
    action_high: float = 1.0
    discrete: bool = False
    hoirzon: int = 200


class WetChicken(environment.Environment[EnvState, EnvParams]):
    # Implements the 2-dimensional discrete Wet Chicken benchmark from 'Efficient Uncertainty Propagation for
    # Reinforcement Learning with Limited Data' by Alexander Hans and Steffen Udluft

    def __init__(self):
        super().__init__()

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters for CartPole-v1
        return EnvParams()

    @staticmethod
    def _velocity(state, params):
        return params.max_velocity * state.x / params.width

    def _turbulence(self, state, params):
        return params.max_turbulence - self._velocity(state, params)

    def step_env(self,
                 key: chex.PRNGKey,
                 state: EnvState,
                 action: Union[int, float, chex.Array],
                 params: EnvParams) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        # if self.discrete:
        #     action_coordinates = list(ACTION_TRANSLATOR.values())[action]
        #     y_hat = self._state[1] + action_coordinates[1] + self._velocity() + self._turbulence() * np.random.uniform(
        #         -1, 1)
        #     x_hat = self._state[0] + action_coordinates[0]
        #     x_hat = round(x_hat)
        #     y_hat = round(y_hat)
        # else:
        key, _key = jrandom.split(key)
        y_hat = state.y + (action[1] - 1) + self._velocity(state, params) + self._turbulence(state, params) * jrandom.uniform(_key, minval=-1, maxval=1)
        x_hat = state.x + action[0]

        if y_hat >= params.length or x_hat < 0:
            x_new = 0
        elif x_hat >= params.width:
            x_new = params.width
        else:
            x_new = x_hat

        if y_hat >= params.length:
            y_new = 0
        elif y_hat < 0:
            y_new = 0
        else:
            y_new = y_hat
        # TODO reformulate these conditionals

        info = {"delta_obs": (x_hat, y_hat)}

        done = False  # TODO add is terminal, not sure about this as pretty sure it auto resets

        reward = -(params.length - y_new)  # TODO pretty sure its the new state y_t

        state = EnvState(x=x_new, y=y_new, time=state.time+1)

        return (lax.stop_gradient(self.get_obs(state)),
                lax.stop_gradient(state),
                jnp.array(reward),  # TODO check reward is the correct way around
                done,
                info)

    def generative_step_env(self, key, obs, action, params):
        state = EnvState(x=obs[0], y=obs[1], time=0)
        return self.step_env(key, state, action, params)

    def reward_function(self, x, next_obs, params: EnvParams):
        return -(params.length - next_obs[..., 1])

    def reset_env(self, key: chex.PRNGKey, params: EnvParams) -> Tuple[chex.Array, EnvState]:
        state = EnvState(x=jnp.zeros((1,)),
                        y=jnp.zeros((1,)),
                        time=0)
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params=None, key=None) -> chex.Array:
        return jnp.array([state.x, state.y])

    @property
    def name(self) -> str:
        """Environment name."""
        return "WetChicken-v1"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 2

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        """Action space of the environment."""
        if params.discrete:
            return spaces.Discrete(len(ACTION_TRANSLATOR))
        else:
            return spaces.Box(-params.action_high, params.action_high, shape=(2,))

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        if params.discrete:
            return spaces.Discrete(2)
        else:
            low = jnp.array([0, 0], dtype=jnp.float64)
            high = jnp.array([params.length, params.width], dtype=jnp.float64)  # TODO check the ordering
            return spaces.Box(low, high, (2,), dtype=jnp.float64)


    # def _get_overlap(self, interval_1, interval_2):
    #     return max(0, min(interval_1[1], interval_2[1]) - max(interval_1[0], interval_2[0]))
    #
    # def get_transition_function(self):
    #     if not self.discrete:
    #         raise AssertionError('You chose a continuous MDP, but requested the transition function.')
    #     nb_states = self.width * self.length
    #     nb_actions = len(ACTION_TRANSLATOR)
    #     P = np.zeros((nb_states, nb_actions, nb_states))
    #     for state in range(nb_states):
    #         x = int(state / self.length)
    #         y = state % self.width
    #         velocity = self.max_velocity * y / self.width
    #         turbulence = self.max_turbulence - velocity
    #
    #         for action_nb, action in enumerate(ACTION_TRANSLATOR.keys()):
    #             action_coordinates = ACTION_TRANSLATOR[action]
    #             target_interval = [x + action_coordinates[0] + velocity - turbulence,
    #                                x + action_coordinates[0] + velocity + turbulence]
    #             prob_mass_on_land = 1 / (2 * turbulence) * self._get_overlap([-self.max_turbulence - 2, -0.5],
    #                                                                          target_interval)  # -self.max_turbulence - 2 should be the lowest possible
    #             prob_mass_waterfall = 1 / (2 * turbulence) * self._get_overlap(
    #                 [self.length - 0.5, self.length + self.max_turbulence + self.max_velocity],
    #                 target_interval)  # self.length + self.max_turbulence + self.max_velocity should be the highest possible
    #             y_hat = y + action_coordinates[1]
    #             if y_hat < 0:
    #                 y_new = 0
    #             elif y_hat >= self.width:
    #                 y_new = self.width - 1
    #             else:
    #                 y_new = y_hat
    #             y_new = int(y_new)
    #             P[state, action_nb, 0] += prob_mass_waterfall
    #             P[state, action_nb, y_new] += prob_mass_on_land
    #             for x_hat in range(self.width):
    #                 x_hat_interval = [x_hat - 0.5, x_hat + 0.5]
    #                 prob_mass = 1 / (2 * turbulence) * self._get_overlap(
    #                     x_hat_interval, target_interval)
    #                 P[state, action_nb, x_hat * self.length + y_new] += prob_mass
    #     return P


def test_wetchicken():
    env = WetChicken()
    n_tests = 100
    for _ in range(n_tests):
        obs = env.reset()
        action = env.action_space.sample()
        next_obs, rew, done, info = env.step(action)
        new_obs = env.reset(obs)
        assert np.allclose(new_obs, obs)
    done = False
    env.reset()
    for _ in range(env.horizon):
        action = env.action_space.sample()
        n, r, done, info = env.step(action)
        if done:
            break
    print("passed")


def plot_some_stuff():
    # Create environment
    env = WetChicken(seed=42)

    # Collect data over multiple episodes
    n_episodes = 1000
    steps_per_episode = 200  # Fixed number of steps per episode
    velocity_data = []
    turbulence_data = []
    reward_data = []
    positions = []

    for _ in range(n_episodes):
        obs = env.reset()

        for step in range(steps_per_episode):
            # Store current state data
            positions.append(obs[1])  # y position
            velocity = env.max_velocity * obs[1] / env.width
            velocity_data.append(velocity)

            # Get turbulence for current state
            turbulence = env._turbulence()  # Note: accessing protected method for visualization
            turbulence_data.append(turbulence)

            # Get reward
            reward = env._get_reward()
            reward_data.append(reward)

            # Take random action
            action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)

    # Convert to numpy arrays
    positions = np.array(positions)
    velocity_data = np.array(velocity_data)
    turbulence_data = np.array(turbulence_data)
    reward_data = np.array(reward_data)

    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

    # Plot 1: Velocity vs Position
    ax1.scatter(positions, velocity_data, alpha=0.1, color='blue', label='Observed')
    ax1.set_xlabel('Position (y)')
    ax1.set_ylabel('Velocity')
    ax1.set_title('Velocity vs Position')
    ax1.grid(True)

    # Theoretical velocity line
    y_pos = np.linspace(0, env.width, 100)
    theoretical_velocity = env.max_velocity * y_pos / env.width
    ax1.plot(y_pos, theoretical_velocity, 'r--', label='Theoretical')
    ax1.legend()

    # Plot 2: Turbulence vs Position
    ax2.scatter(positions, turbulence_data, alpha=0.1, color='green')
    ax2.set_xlabel('Position (y)')
    ax2.set_ylabel('Turbulence')
    ax2.set_title('Turbulence vs Position')
    ax2.grid(True)

    # Add theoretical turbulence bounds
    theoretical_velocity = env.max_velocity * y_pos / env.width
    max_turbulence = env.max_turbulence - theoretical_velocity
    ax2.plot(y_pos, max_turbulence, 'r--', label='Max Turbulence')
    ax2.plot(y_pos, -max_turbulence, 'r--', label='Min Turbulence')
    ax2.legend()

    # Plot 3: Reward vs Position
    ax3.scatter(positions, reward_data, alpha=0.1, color='purple')
    ax3.set_xlabel('Position (y)')
    ax3.set_ylabel('Reward')
    ax3.set_title('Reward vs Position')
    ax3.grid(True)

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

    # Print some statistics
    print(f"\nStatistics over {n_episodes} episodes:")
    print(f"Average velocity: {np.mean(velocity_data):.2f}")
    print(f"Average turbulence: {np.mean(turbulence_data):.2f}")
    print(f"Average reward: {np.mean(reward_data):.2f}")
    print(f"Max position reached: {np.max(positions):.2f}")


def plot_2d_stuff():
    # Create environment
    env = WetChicken(seed=42, discrete=False)

    # Parameters for data collection
    n_episodes = 400
    steps_per_episode = 2000

    # Create grid for storing average values
    grid_size = 50  # Resolution of our grid
    x_grid = np.linspace(0, env.width, grid_size)
    y_grid = np.linspace(0, env.length, grid_size)
    velocity_grid = np.zeros((grid_size, grid_size))
    turbulence_grid = np.zeros((grid_size, grid_size))
    reward_grid = np.zeros((grid_size, grid_size))
    visit_count = np.zeros((grid_size, grid_size))

    for episode in range(n_episodes):
        obs = env.reset()
        for step in range(steps_per_episode):
            # Get current x, y position
            x, y = obs

            # Find grid cell
            x_idx = int(np.clip(x * (grid_size - 1) / env.width, 0, grid_size - 1))
            y_idx = int(np.clip(y * (grid_size - 1) / env.length, 0, grid_size - 1))

            # Calculate values
            velocity = env._velocity()  # env.y_hat
            turbulence = env._turbulence()

            # Take random action
            action = env.action_space.sample()
            obs, reward, _, _ = env.step(action)

            # Update grids
            velocity_grid[x_idx, y_idx] += velocity
            turbulence_grid[x_idx, y_idx] += turbulence
            reward_grid[x_idx, y_idx] += reward
            visit_count[x_idx, y_idx] += 1

    # Average the grids where visited
    mask = visit_count > 0
    velocity_grid[mask] /= visit_count[mask]
    turbulence_grid[mask] /= visit_count[mask]
    reward_grid[mask] /= visit_count[mask]

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))

    # Plot 1: Velocity Heatmap
    im1 = ax1.imshow(velocity_grid.T, origin='lower', extent=[0, env.width, 0, env.length],
                     aspect='auto', cmap='viridis')
    ax1.set_title('Velocity across X-Y plane')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    plt.colorbar(im1, ax=ax1, label='Velocity')

    # Plot 2: Turbulence Heatmap
    im2 = ax2.imshow(turbulence_grid.T, origin='lower', extent=[0, env.width, 0, env.length],
                     aspect='auto', cmap='RdBu')
    ax2.set_title('Turbulence across X-Y plane')
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    plt.colorbar(im2, ax=ax2, label='Turbulence')

    # Plot 3: Reward Heatmap
    im3 = ax3.imshow(reward_grid.T, origin='lower', extent=[0, env.width, 0, env.length],
                     aspect='auto', cmap='plasma')
    ax3.set_title('Reward across X-Y plane')
    ax3.set_xlabel('X Position')
    ax3.set_ylabel('Y Position')
    plt.colorbar(im3, ax=ax3, label='Reward')

    # Plot 4: Visit Count Heatmap (log scale for better visualization)
    visit_count_log = np.log1p(visit_count)  # log1p to handle zeros
    im4 = ax4.imshow(visit_count_log.T, origin='lower', extent=[0, env.width, 0, env.length],
                     aspect='auto', cmap='YlOrRd')
    ax4.set_title('Visit Count across X-Y plane (log scale)')
    ax4.set_xlabel('X Position')
    ax4.set_ylabel('Y Position')
    plt.colorbar(im4, ax=ax4, label='Log(Visit Count + 1)')

    plt.tight_layout()
    plt.show()

    # Print some statistics about the coverage
    print(f"\nStatistics over {n_episodes} episodes ({n_episodes * steps_per_episode} total steps):")
    print(f"Percentage of grid cells visited: {100 * np.sum(visit_count > 0) / (grid_size * grid_size):.1f}%")
    print(f"Average visits per cell (where visited): {np.mean(visit_count[visit_count > 0]):.1f}")
    print(f"Maximum visits to a single cell: {np.max(visit_count):.0f}")


if __name__ == "__main__":
    test_wetchicken()
    # plot_some_stuff()
    # plot_2d_stuff()
