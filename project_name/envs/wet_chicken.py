"""
Based off the following: https://github.com/LAVA-LAB/improved_spi/blob/main/wetChicken.py
"""

# import gymnasium
# from gymnasium import spaces
# from gymnasium.utils import seeding
# import numpy as np
# from os import path
# import matplotlib.pyplot as plt
#
#
# ACTION_TRANSLATOR = {
#     'Drift': np.zeros(2),
#     'Neutral': np.array([-1, 0]),
#     'Max': np.array([-2, 0]),
#     'Left': np.array([0, -1]),
#     'Right': np.array([0, 1])
# }
#
#
# class WetChicken(gymnasium.Env):
#     # Implements the 2-dimensional discrete Wet Chicken benchmark from 'Efficient Uncertainty Propagation for
#     # Reinforcement Learning with Limited Data' by Alexander Hans and Steffen Udluft
#
#     def __init__(self, seed=42, length=5, width=5, max_turbulence=3.5, max_velocity=3.0, discrete=False):
#         self.seed(seed)
#         self.length = length
#         self.width = width
#         self.max_turbulence = max_turbulence
#         self.max_velocity = max_velocity
#         self._state = np.zeros(2)  # Don't use this state outside of this class, it is an array. Instead use get_state_int!
#         self.discrete = discrete
#
#         if discrete:
#             self.observation_space = spaces.Discrete(2)
#             self.action_space = spaces.Discrete(len(ACTION_TRANSLATOR))
#         else:
#             low = np.array([0, 0], dtype=np.float32)
#             high = np.array([length, width], dtype=np.float32)  # TODO check the ordering
#             action_high = 1
#             self.observation_space = spaces.Box(low, high)
#             self.action_space = spaces.Box(-action_high, action_high, shape=(2,))
#
#         self.horizon = 20#0  # TODO tune what the horizon is
#
#         self.x_hat = 0  # for plotting purposes
#         self.y_hat = 0
#
#     def seed(self, seed=None):
#         self.np_random, seed = seeding.np_random(seed)
#         return [seed]
#
#     def _velocity(self):
#         return self.max_velocity * self._state[0] / self.width
#
#     def _turbulence(self):
#         return self.max_turbulence - self._velocity()
#
#     def reset(self, obs=None):
#         if obs is None:
#             self._state = np.zeros(2)
#         else:
#             self._state = obs
#         return self._state
#
#     def _get_obs(self):
#         return self._state
#
#     def step(self, action):
#         if self.discrete:
#             action_coordinates = list(ACTION_TRANSLATOR.values())[action]
#             y_hat = self._state[1] + action_coordinates[1] + self._velocity() + self._turbulence() * np.random.uniform(-1, 1)
#             x_hat = self._state[0] + action_coordinates[0]
#             x_hat = round(x_hat)
#             y_hat = round(y_hat)
#         else:
#             y_hat = self._state[1] + (action[1] - 1) + self._velocity() + self._turbulence() * np.random.uniform(-1, 1)
#             x_hat = self._state[0] + action[0]
#
#         if y_hat >= self.length or x_hat < 0:
#             self._state[0] = 0
#         elif x_hat >= self.width:
#             self._state[0] = self.width
#         else:
#             self._state[0] = x_hat
#
#         if y_hat >= self.length:
#             self._state[1] = 0
#         elif y_hat < 0:
#             self._state[1] = 0
#         else:
#             self._state[1] = y_hat
#
#         self.x_hat = x_hat
#         self.y_hat = y_hat
#
#         info = {"delta_obs": (x_hat, y_hat)}
#
#         done = False  # TODO add is terminal, not sure about this as pretty sure it auto resets
#
#         reward = -(self.length - self._state[1])  # TODO pretty sure its the new state y_t
#
#         return self._get_obs(), reward, done, info
#
#     def _get_overlap(self, interval_1, interval_2):
#         return max(0, min(interval_1[1], interval_2[1]) - max(interval_1[0], interval_2[0]))
#
#     def get_transition_function(self):
#         if not self.discrete:
#             raise AssertionError('You chose a continuous MDP, but requested the transition function.')
#         nb_states = self.width * self.length
#         nb_actions = len(ACTION_TRANSLATOR)
#         P = np.zeros((nb_states, nb_actions, nb_states))
#         for state in range(nb_states):
#             x = int(state / self.length)
#             y = state % self.width
#             velocity = self.max_velocity * y / self.width
#             turbulence = self.max_turbulence - velocity
#
#             for action_nb, action in enumerate(ACTION_TRANSLATOR.keys()):
#                 action_coordinates = ACTION_TRANSLATOR[action]
#                 target_interval = [x + action_coordinates[0] + velocity - turbulence,
#                                    x + action_coordinates[0] + velocity + turbulence]
#                 prob_mass_on_land = 1 / (2 * turbulence) * self._get_overlap([-self.max_turbulence - 2, -0.5],
#                                                                              target_interval)  # -self.max_turbulence - 2 should be the lowest possible
#                 prob_mass_waterfall = 1 / (2 * turbulence) * self._get_overlap(
#                     [self.length - 0.5, self.length + self.max_turbulence + self.max_velocity],
#                     target_interval)  # self.length + self.max_turbulence + self.max_velocity should be the highest possible
#                 y_hat = y + action_coordinates[1]
#                 if y_hat < 0:
#                     y_new = 0
#                 elif y_hat >= self.width:
#                     y_new = self.width - 1
#                 else:
#                     y_new = y_hat
#                 y_new = int(y_new)
#                 P[state, action_nb, 0] += prob_mass_waterfall
#                 P[state, action_nb, y_new] += prob_mass_on_land
#                 for x_hat in range(self.width):
#                     x_hat_interval = [x_hat - 0.5, x_hat + 0.5]
#                     prob_mass = 1 / (2 * turbulence) * self._get_overlap(
#                         x_hat_interval, target_interval)
#                     P[state, action_nb, x_hat * self.length + y_new] += prob_mass
#         return P
#
#
# def wet_chicken_reward(obs, nobs):
#     return -(WetChicken().length - nobs[1])
#
#
# def test_wetchicken():
#     env = WetChicken()
#     n_tests = 100
#     for _ in range(n_tests):
#         obs = env.reset()
#         action = env.action_space.sample()
#         next_obs, rew, done, info = env.step(action)
#         new_obs = env.reset(obs)
#         assert np.allclose(new_obs, obs)
#     done = False
#     env.reset()
#     for _ in range(env.horizon):
#         action = env.action_space.sample()
#         n, r, done, info = env.step(action)
#         if done:
#             break
#     print("passed")
#
#
# def plot_some_stuff():
#     # Create environment
#     env = WetChicken(seed=42)
#
#     # Collect data over multiple episodes
#     n_episodes = 1000
#     steps_per_episode = 200  # Fixed number of steps per episode
#     velocity_data = []
#     turbulence_data = []
#     reward_data = []
#     positions = []
#
#     for _ in range(n_episodes):
#         obs = env.reset()
#
#         for step in range(steps_per_episode):
#             # Store current state data
#             positions.append(obs[1])  # y position
#             velocity = env.max_velocity * obs[1] / env.width
#             velocity_data.append(velocity)
#
#             # Get turbulence for current state
#             turbulence = env._turbulence()  # Note: accessing protected method for visualization
#             turbulence_data.append(turbulence)
#
#             # Get reward
#             reward = env._get_reward()
#             reward_data.append(reward)
#
#             # Take random action
#             action = env.action_space.sample()
#             obs, reward, done, _ = env.step(action)
#
#     # Convert to numpy arrays
#     positions = np.array(positions)
#     velocity_data = np.array(velocity_data)
#     turbulence_data = np.array(turbulence_data)
#     reward_data = np.array(reward_data)
#
#     # Create figure with subplots
#     fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
#
#     # Plot 1: Velocity vs Position
#     ax1.scatter(positions, velocity_data, alpha=0.1, color='blue', label='Observed')
#     ax1.set_xlabel('Position (y)')
#     ax1.set_ylabel('Velocity')
#     ax1.set_title('Velocity vs Position')
#     ax1.grid(True)
#
#     # Theoretical velocity line
#     y_pos = np.linspace(0, env.width, 100)
#     theoretical_velocity = env.max_velocity * y_pos / env.width
#     ax1.plot(y_pos, theoretical_velocity, 'r--', label='Theoretical')
#     ax1.legend()
#
#     # Plot 2: Turbulence vs Position
#     ax2.scatter(positions, turbulence_data, alpha=0.1, color='green')
#     ax2.set_xlabel('Position (y)')
#     ax2.set_ylabel('Turbulence')
#     ax2.set_title('Turbulence vs Position')
#     ax2.grid(True)
#
#     # Add theoretical turbulence bounds
#     theoretical_velocity = env.max_velocity * y_pos / env.width
#     max_turbulence = env.max_turbulence - theoretical_velocity
#     ax2.plot(y_pos, max_turbulence, 'r--', label='Max Turbulence')
#     ax2.plot(y_pos, -max_turbulence, 'r--', label='Min Turbulence')
#     ax2.legend()
#
#     # Plot 3: Reward vs Position
#     ax3.scatter(positions, reward_data, alpha=0.1, color='purple')
#     ax3.set_xlabel('Position (y)')
#     ax3.set_ylabel('Reward')
#     ax3.set_title('Reward vs Position')
#     ax3.grid(True)
#
#     # Adjust layout and display
#     plt.tight_layout()
#     plt.show()
#
#     # Print some statistics
#     print(f"\nStatistics over {n_episodes} episodes:")
#     print(f"Average velocity: {np.mean(velocity_data):.2f}")
#     print(f"Average turbulence: {np.mean(turbulence_data):.2f}")
#     print(f"Average reward: {np.mean(reward_data):.2f}")
#     print(f"Max position reached: {np.max(positions):.2f}")
#
#
# def plot_2d_stuff():
#     # Create environment
#     env = WetChicken(seed=42, discrete=False)
#
#     # Parameters for data collection
#     n_episodes = 400
#     steps_per_episode = 2000
#
#     # Create grid for storing average values
#     grid_size = 50  # Resolution of our grid
#     x_grid = np.linspace(0, env.width, grid_size)
#     y_grid = np.linspace(0, env.length, grid_size)
#     velocity_grid = np.zeros((grid_size, grid_size))
#     turbulence_grid = np.zeros((grid_size, grid_size))
#     reward_grid = np.zeros((grid_size, grid_size))
#     visit_count = np.zeros((grid_size, grid_size))
#
#     for episode in range(n_episodes):
#         obs = env.reset()
#         for step in range(steps_per_episode):
#             # Get current x, y position
#             x, y = obs
#
#             # Find grid cell
#             x_idx = int(np.clip(x * (grid_size - 1) / env.width, 0, grid_size - 1))
#             y_idx = int(np.clip(y * (grid_size - 1) / env.length, 0, grid_size - 1))
#
#             # Calculate values
#             velocity = env._velocity()  # env.y_hat
#             turbulence = env._turbulence()
#
#             # Take random action
#             action = env.action_space.sample()
#             obs, reward, _, _ = env.step(action)
#
#             # Update grids
#             velocity_grid[x_idx, y_idx] += velocity
#             turbulence_grid[x_idx, y_idx] += turbulence
#             reward_grid[x_idx, y_idx] += reward
#             visit_count[x_idx, y_idx] += 1
#
#     # Average the grids where visited
#     mask = visit_count > 0
#     velocity_grid[mask] /= visit_count[mask]
#     turbulence_grid[mask] /= visit_count[mask]
#     reward_grid[mask] /= visit_count[mask]
#
#     # Create figure with subplots
#     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))
#
#     # Plot 1: Velocity Heatmap
#     im1 = ax1.imshow(velocity_grid.T, origin='lower', extent=[0, env.width, 0, env.length],
#                      aspect='auto', cmap='viridis')
#     ax1.set_title('Velocity across X-Y plane')
#     ax1.set_xlabel('X Position')
#     ax1.set_ylabel('Y Position')
#     plt.colorbar(im1, ax=ax1, label='Velocity')
#
#     # Plot 2: Turbulence Heatmap
#     im2 = ax2.imshow(turbulence_grid.T, origin='lower', extent=[0, env.width, 0, env.length],
#                      aspect='auto', cmap='RdBu')
#     ax2.set_title('Turbulence across X-Y plane')
#     ax2.set_xlabel('X Position')
#     ax2.set_ylabel('Y Position')
#     plt.colorbar(im2, ax=ax2, label='Turbulence')
#
#     # Plot 3: Reward Heatmap
#     im3 = ax3.imshow(reward_grid.T, origin='lower', extent=[0, env.width, 0, env.length],
#                      aspect='auto', cmap='plasma')
#     ax3.set_title('Reward across X-Y plane')
#     ax3.set_xlabel('X Position')
#     ax3.set_ylabel('Y Position')
#     plt.colorbar(im3, ax=ax3, label='Reward')
#
#     # Plot 4: Visit Count Heatmap (log scale for better visualization)
#     visit_count_log = np.log1p(visit_count)  # log1p to handle zeros
#     im4 = ax4.imshow(visit_count_log.T, origin='lower', extent=[0, env.width, 0, env.length],
#                      aspect='auto', cmap='YlOrRd')
#     ax4.set_title('Visit Count across X-Y plane (log scale)')
#     ax4.set_xlabel('X Position')
#     ax4.set_ylabel('Y Position')
#     plt.colorbar(im4, ax=ax4, label='Log(Visit Count + 1)')
#
#     plt.tight_layout()
#     plt.show()
#
#     # Print some statistics about the coverage
#     print(f"\nStatistics over {n_episodes} episodes ({n_episodes * steps_per_episode} total steps):")
#     print(f"Percentage of grid cells visited: {100 * np.sum(visit_count > 0) / (grid_size * grid_size):.1f}%")
#     print(f"Average visits per cell (where visited): {np.mean(visit_count[visit_count > 0]):.1f}")
#     print(f"Maximum visits to a single cell: {np.max(visit_count):.0f}")
#
#
# if __name__ == "__main__":
#     test_wetchicken()
#     # plot_some_stuff()
#     # plot_2d_stuff()
