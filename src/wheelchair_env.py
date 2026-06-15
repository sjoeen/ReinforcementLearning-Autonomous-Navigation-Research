from robot_state import RobotState
from typing import Tuple
import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import zmq


class WheelchairEnv(gym.Env):
    def __init__(self, env_id: int, noise_level: float = 0.05):
        super(WheelchairEnv, self).__init__()

        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.bind("ipc:///tmp/giorgio_" + str(env_id))

        self.env_id = env_id
        self.noise_level = noise_level

        """ 
        Action and state space definition.
        Robot will be able to control speed of left and right wheels between 0 and 5.
        State is a vector of 360 lidar readings and the last action taken (as integer).
        """
        self.action_space = gym.spaces.Discrete(6)

        v, w = 1, 3
        self.to_action = lambda x: (
            [
                [v, 0],  # Forward
                [v, w],  # Forward and left
                [v, -w],  # Forward and right
                [0, w],  # Left
                [0, -w],  # Right
                [0, 0],  # Stop
            ]
        )[x]

        self.observation_space = Box(
            # 0 to 5 because there are 6 discrete actions
            low=np.concatenate([np.full(360, 0.0), [0]]),
            high=np.concatenate([np.full(360, 10.0), [5]]),
            dtype=np.float64,
        )
        self.obs_shape = self.observation_space.shape

        self.no_obs()
        self.prev_action = 0
        self.prev_pref = 0.0  # Previous preference for side commitment
        self.time_step = 0
        self.time_limit = 20_000
        self.commitment_threshold = 2.5  # Distance below which we force side commitment

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self.prev_action = action
        action = self.to_action(action)

        obs = self.send_action_get_obs(action)

        reward = self.get_reward(obs.lidar, action)

        if obs.collided:
            reward += self.collision_reward()
        elif obs.goal_reached:
            reward += self.goal_reward()

        self.prev_lidar = obs.lidar
        self.time_step += 1

        done = obs.collided or obs.goal_reached
        info = {"is_success": obs.goal_reached}

        return obs.to_array(), reward, done, False, info

    def get_reward(self, obs: np.ndarray, action: Tuple[int, int]) -> float:
        v, w = action

        # Forward movement reward
        r_distance = 1.0 if v > 0 else 0.0

        # Collision penalty (exponential when very close)
        min_range = np.min(obs[140:220])  # Front sector
        collision_threshold = 1.0
        if min_range < collision_threshold:
            r_collision = -np.exp(3 * (collision_threshold - min_range)) + 1
        else:
            r_collision = 0.0

        r_navigation = self.navigation_reward(obs, action)
        r_stability = self.stability_reward(obs, action)

        total_reward = r_distance + r_collision + r_navigation + r_stability
        return total_reward

    def navigation_reward(self, obs: np.ndarray, action: Tuple[int, int]) -> float:
        _, w = action

        left_sector = obs[100:170]
        front_sector = obs[170:190]
        right_sector = obs[190:260]

        left_clearance = np.mean(left_sector)
        right_clearance = np.mean(right_sector)

        obstacle_ahead = np.any(front_sector < self.commitment_threshold)

        if obstacle_ahead:
            clearance_diff = right_clearance - left_clearance

            alpha = 0.40
            self.prev_pref = (1 - alpha) * self.prev_pref + alpha * clearance_diff

            r = 3.0

            if self.prev_pref > 0.2:  # Prefer right
                return r if w < 0 else -r
            if self.prev_pref < -0.2:  # Prefer left
                return r if w > 0 else -r

            if clearance_diff > 0.2:
                return r if w < 0 else -r * 0.5
            if clearance_diff < -0.2:
                return r if w > 0 else -r * 0.5
            if np.min(front_sector) < 1.0:
                return r if w != 0 else -r
        elif w == 0:
            return 1.0

        return 0

    def stability_reward(self, obs: np.ndarray, action: Tuple[int, int]) -> float:
        _, w = action
        if w != 0:
            return -0.2
        return 0

    def reset_preference(self):
        self.prev_pref = 0.0

    def collision_reward(self) -> int:
        return -10

    def goal_reward(self) -> int:
        return 0

    def send_action_get_obs(self, action: Tuple[int, int]) -> RobotState:
        """Send action to server and get observation"""
        self.socket.send_pyobj(action)
        return self.get_observation()

    def get_observation(self) -> RobotState:
        """Get observation from server"""
        state = self.socket.recv_pyobj()
        state.prev_action = self.prev_action

        # Interpolate lidar back to 360 rays regardless of hardware resolution
        if len(state.lidar) != 360:
            state.lidar = np.interp(
                np.linspace(0, 1, 360),
                np.linspace(0, 1, len(state.lidar)),
                state.lidar
            )

        # Inject Gaussian noise if noise_level > 0
        if self.noise_level > 0:
            noise = np.random.normal(0, self.noise_level, size=state.lidar.shape)
            state.lidar = np.clip(state.lidar + noise, 0.0, 10.0)

        return state

    def no_obs(self) -> np.ndarray:
        return np.zeros(self.obs_shape, dtype=np.float64)

    def reset(self, seed: int = None, options: dict = None) -> Tuple[np.ndarray, dict]:
        obs = self.no_obs()
        self.prev_lidar = obs[:360]
        self.time_step = 0
        self.reset_preference()
        return obs, {}

    def close(self):
        print("Closing environment " + str(self.env_id))
        self.socket.send_pyobj([-1])
        self.socket.close()
        zmq.Context.instance().destroy()
        return super().close()