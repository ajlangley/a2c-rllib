import gymnasium as gym
import numpy as np


class CountingEnv(gym.Env):
    def __init__(self):
        self.count = 0
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=0, high=10, shape=(2,), dtype=np.float32
        )

    def reset(self, **kwargs):
        self.count = 0

        return self.obs(), {}

    def step(self, action):
        reward = self.count
        obs = self.obs()
        if self.count >= 10:
            term = True
        else:
            term = False
        return obs, reward, term, False, {}

    def obs(self):
        obs = np.array([self.count] * 2)
        self.count += 1
        return obs


if __name__ == "__main__":
    env = CountingEnv()
    print(env.reset())
    print(env.step(None))
