import gymnasium as gym

from a2c.a2c import A2C

env = gym.make("LunarLander-v2")
obs, _ = env.reset()
algo = A2C.from_checkpoint("checkpoint")

while True:
    a = algo.compute_single_action(obs, explore=False)
    obs, _, terminatd, truncated, _ = env.step(a)
    env.render()
    exit()
