import gymnasium as gym
import numpy as np
import random as rand

env = gym.make("CartPole-v1", render_mode = "human")
observartion, info = env.reset()

# 0 - left
# 1 - right
action_space = [0, 1]



for _ in range(1000):
    # action = 
    observation, reward, term, trunc, info = env.step(0)
    env.render()

    if (term):
        env.reset()
        quit()

