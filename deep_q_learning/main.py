import gymnasium as gym
import random
import ale_py

from agent import DeepQLearningAgent

gym.register_envs(ale_py)

env = gym.make("ALE/Breakout-v5") # , render_mode="human"

agent = DeepQLearningAgent(0.5, 0.05, env.action_space.n, 32)

def play_and_train(
    env: gym.Env, agent: DeepQLearningAgent
) -> float:
    total_reward = 0.
    s, _ = env.reset(seed=42)
    while True:
        a = random.randint(0, 3)

        next_s, r, done, _, _ = env.step(a)
        total_reward += r
        if done:
            break
        s = next_s

    return total_reward

play_and_train(env, agent)