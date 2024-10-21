import gymnasium as gym
import random
import ale_py
import torch
import numpy as np
import time

from typing import List, Tuple
from agent import DeepQLearningAgent

gym.register_envs(ale_py)

Action = np.uint8
Reward = np.uint8


def update_agent(
    replay_memory: List[Tuple[torch.Tensor, Action, Reward, bool, torch.Tensor]],
    agent: DeepQLearningAgent,
):
    """
    Take `agent.batch_size` random samples from `replay_memory` and update qvalue agent estimator.
    """
    if len(replay_memory) >= agent.batch_size:
        batch = random.sample(replay_memory, agent.batch_size)
        phi_ts = torch.stack([b[0] for b in batch])
        actions = torch.tensor([b[1] for b in batch])
        rewards = torch.tensor([b[2] for b in batch])
        dones = torch.tensor([b[3] for b in batch], dtype=torch.float32)
        phi_tps = torch.stack([b[4] for b in batch])

        agent.update(phi_ts, actions, rewards, dones, phi_tps)


def train_deep_q_learning():
    env = gym.make(
        "ALE/Breakout-v5", obs_type="grayscale"
    )  # render_mode="human" to see it play
    agent = DeepQLearningAgent(1e-4, 1.0, env.action_space.n)  # pyright: ignore

    replay_mem: List[Tuple[torch.Tensor, Action, Reward, bool, torch.Tensor]] = []
    last_total_steps = 0
    total_steps = 0
    total_reward = 0.0
    start = time.time()

    for ep in range(1, 20000):
        s, _ = env.reset(seed=42)
        last_frames = [torch.tensor(s) for _ in range(4)]

        for t in range(1, 20000):
            phi_t = agent.preprocess_frames(torch.stack(last_frames))
            action = agent.get_action(phi_t.unsqueeze(0))

            # execute action in env and clip reward
            new_s, r, done, _, _ = env.step(action)
            reward = np.uint8(max(-1, min(1, r)))  # pyright: ignore
            total_reward += reward

            # pop first oldest state and add new one
            last_frames.pop(0)
            last_frames.append(torch.tensor(new_s))

            phi_tp = agent.preprocess_frames(torch.stack(last_frames)).float()

            # update replay memory
            if len(replay_mem) == 200000:
                replay_mem.pop(0)
            replay_mem.append((phi_t, np.uint8(action), reward, done, phi_tp))

            # update agent q values
            update_agent(replay_mem, agent)

            total_steps += 1
            if total_steps % 5000 == 0:
                agent.update_target_model()

            if done:
                break

        agent.decay_epsilon()

        if ep % 10 == 0:
            print(
                f"mean reward over 10 simulations and {total_steps - last_total_steps} steps : {round(total_reward / 10, 2)} - trained on {ep} simulations - epsilon {round(agent.epsilon, 4)} - {total_steps} total steps - {round(time.time() - start, 2)} sec"
            )
            total_reward = 0
            last_total_steps = total_steps
            start = time.time()

        if ep % 1000 == 0:
            torch.save(agent.model.state_dict(), "dql.pt")


train_deep_q_learning()
