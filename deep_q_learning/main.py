import gymnasium as gym
import random
import ale_py
import torch

from typing import List, Tuple
from gymnasium.core import Env
from agent import DeepQLearningAgent

gym.register_envs(ale_py)

Action = int
Reward = float


def update_agent(
    replay_memory: List[Tuple[torch.Tensor, Action, Reward, bool, torch.Tensor]],
    env: Env,
    agent: DeepQLearningAgent,
):
    batch_size = min(len(replay_memory), agent.batch_size)
    batch = random.sample(replay_memory, batch_size)

    phi_ts = torch.stack([b[0] for b in batch])
    actions = torch.tensor([b[1] for b in batch])
    rewards = torch.tensor([b[2] for b in batch])
    is_not_done = torch.tensor([not b[3] for b in batch], dtype=torch.float32)
    phi_tps = torch.stack([b[4] for b in batch])

    with torch.no_grad():
        y = is_not_done * 0.99 * agent.get_value(phi_tps)
        y += rewards

    agent.update(
        phi_ts,
        y,
        actions,
    )


def train_deep_q_learning(
    learning_rate: float = 0.5,
    M: int = 10000,
    T: int = 50000,
):
    env = gym.make("ALE/Breakout-v5", obs_type="grayscale")  # , render_mode="human"
    agent = DeepQLearningAgent(0.01, 0.1, env.action_space.n)  # pyright: ignore

    replay_memory: List[
        Tuple[torch.Tensor, Action, Reward, bool, torch.Tensor]
    ] = []  # max size is 100 000

    for ep in range(1, M):
        total_reward = 0.0
        s, _ = env.reset(seed=42)
        last_frames = [torch.tensor(s)]
        action = 0
        # warmup
        for _ in range(4 - 1):  # skip first 3 frames
            new_s, r, done, _, _ = env.step(action)
            last_frames.append(torch.tensor(new_s))

        for t in range(1, T):
            phi_t = agent.preprocess_frames(
                torch.stack(last_frames)
            ).float()  # stack last 4 frames and process -> (4, 84, 84)
            action = agent.get_action(phi_t.unsqueeze(0))  # add a batch dim

            new_s, r, done, _, _ = env.step(action)
            reward = float(max(-1, min(1, r)))  # pyright: ignore
            total_reward += reward

            # add new state to seq
            last_frames.pop(0)
            last_frames.append(torch.tensor(new_s))

            # update replay memory
            phi_tp = agent.preprocess_frames(torch.stack(last_frames)).float()
            if len(replay_memory) == 100000:
                replay_memory.pop(0)
            replay_memory.append(
                (
                    phi_t,
                    action,
                    reward,
                    done,
                    phi_tp,
                )
            )

            # update agent q values and get last state dict
            update_agent(replay_memory, env, agent)

            if done:
                break

        if ep % 10 == 0:
            print(
                f"Total reward on last simulation after {ep} simulations: {total_reward}"
            )

        if ep % 1000 == 0:
            torch.save(agent.model.state_dict(), "dql.pt")


train_deep_q_learning()
