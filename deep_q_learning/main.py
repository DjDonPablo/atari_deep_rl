import gymnasium as gym
import random
import ale_py
import torch

from typing import Dict, List, Tuple
from gymnasium.core import Env
from agent import DeepQLearningAgent

gym.register_envs(ale_py)

env = gym.make("ALE/Breakout-v5", obs_type="grayscale")  # , render_mode="human"
s, _ = env.reset(seed=42)

agent = DeepQLearningAgent(0.01, 0.05, env.action_space.n)  # pyright: ignore

Action = int
Reward = float


def update_replay_memory(
    replay_memory: List[Tuple[torch.Tensor, Action, Reward, bool, torch.Tensor]],
    agent: DeepQLearningAgent,
    phi_t: torch.Tensor,
    action: Action,
    reward: Reward,
    phi_tp: torch.Tensor,
    done: bool,
):
    if len(replay_memory) == 50000:
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


def update_agent(
    replay_memory: List[Tuple[torch.Tensor, Action, Reward, bool, torch.Tensor]],
    env: Env,
    agent: DeepQLearningAgent,
    learning_rate: float,
    last_state_dict: Dict,
):
    batch_size = min(len(replay_memory), agent.batch_size)
    batch = random.sample(replay_memory, batch_size)

    phi_t_batch = torch.stack([b[0] for b in batch])
    action_batch = torch.tensor([b[1] for b in batch])
    reward_batch = torch.tensor([b[2] for b in batch])
    is_not_done = torch.tensor([not b[3] for b in batch], dtype=torch.float32)
    phi_tp_batch = torch.stack([transition[4] for transition in batch])

    y = is_not_done * learning_rate * agent.get_value(phi_tp_batch, last_state_dict)
    y += reward_batch

    new_last_state_dict = agent.model.state_dict()
    agent.update(phi_t_batch, y.scatter_(1, action_batch.unsqueeze(1), y.unsqueeze(1)))

    return new_last_state_dict


def train_deep_q_learning(
    agent: DeepQLearningAgent,
    env: Env,
    learning_rate: float = 0.5,
    M: int = 10000,
    T: int = 50000,
):
    replay_memory: List[
        Tuple[torch.Tensor, Action, Reward, bool, torch.Tensor]
    ] = []  # max size is 50 000
    last_state_dict = agent.model.state_dict()

    for ep in range(1, M):
        total_reward = 0.0
        s, _ = env.reset(seed=42)
        last_frames = [s]
        a = 0
        # warmup
        for _ in range(agent.skip_frames - 1):
            new_s, r, done, _, _ = env.step(a)
            total_reward += r  # pyright: ignore
            last_frames.append(new_s)

        for t in range(1, T):
            phi_t = agent.preprocess_frames(
                torch.stack(last_frames)
            )  # stack last 4 frames and process -> (4, 84, 84)
            a = agent.get_action(phi_t.unsqueeze(0))  # add a batch dim

            new_s, r, done, _, _ = env.step(a)
            total_reward += r  # pyright: ignore

            # add new state to seq
            last_frames.pop(0)
            last_frames.append(new_s)

            # update replay memory
            phi_tp = agent.preprocess_frames(torch.stack(last_frames))
            update_replay_memory(replay_memory, agent, phi_t, a, float(r), phi_tp, done)

            # update agent q values and get last state dict
            last_state_dict = update_agent(
                replay_memory, env, agent, learning_rate, last_state_dict
            )

            if done:
                break

        if ep % 10:
            print(f"Total reward after {ep} simulations: {total_reward}")

    torch.save(agent.model.state_dict(), "dql.pt")


# train_deep_q_learning(agent, env)
