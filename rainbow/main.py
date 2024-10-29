import gymnasium as gym
import ale_py
import torch
import time

from agent import RainbowAgent

gym.register_envs(ale_py)

Action = int
Reward = int


def train_rainbow():
    env = gym.make(
        "ALE/Breakout-v5",
        obs_type="grayscale",
        repeat_action_probability=0,
    )
    agent = RainbowAgent(1.5e-4, env.action_space.n)  # pyright: ignore

    last_total_steps = 0
    total_steps = 0
    total_reward = 0
    start = time.time()

    for ep in range(1, agent.max_ep):
        s, _ = env.reset()
        last_frames = [torch.tensor(s) for _ in range(3)]
        s, _, _, _, _ = env.step(1)
        last_frames.append(torch.tensor(s))

        for _ in range(1, agent.max_frames_per_ep):
            agent.model.reset_noise()
            phi_t = agent.preprocess_frames(torch.stack(last_frames))
            action = agent.get_action(phi_t.unsqueeze(0))

            # execute action in env and clip reward
            new_s, r, done, _, info = env.step(action)
            if info["lives"] == 4:
                reward = -1
                done = True
            else:
                reward = min(1, int(r))  # pyright: ignore
            total_reward += int(reward)

            # pop first oldest state and add new one
            last_frames.pop(0)
            last_frames.append(torch.tensor(new_s))

            phi_tp = agent.preprocess_frames(torch.stack(last_frames)).float()

            # update replay buffer
            agent.replay_buffer.add(
                (
                    phi_t,
                    torch.tensor(action, dtype=torch.uint8),
                    torch.tensor(reward, dtype=torch.int8),
                    torch.tensor(done, dtype=torch.uint8),
                    phi_tp,
                )
            )

            # update agent q values
            if total_steps > agent.min_history_start_learning:
                batch = agent.replay_buffer.sample(agent.batch_size)
                agent.update(*batch)

            total_steps += 1
            if total_steps % agent.update_target_model_freq == 0:
                agent.update_target_model()

            if done:
                break

        if ep % 10 == 0:
            print(
                f"mean reward over 10 simulations and {total_steps - last_total_steps} steps : {round(total_reward / 10, 2)} - trained on {ep} simulations - {total_steps} total steps - {round(time.time() - start, 2)} sec",
                flush=True,
            )
            total_reward = 0
            last_total_steps = total_steps
            start = time.time()

        if ep % 2000 == 0:
            torch.save(agent.model.state_dict(), f"ep_{ep}.pt")


if __name__ == "__main__":
    train_rainbow()
