# atari_deep_rl

This repository contains 2 implementations of Deep Q Learning models for the Atari game **Breakout** :
- A basic Deep Q Learning implementation.
- A Rainbow implementation with 4 of the 6 optimizations discussed in the paper.

## Basic Deep Q Learning

Model architecture can be found in */deep_q_learning/model.py*.

### Hyperparameters

- **epsilon**: first 50k steps -> 1.0 | 50k to 1M -> linear decrease to 0.1 | after 1M -> keep at 0.1
- **gamma**: 0.99
- **lr**: 1.5e-4
- **batch_size**: 32
- **replay memory size**: 180k
- **update target model frequency**: every 5k steps
- **clipped reward**: between -1 and 1

Optimizer is RMSProp.

### Demo

When playing, **epsilon** is at 0.05 and the agent uses its 5 lifes. The following agent was trained on 10k simulations.

![Alt Text](https://github.com/DjDonPablo/atari_deep_rl/blob/main/dql_10k_01eps_005inf.gif)

## Rainbow Q Learning

Optimizations implemented from the rainbow paper are : Double Q Learning, Dual Network Architecture, Noisy Networks and Prioritized Experience Replay.\
Model architecture can be found in */rainbow/model.py*.

### Hyperparameters

- **epsilon**: 0.0 (since noisy nets)
- **gamma**: 0.99
- **lr**: 1.5e-4
- **batch_size**: 32
- **noisy nets std init**: 0.5
- **replay memory size**: 180k
- **replay memory alpha**: 0.6
- **replay memory beta**: 0.4
- **min history length before learning**: 80k
- **update target model frequency**: every 5k steps
- **clipped reward**: between -1 and 1

Optimizer is Adam, with **eps** equal to 1.5e-4.

### Demo

When playing, agent uses only its first life. The following agent was trained on 16k simulations.

![Alt Text](https://github.com/DjDonPablo/atari_deep_rl/blob/main/rainbow_16k.gif)

## How to test
### Requirements

- torch
- torchrl
- gymnasium
- ale_py

### Run the script

Go inside either */deep_q_learning* or */rainbow* and do the following :

```sh
python test.py
```

You can change the *weights* file for the Rainbow agent.

## Bibliography

Playing Atari with Deep Reinforcement Learning : https://arxiv.org/pdf/1312.5602 \
Deep Reinforcement Learning with Double Q-learning : https://arxiv.org/pdf/1509.06461 \
Dueling Network Architectures for Deep Reinforcement Learning : https://arxiv.org/pdf/1511.06581 \
Prioritized Experience Replay : https://arxiv.org/pdf/1511.05952 \
Noisy Networks for Exploration : https://arxiv.org/pdf/1706.10295 \
Rainbow: Combining Improvements in Deep Reinforcement Learning : https://arxiv.org/pdf/1710.02298
