# Playing Atari with Deep Reinforcement Learning

CNN trained with a variant of Q-learning, raw pixels as input and output being value function to estimate future rewards.

Agent have only access to images of the current screen and a reward for each action it takes.

Targets depend on the network weights (ones from previous iteration).

They use a technique call _experience replay_, which stores the agent's experiences at each time step, to apply Q-learning updates to sample of experience during the inner loop of the algorithm. This smooth out learning and avoid oscillations or divergence in the parameters.

# Deep Reinforcement Learning with Double Q-learning

# Dueling Network Architectures for Deep Reinforcement Learning

# Hindsight Experience Replay

# Rainbow: Combining Improvements in Deep Reinforcement Learning
