from gymnasium.spaces import Discrete

class DeepQLearningAgent:
    def __init__(self, learning_rate: float, epsilon: float, nb_actions: int, batch_size: int):
        self.action_space = Discrete(nb_actions)
        self.batch_size = batch_size