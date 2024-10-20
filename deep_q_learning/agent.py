import torch
import random

from gymnasium.spaces import Discrete
from torch.nn import MSELoss
from model import QFunction
from torchvision.transforms import Resize

Action = int
State = torch.Tensor


class DeepQLearningAgent:
    def __init__(
        self,
        learning_rate: float,
        epsilon: float,
        n_actions: int,
        batch_size: int = 32,
        skip_frames: int = 4,
    ):
        self.epsilon = epsilon
        self.legal_actions = list(range(n_actions))
        self.action_space = Discrete(n_actions)
        self.batch_size = batch_size
        self.model = QFunction(n_actions)
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), learning_rate)  # pyright: ignore
        self.loss_fn = MSELoss()
        self.resizer = Resize((110, 84))

    def preprocess_frames(self, state: State) -> State:
        """
        Preprocess last 4 frames to get them to shape (4, 84, 84)
        """
        return self.resizer(state).clone()[:, 16:-10]

    def update(
        self,
        preprocessed_sequence: torch.Tensor,
        y: torch.Tensor,
        actions: torch.Tensor,
    ) -> None:
        self.optimizer.zero_grad()

        output = self.model(preprocessed_sequence)
        output = output.gather(1, actions.unsqueeze(1)).squeeze(1)

        loss_value = self.loss_fn(output, y)
        loss_value.backward()

        self.optimizer.step()

    def get_value(self, phi_tp: torch.Tensor) -> torch.Tensor:
        return torch.max(self.model(phi_tp), dim=1)[0]

    def get_best_action(self, state: State) -> Action:  # state is [(84, 84) * 4]
        return Action(torch.argmax(self.model(state)[0]).item())

    def get_action(self, state: State) -> Action:
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.legal_actions)
        else:
            return self.get_best_action(state)
