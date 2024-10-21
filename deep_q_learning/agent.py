import torch
import random

from torch.nn import MSELoss
from model import QFunction

Action = int
State = torch.Tensor


class DeepQLearningAgent:
    def __init__(
        self,
        learning_rate: float,
        epsilon: float,
        n_actions: int,
        batch_size: int = 32,
    ):
        self.epsilon = epsilon
        self.epsilon_decay = 0.998
        self.epsilon_min = 0.08
        self.legal_actions = list(range(n_actions))
        self.batch_size = batch_size

        self.model = QFunction(n_actions)
        self.target_model = QFunction(n_actions)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = torch.optim.AdamW(  # pyright: ignore
            self.model.parameters(), learning_rate, amsgrad=True
        )
        self.loss_fn = MSELoss()
        self.resizer = torch.nn.functional.interpolate

    def preprocess_frames(self, state: State) -> State:
        """
        Downscaling `state` of shape (4, 210, 160) to (4, 84, 84) by interpolation
        """
        return self.resizer(
            state.unsqueeze(0),
            size=(84, 84),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    def update(
        self,
        phi_ts: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        phi_tps: torch.Tensor,
    ):
        """
        Update qvalue estimator
        """
        self.optimizer.zero_grad()

        outputs = (
            self.model(phi_ts / 255)
            .gather(1, actions.to(torch.int64).unsqueeze(1))
            .squeeze(1)
        )

        with torch.no_grad():
            targets = rewards + (1 - dones) * 0.99 * self.get_value(phi_tps)

        loss = self.loss_fn(outputs, targets)
        loss.backward()

        self.optimizer.step()

    def get_value(self, phi_tp: torch.Tensor) -> torch.Tensor:
        """
        Get best qvalue from target model
        """
        return torch.max(self.target_model(phi_tp / 255), dim=1)[0]

    def get_best_action(self, state: State) -> Action:
        """
        Get best action from model
        """
        return Action(torch.argmax(self.model(state / 255)[0]).item())

    def get_action(self, state: State) -> Action:
        """
        Return an action following an epsilon-greedy algorithm
        """
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.legal_actions)
        else:
            return self.get_best_action(state)

    def update_target_model(self):
        """
        Update target model with model state dict
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def decay_epsilon(self):
        """
        Update epsilon value using `epsilon_decay` and `epsilon_min`
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
