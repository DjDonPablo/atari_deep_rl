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
        self.gamma = 0.99
        self.legal_actions = list(range(n_actions))
        self.batch_size = batch_size
        self.device = torch.device("cuda:0")

        self.model = QFunction(n_actions).to(self.device)
        self.target_model = QFunction(n_actions).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = torch.optim.RMSprop(self.model.parameters(), learning_rate)  # pyright: ignore
        self.loss_fn = MSELoss()
        self.resizer = torch.nn.functional.interpolate

    def preprocess_frames(self, state: State) -> State:
        """
        Downscaling `state` of shape (4, 210, 160) to (4, 84, 84) by interpolation
        """
        return self.resizer(
            state.unsqueeze(0),
            size=(110, 84),
            mode="bilinear",
            align_corners=False,
        )[:, :, 17:-9].squeeze(0)

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
            self.model((phi_ts / 255).to(self.device))
            .gather(1, actions.to(torch.int64).unsqueeze(1).to(self.device))
            .squeeze(1)
        )

        with torch.no_grad():
            targets = rewards.to(self.device) + (1 - dones).to(self.device) * self.gamma * self.get_value(phi_tps)

        loss = self.loss_fn(outputs, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()

    def get_value(self, phi_tp: torch.Tensor) -> torch.Tensor:
        """
        Get best qvalue from target model
        """
        return torch.max(self.target_model((phi_tp / 255).to(self.device)), dim=1)[0]

    def get_best_action(self, state: State) -> Action:
        """
        Get best action from model
        """
        return Action(torch.argmax(self.model((state / 255).to(self.device))[0]).item())

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

    def decay_epsilon(self, step):
        """
        Update epsilon value using `epsilon_decay` and `epsilon_min`
        """
        if step >= 50000 and step <= 1000000:
            self.epsilon -= 0.9 / 950000
