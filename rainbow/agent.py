import torch

from torch.nn import MSELoss
from model import RainbowQFunction

Action = int
State = torch.Tensor


class RainbowAgent:
    def __init__(
        self,
        learning_rate: float,
        n_actions: int,
        batch_size: int = 32,
    ):
        self.min_history_start_learning = 80000
        self.max_ep = 100000
        self.max_frames_per_ep = 100000
        self.replay_memory_size = 180000
        self.update_target_model_freq = 10000
        self.gamma = 0.99
        self.legal_actions = list(range(n_actions))
        self.batch_size = batch_size
        self.device = torch.device("cpu")

        self.model = RainbowQFunction(n_actions).to(self.device)
        self.target_model = RainbowQFunction(n_actions).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), learning_rate, eps=1.5e-4
        )
        self.loss_fn = MSELoss()
        self.resizer = torch.nn.functional.interpolate

    def preprocess_frames(self, state: State) -> State:
        """
        Downscaling `state` of shape (4, 210, 160) to (4, 84, 84) by interpolation
        Output is not divide by 255 to keep the uint8 type to take less memory
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
            targets = rewards.to(self.device) + (1 - dones).to(
                self.device
            ) * self.gamma * self.get_target_qvalue(phi_tps)

        loss = self.loss_fn(outputs, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()

    def get_target_qvalue(self, phi_tp: torch.Tensor) -> torch.Tensor:
        """
        Get target Qvalue using the base model for action selection and target model for action evaluation
        """
        # use base model for action selection
        best_actions = torch.argmax(self.model((phi_tp / 255).to(self.device)), dim=1)

        # use target model for action evaluation
        return (
            self.target_model((phi_tp / 255).to(self.device))
            .gather(1, best_actions.unsqueeze(1))
            .flatten()
        )

    def get_action(self, state: State) -> Action:
        """
        Return an action following an epsilon-greedy algorithm
        """
        return Action(torch.argmax(self.model((state / 255).to(self.device))[0]).item())

    def update_target_model(self):
        """
        Update target model with model state dict
        """
        self.target_model.load_state_dict(self.model.state_dict())
