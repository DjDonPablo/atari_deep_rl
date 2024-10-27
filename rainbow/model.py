import torch

from torch import nn
from torch.nn.functional import relu
from torchrl.modules import NoisyLinear


class RainbowQFunction(nn.Module):
    def __init__(self, nb_actions: int, noisy_nets_std: float = 0.5):
        super().__init__()
        self.nb_actions = nb_actions

        self.conv1 = nn.Conv2d(4, 32, kernel_size=(8, 8), stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1)

        self.dense1_adv = NoisyLinear(3136, 512, std_init=noisy_nets_std)
        self.dense1_val = NoisyLinear(3136, 512, std_init=noisy_nets_std)

        self.dense2_adv = NoisyLinear(512, nb_actions, std_init=noisy_nets_std)
        self.dense2_val = NoisyLinear(512, 1, std_init=noisy_nets_std)

    def forward(self, x: torch.Tensor):
        x = relu(self.conv1(x))
        x = relu(self.conv2(x))
        x = relu(self.conv3(x))
        x = x.view(x.shape[0], -1)

        adv = relu(self.dense1_adv(x))
        val = relu(self.dense1_val(x))

        adv = self.dense2_adv(adv)
        val = self.dense2_val(val).expand(x.shape[0], self.nb_actions)

        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.shape[0], self.nb_actions)
        return x
