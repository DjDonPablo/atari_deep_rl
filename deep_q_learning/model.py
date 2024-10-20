import torch

from torch import nn
from torch.functional import Tensor
from torch.nn.functional import relu


class QFunction(nn.Module):
    def __init__(self, nb_actions: int):
        super().__init__()
        # input is of shape (BATCH_SIZE, NB_CHANNELS, HEIGHT, WIDTH) = (1-32, 4, 84, 84)
        self.conv1 = nn.Conv2d(4, 16, kernel_size=(8, 8), stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(4, 4), stride=2)
        self.dense = nn.Linear(2592, nb_actions)

    def forward(self, x: Tensor):
        x = relu(self.conv1(x))
        x = relu(self.conv2(x))
        x = x.flatten(1)
        return self.dense(x)
