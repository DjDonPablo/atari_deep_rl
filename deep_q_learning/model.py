from torch import nn
from torch.functional import Tensor
from torch.nn.functional import relu


class QFunction(nn.Module):
    def __init__(self, nb_actions: int):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=(8, 8), stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1)
        self.dense1 = nn.Linear(3136, 512)
        self.dense2 = nn.Linear(512, nb_actions)

    def forward(self, x: Tensor):
        x = relu(self.conv1(x))
        x = relu(self.conv2(x))
        x = relu(self.conv3(x))
        x = x.flatten(1)
        x = relu(self.dense1(x))
        return self.dense2(x)
