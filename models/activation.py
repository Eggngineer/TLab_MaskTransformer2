from torch import nn
import torch.nn.functional as F

import torch


# Mish
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


# Swish
class Swish(nn.Module):
    def __init__(self, beta: float = 1.0) -> None:
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)
