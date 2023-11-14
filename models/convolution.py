from torch import nn
from activation import Mish
import torch


class BasicConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, active=True):
        super(BasicConv1D, self).__init__()
        self.active = active
        self.bn = nn.BatchNorm1d(out_channels)
        if self.active:
            self.activation = Mish()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, bias=False
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.active:
            x = self.activation(x)
        return x

