import torch
import torch.nn as nn
import torch.nn.functional as F


def _torch_max_(
    x: torch.Tensor,
    dim: int = -1,
    keepdim: bool = False,
):
    return torch.max(x, dim=dim, keepdim=keepdim)[0]


def _torch_argmax_(
    x: torch.Tensor,
    dim: int = -1,
    keepdim: bool = False,
):
    return torch.max(x, dim=dim, keepdim=keepdim)[1]



class old_Pooling(torch.nn.Module):
    def __init__(self, pool_type="max"):
        self.pool_type = pool_type
        super(old_Pooling, self).__init__()

    def forward(self, input):
        if self.pool_type == "max":
            return torch.max(input, 2)[0].contiguous()
        elif self.pool_type == "avg" or self.pool_type == "average":
            return torch.mean(input, 2).contiguous()


class Pooling(torch.nn.Module):
    def __init__(
        self,
        pool_type: str = "max",
        dim: int = -1,
        keepdim: bool = False,
    ):
        self.pool_type = pool_type
        self.dim = dim
        self.keepdim = keepdim
        super(Pooling, self).__init__()

    def forward(self, input):
        if self.pool_type in {"max", "Max", "MAX"}:
            return _torch_argmax_(
                x=input,
                dim=self.dim,
                keepdim=self.keepdim
            )
        elif self.pool_type in {"avg", "average", "Avg"}:
            return torch.mean(
                input=input,
                dim=self.dim,
                keepdim=self.keepdim,
            )