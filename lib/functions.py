import datetime
import torch
import open3d as o3d
import numpy as np

from typing import Union

from ..models.masktransformer import MaskTransformer_ver2
from ..config.train_config import TrainConfig

def thresholding(mask: torch.Tensor, th: float = 0.5):
    return (mask > th) * 1.0

def now(form="%Y%m%d%H%M%S"):
    """
    """
    get_now = datetime.datetime.now()
    return get_now.strftime(form)


def calc_overlap(
    gt_mask: torch.Tensor,
) -> float:
    """params
    :param: gt_mask, Ground Truth Mask (B, N)
    """
    non_zero_counts = torch.count_nonzero(gt_mask != 0, dim=1)
    return non_zero_counts.to(torch.float).mean().item() / gt_mask.shape[1]


def tensor2points(tensor: torch.Tensor) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(tensor.numpy())
    return pcd


def ndarray2points(array: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(array)
    return pcd


def points2ndarray(
    pcd: o3d.geometry.PointCloud,
) -> np.ndarray:
    return np.asarray(pcd.points)


class AverageMeter(object):
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def evaluate_mask(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
):
    """culcurating confusion matrix

    Args:
        pred_mask (torch.Tensor): BxN in {0,1}
        gt_mask (torch.Tensor): BxN, in {0,1}

    Returns:
        metrics (torch.Tensor): (
            accuracy: B,
            recall: B,
            precision: B,
            f1-score: B,
        )
    """
    tp = torch.count_nonzero(torch.logical_and(input=pred_mask, other=gt_mask)).item()
    fp = torch.count_nonzero(
        torch.logical_xor(input=pred_mask, other=gt_mask) * pred_mask
    ).item()
    fn = torch.count_nonzero(
        torch.logical_xor(input=pred_mask, other=gt_mask) * gt_mask
    ).item()
    tn = torch.count_nonzero(
        torch.logical_and(
            input=torch.logical_not(pred_mask), other=torch.logical_not(gt_mask)
        )
    ).item()

    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    recall = (tp) / max(tp + fn, 1)
    precision = (tp) / max(tp + fp, 1)
    f1score = 2 / (1 / max(recall, 1e-5) + 1 / max(precision, 1e-5))

    return accuracy, recall, precision, f1score


def tensor_max(
    x: torch.Tensor,
    dim: int = -1,
    keepdim: bool = True,
):
    return torch.max(x, dim=dim, keepdim=keepdim)[0]


def tensor_argmax(
    x: torch.Tensor,
    dim: int = -1,
    keepdim: bool = True,
):
    return torch.max(x, dim=dim, keepdim=keepdim)[1]


def index_points(points, idx):
    """
    :param points: points.shape == (B, N, C)
    :param idx: idx.shape == (B, N, K)
    :return:indexed_points.shape == (B, N, K, C)
    """
    raw_shape = idx.shape
    idx = idx.reshape(raw_shape[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.shape[-1]))
    return res.view(*raw_shape, -1)


def knn(a, b, k):
    """
    :param a: a.shape == (B, N, C)
    :param b: b.shape == (B, M, C)
    :param k: int
    """
    inner = -2 * torch.matmul(a, b.transpose(2, 1))  # inner.shape == (B, N, M)
    aa = torch.sum(a**2, dim=-1, keepdim=True)  # aa.shape == (B, N, 1)
    bb = torch.sum(b**2, dim=-1, keepdim=True)  # bb.shape == (B, M, 1)
    pairwise_distance = (
        - aa - inner - bb.transpose(2, 1)
    )  # pairwise_distance.shape == (B, N, M)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # idx.shape == (B, N, K)
    return idx


def create_model(
    #TODO Test作成し次第Union[TrainConfig, TestConfig]へと型変更
    cfg: TrainConfig,
):
    model_name = cfg.model.name
    if model_name == "MaskTransformer_ver2":
        return MaskTransformer_ver2(
            last_layer_sigmoid = cfg.model.last_layer_sigmoid,
        )
    else:
        return MaskTransformer_ver2(
            last_layer_sigmoid = cfg.model.last_layer_sigmoid,
        )

