import torch
import wandb
import hydra

from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from .. config.train_config import TrainConfig
from .. lib.functions import index_points, thresholding, create_model
from .. data.dataloader import get_dataloader


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


class Boundary:
    def __init__(
            self,
            cfg: TrainConfig = TrainConfig(),
    ) -> None:
        super(Boundary,self).__init__()
        self.cfg = cfg
        self.boundary_th = self.cfg.boundary.th

    def get_method(
        self,
    ) -> None:
        if self.cfg.boundary.method == "and":
            self.method = self.boudary_and
            self.th = self.cfg.mask.th
        elif self.cfg.boundary.method == "param_maxmin":
            self.method = self.boudary_param_maxmin
            self.alpha = self.cfg.boundary.alpha
            self.beta = self.cfg.boundary.beta
        elif self.cfg.boundary.method == "param_max":
            self.method = self.boudary_param_max
            self.gamma = self.cfg.boundary.gamma
            self.th = self.cfg.mask.th
        elif self.cfg.boundary.method == "param_mean":
            self.method = self.boudary_param_mean
            self.mu = self.cfg.boundary.mu
            self.k = self.cfg.boundary.nearest_k

    def get_boundary(
        self,
        pos: torch.Tensor,
        mask: torch.Tensor,
        gt: bool = False,
    ) -> torch.Tensor:
        ret = self.method(pos=pos, mask=mask)
        if gt:
            return thresholding(mask=ret, th=self.boundary_th)
        else:
            return ret

    def boudary_and(
            self,
            pos: torch.Tensor,
            mask: torch.Tensor,
    ):
        """boudary_and

        Args:
            mask (torch.Tensor): (B x N x 1)
                B: number of batch size,
                N: number of points

            knn (torch.Tensor): (B x N x k)
                B: number of batch size,
                N: number of points,
                k: num of neighbors

        Returns:
            boudary (torch.Tensor): (B x N x 1)
                B: number of batch size,
                N: number of points
        """
        boundary = torch.zeros_like(mask)
        knn = index_points(
            points=mask.unsqueeze(-1),
            idx=knn(pos, pos, k=self.k),
        ).squeeze(-1)

        boundary[mask > self.th] = (mask * (1 - torch.min(knn, dim=-1)[0].squeeze(-1)))[mask > self.th]

        return boundary


    def boudary_param_maxmin(
            self,
            pos: torch.Tensor,
            mask: torch.Tensor,
    ):
        """boudary_param_maxmin

        Args:
            pos (torch.Tensor): (B x N x 3)
                B: number of batch size,
                N: number of points
            mask (torch.Tensor): (B x N x 1)
                B: number of batch size,
                N: number of points

        Returns:
            boudary (torch.Tensor): (B x N x 1)
                B: number of batch size,
                N: number of points
        """
        knn = index_points(
            points=mask.unsqueeze(-1),
            idx=knn(pos, pos, k=self.k),
        ).squeeze(-1)

        knn_withme = torch.cat((mask.unsqueeze(-1), knn), dim=-1)

        boundary = (
            self.alpha
            * (torch.max(knn_withme, dim=-1)[0] - self.beta * torch.min(knn_withme, dim=-1)[0])
        )

        return boundary

    def boudary_param_max(
            self,
            pos: torch.Tensor,
            mask: torch.Tensor,
    ):
        """boudary_param_max

        Args:
            pos (torch.Tensor): (B x N x 3)
                B: number of batch size,
                N: number of points
            mask (torch.Tensor): (B x N x 1)
                B: number of batch size,
                N: number of points

        Returns:
            boudary (torch.Tensor): (B x N x 1)
                B: number of batch size,
                N: number of points
        """
        boundary = torch.zeros_like(mask)
        knn = index_points(
            points=mask.unsqueeze(-1),
            idx=knn(pos, pos, k=self.k),
        ).squeeze(-1)

        boundary[mask <= self.th] = (
            self.gamma
            * (torch.max(knn, dim=-1)[0] - 1 * mask)
        )[mask <= self.th]

        return boundary

    def boudary_param_mean(
            self,
            pos: torch.Tensor,
            mask: torch.Tensor,
    ):
        """boudary_param_mean

        Args:
            pos (torch.Tensor): (B x N x 3)
                B: number of batch size,
                N: number of points
            mask (torch.Tensor): (B x N x 1)
                B: number of batch size,
                N: number of points

        Returns:
            boudary (torch.Tensor): (B x N x 1)
                B: number of batch size,
                N: number of points
        """
        knn = index_points(
            points=mask.unsqueeze(-1),
            idx=knn(pos, pos, k=self.k),
        ).squeeze(-1)

        boundary = (
            torch.sigmoid(
                self.mu * (
                    (
                        mask
                        - knn.mean(dim=-1)
                    ) ** 2 - 1 / (2 * float(self.k) ** 2)
                )
            )
        )

        return boundary


def select_optimizer(
    model: torch.nn.Module,
    cfg: TrainConfig,
):
    optim_name = cfg.train.optimizer
    params = model.parameters()
    if optim_name == "Adam":
        return torch.optim.Adam(
            params=params,
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )
    elif optim_name == "AdamW":
        return torch.optim.AdamW(
            params=params,
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )
    elif optim_name == "SGD":
        return torch.optim.SGD(
            params=params,
            lr=cfg.train.lr,
        )
    else:
        return torch.optim.SGD(
            params=params,
            lr=cfg.train.lr,
        )


def select_loss_fn(
    cfg: TrainConfig,
    positive_weight: float = 1.0
):
    if cfg.train.loss == 'BCE':
        return nn.BCELoss()
    elif cfg.train.loss == 'MSE':
        return nn.MSELoss()
    elif cfg.train.loss == 'BCEWithLogit':
        return nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    else:
        return nn.BCELoss()


def train(cfg: TrainConfig):
    model = create_model(cfg=cfg)
    wandb.watch(model)
    optimizer = select_optimizer(cfg=cfg)
    boundary = Boundary(cfg=cfg)
    train_loader = get_dataloader(cfg=cfg, phase="train")
    val_loader = get_dataloader(cfg=cfg, phase="val")
    loss_fn = select_loss_fn(cfg=cfg, positive_weight=1.0)
    boundary_loss_fn = select_loss_fn(cfg=cfg, positive_weight=cfg.train.positive_weight)

    for epoch in enumerate(tqdm(range(1,cfg.train.epoch + 1))):
        train_loss = train_one_epoch(
            cfg=cfg,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            boundary=boundary,
            mask_loss_fn=loss_fn,
            boundary_loss_fn=boundary_loss_fn,
            late_start=(epoch >= cfg.train.late_start)
        )

        if epoch % cfg.valid.per_epochs == 0:
            val_loss = val_one_epoch(
                cfg=cfg,
                model=model,
                loader=val_loader,
                boundary=boundary,
                mask_loss_fn=loss_fn,
                boundary_loss_fn=boundary_loss_fn,
                late_start=(epoch >= cfg.train.late_start)
            )
            wandb.log(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }
            )

            hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
            output_dir = hydra_cfg['runtime']['output_dir']

            torch.save(obj=model.state(),f=str(Path(output_dir) / f"ckpt_{epoch:04}.t7"))

        else:
            wandb.log(
                {
                    "train_loss": train_loss,
                }
            )


def train_one_epoch(
    cfg: TrainConfig,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    boundary: Boundary,
    mask_loss_fn,
    boundary_loss_fn,
    late_start: bool = False,
):
    model.train()
    loss_sum_epoch = AverageMeter(0)
    loss_mask_epoch = AverageMeter(0)
    loss_boundary_epoch = AverageMeter(0)


    for idx, data in enumerate(tqdm(loader)):
        source, target, igt, gt_src_mask, gt_tgt_mask, label, rot = data

        source.to("cuda")
        target.to("cuda")
        gt_src_mask.to("cuda")
        gt_tgt_mask.to("cuda")

        pred_src_mask, pred_tgt_mask = model()

        loss_iter = torch.tensor(0,requires_grad=True)

        if cfg.model.boundary and late_start:
            pred_src_boundary = boundary.get_boudnary(pos=source, mask=pred_src_mask).to("cuda")
            pred_tgt_boundary = boundary.get_boudnary(pos=target, mask=pred_tgt_mask).to("cuda")

            gt_src_boundary = boundary.get_boudnary(pos=source, mask=gt_src_mask).to("cuda")
            gt_tgt_boundary = boundary.get_boudnary(pos=target, mask=gt_tgt_mask).to("cuda")

            loss_boundary_src = boundary_loss_fn(pred_src_boundary, gt_src_boundary)
            loss_boundary_tgt = boundary_loss_fn(pred_tgt_boundary, gt_tgt_boundary)
            loss_boundary = (loss_boundary_src + loss_boundary_tgt) / 2

        loss_mask_src = mask_loss_fn(pred_src_mask, gt_src_mask)
        loss_mask_tgt = mask_loss_fn(pred_tgt_mask, gt_tgt_mask)
        loss_mask = (loss_mask_src + loss_mask_tgt) / 2


        loss_iter = cfg.train.loss_balance[0] * loss_mask + cfg.train.loss_balance[1] * loss_boundary

        optimizer.zero_grad()
        loss_iter.backward()
        optimizer.step()

        loss_sum_epoch.update(loss_iter.item())
        loss_mask_epoch.update(loss_mask.item())
        loss_boundary_epoch.update(loss_boundary.item())

    wandb.log(
        {
            "train_loss_mask": loss_mask_epoch.avg,
            "train_loss_boundary": loss_boundary_epoch.avg,
            "train_loss_ssum": loss_sum_epoch.avg,
        }
    )

    return loss_iter.item()


def val_one_epoch(
    cfg: TrainConfig,
    model: nn.Module,
    loader: DataLoader,
    boundary: Boundary,
    mask_loss_fn,
    boundary_loss_fn,
    late_start: bool = False,
):
    model.eval()
    loss_sum_epoch = AverageMeter(0)
    loss_mask_epoch = AverageMeter(0)
    loss_boundary_epoch = AverageMeter(0)


    for idx, data in enumerate(tqdm(loader)):
        source, target, igt, gt_src_mask, gt_tgt_mask, label, rot = data

        source.to("cuda")
        target.to("cuda")
        gt_src_mask.to("cuda")
        gt_tgt_mask.to("cuda")

        pred_src_mask, pred_tgt_mask = model()

        loss_iter = torch.tensor(0,requires_grad=True)

        if cfg.model.boundary and late_start:
            pred_src_boundary = boundary.get_boudnary(pos=source, mask=pred_src_mask).to("cuda")
            pred_tgt_boundary = boundary.get_boudnary(pos=target, mask=pred_tgt_mask).to("cuda")

            gt_src_boundary = boundary.get_boudnary(pos=source, mask=gt_src_mask).to("cuda")
            gt_tgt_boundary = boundary.get_boudnary(pos=target, mask=gt_tgt_mask).to("cuda")

            loss_boundary_src = boundary_loss_fn(pred_src_boundary, gt_src_boundary)
            loss_boundary_tgt = boundary_loss_fn(pred_tgt_boundary, gt_tgt_boundary)
            loss_boundary = (loss_boundary_src + loss_boundary_tgt) / 2

        loss_mask_src = mask_loss_fn(pred_src_mask, gt_src_mask)
        loss_mask_tgt = mask_loss_fn(pred_tgt_mask, gt_tgt_mask)
        loss_mask = (loss_mask_src + loss_mask_tgt) / 2


        loss_iter = cfg.train.loss_balance[0] * loss_mask + cfg.train.loss_balance[1] * loss_boundary

        loss_sum_epoch.update(loss_iter.item())
        loss_mask_epoch.update(loss_mask.item())
        loss_boundary_epoch.update(loss_boundary.item())

    wandb.log(
        {
            "val_loss_mask": loss_mask_epoch.avg,
            "val_loss_boundary": loss_boundary_epoch.avg,
            "val_loss_ssum": loss_sum_epoch.avg,
        }
    )

    return loss_iter.item()








