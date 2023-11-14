import torch
from numpy import pi
from typing import Union

from torch.utils.data import DataLoader

from .. config.train_config import TrainConfig
from .datasets import (
    ModelNet40Data,
    RegistrationData3DM,
)


def select_datasets(
    #TODO test実装時にUnion[TrainConfig, TestConfig]に変更
    cfg: TrainConfig,
    phase: str = "train",
):
    pidict = {
        '0': 0.0,
        'pi/6': pi/6,
        'pi/4': pi/4,
        'pi/3': pi/3,
        'pi/2': pi/2,
        'pi': pi,
        '2pi': 2 * pi,
    }

    if phase == "train":
        if cfg.loader.train.dataset_name == "ThreeDMatch":
            return RegistrationData3DM(
                num_points=cfg.loader.train.three_d_match.num_points,
                noise=cfg.loader.train.three_d_match.noise,
                outliers=cfg.loader.train.three_d_match.outliers,
                rotMax=pidict(cfg.loader.train.three_d_match.rotMax),
                rate=cfg.loader.train.three_d_match.rate,
                randomize=cfg.loader.train.three_d_match.randomize,
                phase="train"
            )
        elif cfg.loader.train.dataset_name == "ModelNet40":
            return ModelNet40Data(
                train=True,
                num_points=cfg.loader.train.modelnet40.num_points,
                randomize_data=cfg.loader.train.modelnet40.randomize_data,
                unseen=cfg.loader.train.modelnet40.unseen,
                use_normals=cfg.loader.train.modelnet40.use_normals,
            )
    elif phase == "val":
        if cfg.loader.valid.dataset_name == "ThreeDMatch":
            return RegistrationData3DM(
                num_points=cfg.loader.valid.three_d_match.num_points,
                noise=cfg.loader.valid.three_d_match.noise,
                outliers=cfg.loader.valid.three_d_match.outliers,
                rotMax=pidict(cfg.loader.valid.three_d_match.rotMax),
                rate=cfg.loader.valid.three_d_match.rate,
                randomize=cfg.loader.valid.three_d_match.randomize,
                phase="val"
            )
        elif cfg.loader.valid.dataset_name == "ModelNet40":
            return ModelNet40Data(
                train=False,
                num_points=cfg.loader.valid.modelnet40.num_points,
                randomize_data=cfg.loader.valid.modelnet40.randomize_data,
                unseen=cfg.loader.valid.modelnet40.unseen,
                use_normals=cfg.loader.valid.modelnet40.use_normals,
            )
    else:
        if cfg.loader.train.dataset_name == "ThreeDMatch":
            return RegistrationData3DM(
                num_points=cfg.loader.train.three_d_match.num_points,
                noise=cfg.loader.train.three_d_match.noise,
                outliers=cfg.loader.train.three_d_match.outliers,
                rotMax=pidict(cfg.loader.train.three_d_match.rotMax),
                rate=cfg.loader.train.three_d_match.rate,
                randomize=cfg.loader.train.three_d_match.randomize,
                phase="test"
            )
        elif cfg.loader.train.dataset_name == "ModelNet40":
            return ModelNet40Data(
                train=False,
                num_points=cfg.loader.train.modelnet40.num_points,
                randomize_data=cfg.loader.train.modelnet40.randomize_data,
                unseen=cfg.loader.train.modelnet40.unseen,
                use_normals=cfg.loader.train.modelnet40.use_normals,
            )


def get_dataloader(
    cfg: TrainConfig,
    phase: str = "train",
) -> DataLoader:
    dataset = select_datasets(cfg=cfg, phase=phase)

    if phase == "train":
        batch_size = cfg.loader.train.batch_size
        shuffle = cfg.loader.train.shuffle
        drop_last = cfg.loader.train.drop_last
        num_workers = cfg.loader.train.num_workers
    elif phase == "val":
        batch_size = cfg.loader.valid.batch_size
        shuffle = cfg.loader.valid.shuffle
        drop_last = cfg.loader.valid.drop_last
        num_workers = cfg.loader.valid.num_workers
    else:
        batch_size = cfg.loader.train.batch_size
        shuffle = cfg.loader.train.shuffle
        drop_last = cfg.loader.train.drop_last
        num_workers = cfg.loader.train.num_workers

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return loader