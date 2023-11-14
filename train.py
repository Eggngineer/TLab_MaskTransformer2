import hydra
import wandb
import open3d as o3d
import numpy as np

from omegaconf import OmegaConf
from config.train_config import TrainConfig
from lib.train_utils import train

@hydra.main(config_name="train_config", version_base=None, config_path="config/")
def main(cfg: TrainConfig):
    wandb.config = OmegaConf.to_container(
        cfg=cfg,
        resolve=True,
        throw_on_missing=True,
    )
    wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        name=cfg.wandb.experiment,
    )
    train(cfg=cfg)
    pcd = o3d.io.read_point_cloud("/home/yamazaki/TLab_MaskTransformer/results/train_MaskTransformerBoarderSigmoidless_3dmatch_method-param_mean2_k-9_weight-1.0_seed-261_20231027221142/test_3dmatch_20231101132638/ply_files/7-scenes-redkitchen_1/src_gt_boardered.ply")
    wandb.log(
        {
            "hoge": 1,
            "pcd": wandb.Object3D(np.concatenate([np.array(pcd.points),np.array(pcd.colors)*255],axis=-1))
        }
    )


if __name__ == "__main__":
    main()
    wandb.finish()
