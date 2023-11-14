import torch
import h5py
import os
import numpy as np

from pathlib import Path

from torch.utils.data import Dataset

def download_modelnet40():
    data_dir = Path("./dataset")
    data_path = data_dir / "modelnet40_ply_hdf5_2048"
    if not data_path.exists():
        www = Path(
            "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"
        )
        zip_name = www.name
        save_name = www.stem
        ops = "--no-check-certificate"
        os.system(f"wget {www} {ops}; unzip {zip_name}")
        os.system(f"mv {save_name} {str(data_path)}")
        os.system(f"rm {zip_name}")


def load_modelnet(
    is_train: bool,
    use_normals: bool = False,
):
    if is_train:
        partition = "train"
    else:
        partition = "test"

    base_dir = Path("./dataset")
    if not (base_dir / "modelnet40_ply_hdf5_2048").exists():
        download_modelnet40()
    all_data = []
    all_label = []
    for h5_name in sorted(
        (base_dir / "modelnet40_ply_hdf5_2048").glob(f"ply_data_{partition}*.h5")
    ):
        f = h5py.File(h5_name)
        if use_normals:
            data = np.concatenate([f["data"][:], f["normal"][:]], axis=-1).astype(
                "float32"
            )
        else:
            data = f["data"][:].astype("float32")
        label = f["label"][:].astype("int64")
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def add_outliers(pointcloud, gt_mask):
    N, C = pointcloud.shape
    outliers = 2 * torch.rand(100, C) - 1
    pointcloud = torch.cat([pointcloud, outliers], dim=0)
    gt_mask = torch.cat([gt_mask, torch.zeros(100)])

    idx = torch.randperm(pointcloud.shape[0])
    pointcloud, gt_mask = pointcloud[idx], gt_mask[idx]

    return pointcloud, gt_mask


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.5):
    pointcloud += (
        torch.empty(pointcloud.shape).normal_(mean=0, std=sigma).clamp(-clip, clip)
    )
    return pointcloud


class ModelNet40Data(Dataset):
    def __init__(
        self,
        train=True,
        num_points=1024,
        randomize_data=False,
        unseen=False,
        use_normals=False,
    ):
        super(ModelNet40Data, self).__init__()

        self.data, self.labels = load_modelnet(train, use_normals)
        self.num_points = num_points
        self.randomize_data = randomize_data
        self.unseen = unseen
        if self.unseen:
            self.labels = self.labels.reshape(-1)
            if not train:
                self.data = self.data[self.labels >= 20]
                self.labels = self.labels[self.labels >= 20]
            if train:
                self.data = self.data[self.labels < 20]
                self.labels = self.labels[self.labels < 20]
                print(
                    "Successfully loaded first 20 categories for training and last 20 for testing!"
                )
            self.labels = self.labels.reshape(-1, 1)  # [N,]   -> [N, 1]

    def __getitem__(self, idx):
        if self.randomize_data:
            current_points = self.randomize(idx)
        else:
            current_points = self.data[idx].copy()

        current_points = torch.from_numpy(current_points[: self.num_points, :]).float()
        label = torch.from_numpy(self.labels[idx]).type(torch.LongTensor)

        return current_points, label

    def __len__(self):
        return self.data.shape[0]


class RegistrationData3DM(Dataset):
    def __init__(
        self,
        num_points=2048,
        noise=False,
        outliers=False,
        rotMax=np.pi,
        rate=1,
        randomize=False,
        phase="train",
    ):
        super(RegistrationData3DM, self).__init__()
        self.data = []
        self.noise = noise
        self.outliers = outliers
        self.num_points = num_points
        self.rotMax = rotMax  # default: pi
        self.rate = rate
        self.randomize = randomize
        self.phase = phase
        self._init()

        from .ops.transform_functions import PNLKTransform

        self.PNLKTransform = PNLKTransform(0)

        self.renew_transforms()

    def _init(self):
        if self.phase == "train":
            self.data = np.load(
                "./dataset/3dmatch/train_0_0375.npy", allow_pickle=True
            ).T

        elif self.phase == "val":
            self.data = np.load("./dataset/3dmatch/val_0_0375.npy", allow_pickle=True).T
        else:
            self.data = np.load("./dataset/3dmatch/test_0_0375.npy", allow_pickle=True)

    def __len__(self):
        return len(self.data)

    def renew_transforms(self):
        self.transforms = self.PNLKTransform(self.rotMax * self.rate, True)

    def __getitem__(self, index):
        data = self.data[index]
        source, target, gt_xmask, gt_ymask, cls = data

        l_source = []
        l_target = []
        l_gt_xmask = []
        l_gt_ymask = []

        for tmp, src, xmask, ymask in zip(source, target, gt_xmask, gt_ymask):
            tmp_idx = np.random.randint(0, tmp.shape[0], self.num_points)
            src_idx = np.random.randint(0, src.shape[0], self.num_points)
            l_source.append(tmp[tmp_idx])
            l_target.append(src[src_idx])
            l_gt_xmask.append(xmask[tmp_idx])
            l_gt_ymask.append(ymask[src_idx])

        source = torch.from_numpy(np.asarray(l_source))[0]
        target = torch.from_numpy(np.asarray(l_target))[0]
        gt_xmask = torch.from_numpy(np.asarray(l_gt_xmask))[0]
        gt_ymask = torch.from_numpy(np.asarray(l_gt_ymask))[0]
        cls = [cls[0]]

        target, rot = self.transforms(target)

        if self.outliers:
            source, gt_xmask = add_outliers(source, gt_xmask)
        if self.outliers:
            target, gt_ymask = add_outliers(target, gt_ymask)
        if self.randomize:
            rand_id = torch.randperm(target.shape[0])
            target = target[rand_id, :]
            gt_ymask = gt_ymask[rand_id]

        if self.noise:
            target = jitter_pointcloud(target)
        if self.outliers:
            source, gt_xmask = add_outliers(source, gt_xmask)
        igt = self.transforms.igt

        return (
            source.float(),
            target.float(),
            igt.float(),
            gt_xmask.float(),
            gt_ymask.float(),
            cls,
            rot.float(),
        )
