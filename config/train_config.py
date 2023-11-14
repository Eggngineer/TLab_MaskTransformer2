from dataclasses import dataclass
from numpy import pi

@dataclass
class ThreeDMatchConfig:
    num_points: int = 2048
    noise: bool = False
    outliers: bool = False
    # rotMax in {0, pi/6, pi/4, pi/3, pi/2, pi, 2pi}
    rotMax: str = '0'
    rate: float = 1.0
    randomize: bool = False
    unseen: bool = True


@dataclass
class ModelNet40Config:
    num_points: int = 2048
    randomize_data: bool = True
    unseen: bool = True
    use_normals: False


@dataclass
class TrainLoader:
    dataset_name: str = "ThreeDMatch"
    three_d_match: ThreeDMatchConfig = ThreeDMatchConfig()
    modelnet40: ModelNet40Config = ModelNet40Config()
    batch_size: int = 16
    num_workers: int = 16
    shuffle: bool = True
    drop_last: bool = True


@dataclass
class ValidationLoader:
    dataset_name: str = "ThreeDMatch"
    three_d_match: ThreeDMatchConfig = ThreeDMatchConfig()
    batch_size: int = 16
    num_workers: int = 16
    shuffle: bool = True
    drop_last: bool = True


@dataclass
class Loader:
    train: TrainLoader = TrainLoader()
    valid: ValidationLoader = ValidationLoader()


@dataclass
class Train:
    epoch: int = 5000
    optimizer: str = "Adam"
    weight_decay: float = 1.0e-2
    lr: float = 1.0e-4
    positive_weight: float = 0.309
    loss: str = "BCE"
    late_start: int = 200


@dataclass
class Validation:
    per_epochs: int = 5


@dataclass
class WandB:
    entity: str = "eggng1neer"
    project: str = "test"
    experiment: str = "point cloud test"


@dataclass
class Model:
    name: str = "MaskTransformer_ver2"
    boundary: bool = True
    last_layer_sigmoid: bool = True
    late_start: int = 0


@dataclass
class Mask:
    th: float = 0.5


@dataclass
class Boundary:
    method: str = "param_mean"
    alpha: float = 4.0
    beta: float = 4.0
    gamma: float = 4.0
    mu: float = 4.0
    th: float = 0.5
    nearest_k: float = 9


@dataclass
class TrainConfig:
    train: Train = Train()
    valid: Validation = Validation()
    loader: Loader = Loader()
    wandb: WandB = WandB()
    model: Model = Model()
    mask: Mask = Mask()
    boundary: Boundary = Boundary()