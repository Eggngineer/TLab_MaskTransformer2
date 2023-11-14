import torch

from torch import nn
from feature_extractor import (
    PPF_Extraction,
    SharedGlobalFeatureExtraction,
    FeatureExtractor_SACA,
)
from pooling import Pooling

class MaskTransformerBoarder(nn.Module):
    def __init__(
            self,
            feature_model:PPF_Extraction = PPF_Extraction(use_bn=True),
            last_layer_sigmoid: bool = True,
    ):
        super().__init__()
        self.maskNet = SharedGlobalFeatureExtraction(
            feature_model=feature_model,
            last_layer_sigmoid=last_layer_sigmoid
        )

    @staticmethod
    def index_points(points, idx):
        """
        Input:
                points: input points data, [B, N, C]
                idx: sample index data, [B, S]
        Return:
                new_points:, indexed points data, [B, S, C]
        """
        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = (
            torch.arange(B, dtype=torch.long)
            .to(device)
            .view(view_shape)
            .repeat(repeat_shape)
        )
        new_points = points[batch_indices, idx, :]

        return new_points

    @staticmethod
    def find_index(mask_val):
        mask_idx = torch.nonzero((mask_val[0] > 0.5) * 1.0)
        return mask_idx.view(1, -1)

    def forward(self, x, y):
        mask_x, mask_y = self.maskNet(x, y)  # B, N

        return mask_x, mask_y


class MaskTransformer_ver2(nn.Module):
    def __init__(
            self,
            last_layer_sigmoid: bool = True,
    ):
        super(MaskTransformer_ver2, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(3,32),
            # nn.BatchNorm1d(32),
            nn.Linear(32,128),
            # nn.BatchNorm1d(128),
            nn.Linear(128,512),
            # nn.BatchNorm1d(512),
        )
        self.attention = FeatureExtractor_SACA(
            input_features=1024,
            output_features=1024,
        )

        self.tail = nn.Sequential(
            nn.Linear(1024,256),
            # nn.BatchNorm1d(256),
            nn.Linear(256,64),
            # nn.BatchNorm1d(64),
            nn.Linear(64,16),
            # nn.BatchNorm1d(16),
            nn.Linear(16,1),
            nn.Softmax(dim=-1),
        )

        self.pool = Pooling(pool_type="max", dim=1, keepdim=True)

    def forward(self, src, tgt):
        localf_src = self.head(src)
        localf_tgt = self.head(tgt)

        globalf_src = self.pool(localf_src)
        globalf_tgt = self.pool(localf_tgt)

        f_src = torch.cat((localf_src, globalf_src.repeat(1, localf_src.shape[1], 1)),dim=-1)
        f_tgt = torch.cat((localf_tgt, globalf_tgt.repeat(1, localf_tgt.shape[1], 1)),dim=-1)


        f2_src, f2_tgt = self.attention(f_src, f_tgt)

        mask_x = self.tail(f2_src)
        mask_y = self.tail(f2_tgt)

        return mask_x, mask_y

if __name__ == '__main__':
    from torchinfo import summary
    model = MaskTransformer_ver2()
    summary(model, [(1,2048,3),(1,2048,3)])