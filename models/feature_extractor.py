import torch
from torch import nn

from pooling import Pooling
from attention import (
    Self_Attn,
    Cross_Attn,
    Scaled_Dot_Product_Self_Attention,
    Scaled_Dot_Product_Cross_Attention,
    FeedForwadNetwork,
)
from convolution import BasicConv1D

class PPF_Extraction(torch.nn.Module):
    def __init__(
        self,
        emb_dims=192,
        input_shape="bnc",
        use_bn=False,
        global_feat=True,
        input_dim=3,
    ):
        super(PPF_Extraction, self).__init__()
        if input_shape not in ["bcn", "bnc"]:
            raise ValueError(
                "Allowed shapes are 'bcn' (batch * channels * num_in_points), 'bnc' "
            )
        self.input_shape = input_shape
        self.emb_dims = emb_dims
        self.use_bn = use_bn
        self.global_feat = global_feat
        self.input_dim = input_dim
        if not self.global_feat:
            self.pooling = Pooling("max")

        self.layers = self.create_structure()

    def create_structure(self):
        self.conv1 = Self_Attn(self.input_dim, 64)  # torch.nn.Conv1d(3, 64, 1)
        self.conv2 = Self_Attn(64, 64)  # torch.nn.Conv1d(64, 64, 1)
        self.conv3 = Self_Attn(64, 64)  # torch.nn.Conv1d(64, 64, 1)
        self.conv4 = Self_Attn(64, 128)  # torch.nn.Conv1d(64, 128, 1)
        self.conv5 = Self_Attn(128, self.emb_dims)

    def forward(self, input_data):
        # input_data: 		Point Cloud having shape input_shape.
        # output:			PointNet features (Batch x emb_dims)
        if self.input_shape == "bnc":
            num_points = input_data.shape[1]
            input_data = input_data.permute(0, 2, 1)
        else:
            num_points = input_data.shape[2]
        if input_data.shape[1] != 3:
            raise RuntimeError("shape of x must be of [Batch x 3 x NumInPoints]")

        output = input_data

        x1 = self.conv1(output)  # 64
        x2 = self.conv2(x1)  # 64
        x3 = self.conv3(x2)  # 64
        x4 = self.conv4(x3)  # 128
        x5 = self.conv5(x4)  # 192

        output = torch.cat([x1, x2, x3, x4, x5], dim=1)  # 256, x4 x0,
        point_feature = output

        if self.global_feat:
            return output
        else:
            output = self.pooling(output)
            output = output.view(-1, self.emb_dims, 1).repeat(1, 1, num_points)
            return torch.cat([output, point_feature], 1)

class SharedGlobalFeatureExtraction(nn.Module):
    def __init__(
        self,
        source_feature_size=1024,
        target_feature_size=1024,
        feature_model=PPF_Extraction(),
        last_layer_sigmoid=True,
    ):
        super().__init__()
        self.feature_model = feature_model
        self.pooling_max = Pooling(pool_type="max")
        self.pooling_avg = Pooling(pool_type="avg")

        self.global_feat_1 = Cross_Attn(1024, 512)
        self.global_feat_2 = Cross_Attn(512, 256)
        self.global_feat_3 = Cross_Attn(256, 512)

        if last_layer_sigmoid:
            self.mask_estimator = nn.Sequential(
                BasicConv1D(1024, 512),
                BasicConv1D(512, 256),
                BasicConv1D(256, 128),
                nn.Conv1d(128, 1, 1),
                nn.Sigmoid(),
            )
        else:
            self.mask_estimator = nn.Sequential(
                BasicConv1D(1024, 512),
                BasicConv1D(512, 256),
                BasicConv1D(256, 128),
                nn.Conv1d(128, 1, 1),
            )

    def find_mask(self, source_features, target_features):
        global_source_features_max = self.pooling_max(source_features)
        global_target_features_max = self.pooling_max(target_features)

        global_source_features_avg = self.pooling_avg(source_features)
        global_target_features_avg = self.pooling_avg(target_features)

        global_source_features = torch.cat(
            [global_source_features_max, global_source_features_avg], dim=1
        )
        global_target_features = torch.cat(
            [global_target_features_max, global_target_features_avg], dim=1
        )

        shared_feat_source, shared_feat_target = self.global_feat_1(
            global_source_features.unsqueeze(2), global_target_features.unsqueeze(2)
        )
        shared_feat_source, shared_feat_target = self.global_feat_2(shared_feat_source, shared_feat_target)
        shared_feat_source, shared_feat_target = self.global_feat_3(shared_feat_source, shared_feat_target)

        batch_size, _, num_points = source_features.size()
        global_target_features = shared_feat_target  # .unsqueeze(2)
        global_target_features = global_target_features.repeat(1, 1, num_points)
        source_xfeatures = torch.cat([source_features, global_target_features], dim=1)
        source_mask = self.mask_estimator(source_xfeatures)

        batch_size, _, num_points = target_features.shape
        global_source_features = shared_feat_source  # .unsqueeze(2)
        global_source_features = global_source_features.repeat(1, 1, num_points)
        target_xfeatures = torch.cat([target_features, global_source_features], dim=1)
        target_mask = self.mask_estimator(target_xfeatures)

        return source_mask.view(batch_size, -1), target_mask.view(batch_size, -1)

    def forward(self, source, target):
        target_features = self.feature_model(target)  # [B x C x N]
        source_features = self.feature_model(source)  # [B x C x N]

        print(target_features.shape)

        src_mask, tgt_mask = self.find_mask(source_features=source_features, target_features=target_features)
        return src_mask, tgt_mask


class FeatureExtractor_SACA(nn.Module):
    def __init__(
        self,
        input_features: int = 1024,
        output_features: int = 1024,
    ):
        super(FeatureExtractor_SACA, self).__init__()
        self.constructor(
            input_features=input_features,
            output_features=output_features,
        )

    def constructor(self, input_features=1024, output_features=1024):
        self.sa1 = Scaled_Dot_Product_Self_Attention(input_features, output_features)
        # self.ffn1_s = nn.Linear(input_features, output_features)
        # self.norm1_s = nn.LayerNorm(output_features)
        self.ca1 = Scaled_Dot_Product_Cross_Attention(input_features, output_features)
        # self.ffn1_c = nn.Linear(input_features, output_features)
        # self.norm1_c = nn.LayerNorm(output_features)

        self.sa2 = Scaled_Dot_Product_Self_Attention(input_features, output_features)
        # self.ffn2_s = nn.Linear(input_features, output_features)
        # self.norm2_s = nn.LayerNorm(output_features)
        self.ca2 = Scaled_Dot_Product_Cross_Attention(input_features, output_features)
        # self.ffn2_c = nn.Linear(input_features, output_features)
        # self.norm2_c = nn.LayerNorm(output_features)

        self.sa3 = Scaled_Dot_Product_Self_Attention(input_features, output_features)
        # self.ffn3_s = nn.Linear(input_features, output_features)
        # self.norm3_s = nn.LayerNorm(output_features)
        self.ca3 = Scaled_Dot_Product_Cross_Attention(input_features, output_features)
        # self.ffn3_c = nn.Linear(input_features, output_features)
        # self.norm3_c = nn.LayerNorm(output_features)


        self.ffn = nn.Linear(output_features, output_features)
        self.norm = nn.LayerNorm(output_features)

    def forward(self, x, y):
        x_l1 = self.sa1(x)
        # x_l1 = self.ffn1_s(x_l1)
        # x_l1 = self.norm1_s(x_l1+x)
        x_l1 = self.ffn(x_l1)
        x_l1 = self.norm(x_l1+x)


        y_l1 = self.sa1(y)
        # y_l1 = self.ffn1_s(y_l1)
        # y_l1 = self.norm1_s(y_l1+y)
        y_l1 = self.ffn(y_l1)
        y_l1 = self.norm(y_l1+y)

        x_out, y_out = self.ca1(x_l1, y_l1)

        x_l2 = self.sa1(x_out)
        # x_l2 = self.ffn1_s(x_l2)
        # x_l2 = self.norm1_s(x_l2+x)
        x_l2 = self.ffn(x_l2)
        x_l2 = self.norm(x_l2+x)

        y_l2 = self.sa2(y_out)
        # y_l2 = self.ffn1_s(y_l2)
        # y_l2 = self.norm1_s(y_l2+y)
        y_l2 = self.ffn(y_l2)
        y_l2 = self.norm(y_l2+y)

        x_out, y_out = self.ca2(x_l2, y_l2)

        x_l3 = self.sa3(x_out)
        # x_l3 = self.ffn1_s(x_l3)
        # x_l3 = self.norm1_s(x_l3+x)
        x_l3 = self.ffn(x_l3)
        x_l3 = self.norm(x_l3+x)

        y_l3 = self.sa3(y_out)
        y_l3 = self.ffn(y_l3)
        y_l3 = self.norm(y_l3+y)

        x_out, y_out = self.ca3(x_l3, y_l3)

        return x_out, y_out



if __name__ == '__main__':
    model = FeatureExtractor_SACA(
        input_features=1024,
        output_features=1024
    ).to("cuda")

    from torchinfo import summary
    summary(model,[(1,2048,1024),(1, 2048,1024)])

    # model = SharedGlobalFeatureExtraction().to("cuda")
    # from torchinfo import summary
    # summary(model,[(1,2048,3),(1, 2048,3)])
