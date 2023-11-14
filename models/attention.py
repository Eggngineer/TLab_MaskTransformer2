from torch import nn

from activation import Mish
from convolution import BasicConv1D

import torch


class Self_Attn(nn.Module):
    """Self Attention

    Args:
        in_dim(int): dimension of input tensor
        out_dim(int): dimension of output tensor
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        share: bool = False,
        layer_type: str = "BasicConv1D",
    ):
        super(Self_Attn, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.share = share
        self.layer_type = layer_type

        if self.share:
            self.query_conv = layer(
                layer_type=layer_type,
                input_dim=in_dim,
                output_dim=out_dim,
            )
        else:
            self.query_conv = layer(
                layer_type=layer_type,
                input_dim=in_dim,
                output_dim=out_dim,
            )
            self.key_conv = layer(
                layer_type=layer_type,
                input_dim=in_dim,
                output_dim=out_dim,
            )
            self.value_conv = layer(
                layer_type=layer_type,
                input_dim=in_dim,
                output_dim=out_dim,
            )

        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
                x : input feature maps( B X C X N)  32, 1024, 64
        returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """

        proj_query = self.query_conv(x)  # B, in_dim, N   ---> B, in_dim // 8, N
        if self.share:
            proj_key = proj_query.permute(0, 2, 1)

            energy = torch.bmm(proj_query, proj_key)  # transpose check    B, N, N
            attention = self.softmax(energy)  # B , N,  N

            out_x = torch.bmm(proj_key, attention.permute(0, 2, 1))  # B, out_dim, N
            out = self.beta * out_x + proj_key

        else:
            proj_key = self.key_conv(x).permute(0, 2, 1)
            proj_value = self.value_conv(x)

            energy = torch.bmm(proj_query, proj_key)
            attention = self.softmax(energy)

            out_x = torch.bmm(attention, proj_value)
            out = self.beta * out_x + proj_value

        return out


class Cross_Attn(nn.Module):
    """Cross Attention

    Args:
        in_dim (int): dimension of input tensor
        out_dim (int): dimension of output tensor
    """

    def __init__(self, in_dim, out_dim, layer_type: str = "BasicConv1D"):
        super(Cross_Attn, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layer_type = layer_type

        self.query_conv = layer(
            layer_type=layer_type,
            input_dim=in_dim,
            output_dim=out_dim,
        )
        self.key_conv = layer(
            layer_type=layer_type,
            input_dim=in_dim,
            output_dim=out_dim,
        )
        self.value_conv = layer(
            layer_type=layer_type,
            input_dim=in_dim,
            output_dim=out_dim,
        )

        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x, y):  # B, 1024 , 1
        """
        inputs :
            x : input feature maps( B X C,1 )
        returns :
            out : self attention value + input feature
            attention: B X N X N (N is Width*Height)
        """
        proj_query_x = self.query_conv(x)  # [B, in_dim, 1] -> [B, out_dim, 1]

        proj_key_y = self.key_conv(y).permute(0, 2, 1)  # [B, 1, out_dim]

        energy_xy = torch.bmm(proj_query_x, proj_key_y)  # [B, out_dim, out_dim]

        attention_xy = self.softmax(energy_xy)
        attention_yx = self.softmax(energy_xy.permute(0, 2, 1))

        proj_value_x = self.value_conv(x)  # [B, out_dim, 64]
        proj_value_y = self.value_conv(y)  # [B, out_dim, 64]

        out_x = torch.bmm(attention_xy, proj_value_x)  # [B, out_dim]
        out_x = self.beta * out_x + proj_value_x

        out_y = torch.bmm(attention_yx, proj_value_y)  # [B, out_dim]
        out_x = self.beta * out_x + proj_value_x

        return out_x, out_y


class Scaled_Dot_Product_Self_Attention(nn.Module):
    """Scaled Dot-Product Self Attention

    Args:
        in_dim(int): dimension of input tensor
        out_dim(int): dimension of output tensor
    """

    def __init__(
        self,
        in_dim,
        out_dim,
    ):
        super(Scaled_Dot_Product_Self_Attention, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.q_conv = nn.Linear(
            in_features=in_dim,
            out_features=out_dim,
            bias=False,
        )
        self.k_conv = nn.Linear(
            in_features=in_dim,
            out_features=out_dim,
            bias=False,
        )
        self.v_conv = nn.Linear(
            in_features=in_dim,
            out_features=out_dim,
            bias=False,
        )
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x):
        """
        inputs :
                x: input feature maps ( B x N x C )
        returns :
                out : self attention value + input feature
        """

        q = self.q_conv(x)
        k = self.k_conv(x)
        v = self.v_conv(x)

        d_k = torch.tensor(k.shape[-1])

        energy = torch.bmm(q,k.permute((0,2,1)))
        scaling = torch.sqrt(d_k)
        attention = self.softmax(energy / scaling)

        out = torch.bmm(attention,v)

        return out


class Scaled_Dot_Product_Cross_Attention(nn.Module):
    """Scaled Dot-Product Cross Attention

    Args:
        in_dim(int): dimension of input tensor
        out_dim(int): dimension of output tensor
    """

    def __init__(
        self,
        in_dim,
        out_dim,
    ):
        super(Scaled_Dot_Product_Cross_Attention, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # self.qx_conv = nn.Linear(
        #     in_features=in_dim,
        #     out_features=out_dim,
        #     bias=False,
        # )
        # self.kx_conv = nn.Linear(
        #     in_features=in_dim,
        #     out_features=out_dim,
        #     bias=False,
        # )
        # self.vx_conv = nn.Linear(
        #     in_features=in_dim,
        #     out_features=out_dim,
        #     bias=False,
        # )

        # self.qy_conv = nn.Linear(
        #     in_features=in_dim,
        #     out_features=out_dim,
        #     bias=False,
        # )
        # self.ky_conv = nn.Linear(
        #     in_features=in_dim,
        #     out_features=out_dim,
        #     bias=False,
        # )
        # self.vy_conv = nn.Linear(
        #     in_features=in_dim,
        #     out_features=out_dim,
        #     bias=False,
        # )

        self.q_conv = nn.Linear(
            in_features=in_dim,
            out_features=out_dim,
            bias=False,
        )
        self.k_conv = nn.Linear(
            in_features=in_dim,
            out_features=out_dim,
            bias=False,
        )
        self.v_conv = nn.Linear(
            in_features=in_dim,
            out_features=out_dim,
            bias=False,
        )

        self.softmax = nn.Softmax(dim=-1)
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        """
        inputs :
                x: input feature maps ( B x N x C )
                y: input feature maps ( B x N x C )
        returns :
                out : self attention value + input feature
        """

        # qx = self.q_conv(x)
        # kx = self.k_conv(x)
        # vx = self.v_conv(x)

        # qy = self.q_conv(y)
        # ky = self.k_conv(y)
        # vy = self.v_conv(y)

        # d_kx = torch.tensor(kx.shape[-1])
        # d_ky = torch.tensor(ky.shape[-1])

        # energy_x = torch.bmm(qy,kx.permute((0,2,1)))
        # scaling_x = torch.sqrt(d_kx)
        # attention_x = self.softmax(energy_x / scaling_x)

        # energy_y = torch.bmm(qx,ky.permute((0,2,1)))
        # scaling_y = torch.sqrt(d_ky)
        # attention_y = self.softmax(energy_y / scaling_y)

        # out_x = torch.bmm(attention_x,vx)
        # out_y = torch.bmm(attention_y,vy)

        proj_query_x = self.q_conv(x)  # [B, in_dim, 1] -> [B, out_dim, 1]

        proj_key_y = self.k_conv(y).permute(0, 2, 1)  # [B, 1, out_dim]

        energy_xy = torch.bmm(proj_query_x, proj_key_y)  # [B, out_dim, out_dim]

        attention_xy = self.softmax(energy_xy)
        attention_yx = self.softmax(energy_xy.permute(0, 2, 1))

        proj_value_x = self.v_conv(x)  # [B, out_dim, 64]
        proj_value_y = self.v_conv(y)  # [B, out_dim, 64]

        out_x = torch.bmm(attention_xy, proj_value_x)  # [B, out_dim]
        out_x = self.beta * out_x + proj_value_x
        out_y = torch.bmm(attention_yx, proj_value_y)  # [B, out_dim]
        out_x = self.beta * out_x + proj_value_x

        return out_x, out_y


class FeedForwadNetwork(nn.Module):
    def __init__(
        self,
        input_dimension: int = 100,
        output_dimension: int = 100,
    ) -> None:
        super(FeedForwadNetwork, self).__init__()
        self.linear = torch.nn.Linear(
            in_features=input_dimension,
            out_features=output_dimension,
            bias=True,
        )
        self.relu = torch.nn.ReLU()

    def forward(self,x):
        out_x = self.linear(x)
        out_x = self.relu(out_x)
        out_x = self.linear(out_x)

        return out_x


def layer(
    layer_type: str,
    input_dim: int,
    output_dim: int,
):
    """layer

    Args:
        layer_type (str): {"BasicConv1D", "Linear"}
        input_dim (int): input dimension of the layer
        output_dim (int): output dimension of the layer

    Returns:
        layer function(any): selective layer function
    """
    if layer_type == "BasicConv1D":
        return BasicConv1D(
            in_channels=input_dim,
            out_channels=output_dim,
        )
    elif layer_type == "Linear":
        return torch.nn.Linear(
            in_features=input_dim,
            out_features=input_dim,
        )
    else:
        assert f"layer type '{layer_type}' is not found "


if __name__ == "__main__":
    # cross_att = Cross_Attn(in_dim=10, out_dim=4).to("cuda")
    # self_att = Self_Attn(in_dim=10, out_dim=4).to("cuda")

    from torchinfo import summary
    # summary(cross_att, [(10, 3), (10, 3)])
    # summary(self_att, (10, 3))

    # module = Scaled_Dot_Product_Self_Attention(in_dim=1024, out_dim=1024).to("cuda")
    # summary(module, [(1,2048,1024)])
    module = Scaled_Dot_Product_Cross_Attention(in_dim=1024, out_dim=1024).to("cuda")
    summary(module, [(1,2048, 1024), (1, 2048, 1024)])

    # module = Cross_Attn(in_dim=1024, out_dim=1024).to("cuda")
    # summary(module, [(1024,1024),(1024,1024)])
    pass