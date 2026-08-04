# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# @Author  : Shuai Yuan
# @File    : SCTransNet.py
# @Software: PyCharm
# coding=utf-8###轻量化 32通道###修改了一下uiu和uiu_trans，最后输出的那个纹理模块模块，之前是单通道和4通道广播后相加，现在变成了先把通道统一，再相加；
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import math
from torch.nn import Dropout, Softmax, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
import torch.nn as nn
import torch
import torch.nn.functional as F
import ml_collections
from einops import rearrange
import numbers
from thop import profile

def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 480  # KV_size = Q1 + Q2 + Q3 + Q4
    config.transformer.num_heads = 4
    config.transformer.num_layers = 4
    config.patch_sizes = [16, 8, 4, 2]
    config.base_channel = 32  # base channel of U-Net
    config.n_classes = 1

    # ********** useless **********
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    return config


class Channel_Embeddings(nn.Module):
    def __init__(self, config, patchsize, img_size, in_channels):
        super().__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patchsize)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])  # 14 * 14 = 196

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=in_channels,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_channels))
        self.dropout = Dropout(config.transformer["embeddings_dropout_rate"])

    def forward(self, x):
        if x is None:
            return None
        x = self.patch_embeddings(x)
        return x


class Reconstruct(nn.Module):##就是进行了上采样加cbl模块
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Reconstruct, self).__init__()
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    # def forward(self, x, h, w):
    def forward(self, x):
        if x is None:
            return None

        x = nn.Upsample(scale_factor=self.scale_factor, mode='bilinear')(x)

        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


# spatial-embedded Single-head Channel-cross Attention (SSCA)
class Attention_org_out(nn.Module):###这一块对应JX之后1*1卷积开始到输出相加的那个p1 ，2，3，4之前的，还没add
    def __init__(self, channel_num):
        super(Attention_org_out, self).__init__()
        self.KV_size = channel_num
        self.channel_num = channel_num
        self.num_attention_heads = 1
        self.psi = nn.InstanceNorm2d(self.num_attention_heads)
        self.softmax = Softmax(dim=3)

        # self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.mhead1 = nn.Conv2d(channel_num, channel_num * self.num_attention_heads, kernel_size=1, bias=False)
        self.mheadk = nn.Conv2d(self.KV_size, self.KV_size * self.num_attention_heads, kernel_size=1, bias=False)
        self.mheadv = nn.Conv2d(self.KV_size, self.KV_size * self.num_attention_heads, kernel_size=1, bias=False)

        self.q1 = nn.Conv2d(channel_num * self.num_attention_heads, channel_num * self.num_attention_heads, kernel_size=3, stride=1,
                            padding=1,
                            groups=channel_num * self.num_attention_heads, bias=False)
        self.k = nn.Conv2d(self.KV_size * self.num_attention_heads, self.KV_size * self.num_attention_heads, kernel_size=3, stride=1,
                           padding=1, groups=self.KV_size * self.num_attention_heads, bias=False)
        self.v = nn.Conv2d(self.KV_size * self.num_attention_heads, self.KV_size * self.num_attention_heads, kernel_size=3, stride=1,
                           padding=1, groups=self.KV_size * self.num_attention_heads, bias=False)

        self.project_out1 = nn.Conv2d(channel_num, channel_num, kernel_size=1, bias=False)


    def forward(self, emb1):
        b, c, h, w = emb1.shape
        q1 = self.q1(self.mhead1(emb1))
        k = self.k(self.mheadk(emb1))
        v = self.v(self.mheadv(emb1))
        # k, v = kv.chunk(2, dim=1)

        q1 = rearrange(q1, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)

        q1 = torch.nn.functional.normalize(q1, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, c1, _ = q1.shape
        _, _, c, _ = k.shape

        attn1 = (q1 @ k.transpose(-2, -1)) / math.sqrt(self.KV_size)

        attention_probs1 = self.softmax(self.psi(attn1))

        out1 = (attention_probs1 @ v)
        out_1 = out1.mean(dim=1)

        out_1 = rearrange(out_1, 'b  c (h w) -> b c h w', h=h, w=w)

        O1 = self.project_out1(out_1)


        return O1
class Attention_org(nn.Module):###这一块对应JX之后1*1卷积开始到输出相加的那个p1 ，2，3，4之前的，还没add
    def __init__(self, channel_num):
        super(Attention_org, self).__init__()
        self.KV_size = channel_num
        self.channel_num = channel_num
        self.num_attention_heads = 1
        self.psi = nn.InstanceNorm2d(self.num_attention_heads)
        self.softmax = Softmax(dim=3)

        # self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.mhead1 = nn.Conv2d(channel_num, channel_num * self.num_attention_heads, kernel_size=1, bias=False)
        self.mheadk = nn.Conv2d(self.KV_size, self.KV_size * self.num_attention_heads, kernel_size=1, bias=False)
        self.mheadv = nn.Conv2d(self.KV_size, self.KV_size * self.num_attention_heads, kernel_size=1, bias=False)

        self.q1 = nn.Conv2d(channel_num * self.num_attention_heads, channel_num * self.num_attention_heads, kernel_size=3, stride=1,
                            padding=1,
                            groups=channel_num * self.num_attention_heads // 2, bias=False)
        self.k = nn.Conv2d(self.KV_size * self.num_attention_heads, self.KV_size * self.num_attention_heads, kernel_size=3, stride=1,
                           padding=1, groups=self.KV_size * self.num_attention_heads, bias=False)
        self.v = nn.Conv2d(self.KV_size * self.num_attention_heads, self.KV_size * self.num_attention_heads, kernel_size=3, stride=1,
                           padding=1, groups=self.KV_size * self.num_attention_heads, bias=False)

        self.project_out1 = nn.Conv2d(channel_num, channel_num, kernel_size=1, bias=False)


    def forward(self, emb1):
        b, c, h, w = emb1.shape
        q1 = self.q1(self.mhead1(emb1))
        k = self.k(self.mheadk(emb1))
        v = self.v(self.mheadv(emb1))
        # k, v = kv.chunk(2, dim=1)

        q1 = rearrange(q1, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)

        q1 = torch.nn.functional.normalize(q1, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, c1, _ = q1.shape
        _, _, c, _ = k.shape

        attn1 = (q1 @ k.transpose(-2, -1)) / math.sqrt(self.KV_size)

        attention_probs1 = self.softmax(self.psi(attn1))

        out1 = (attention_probs1 @ v)
        out_1 = out1.mean(dim=1)

        out_1 = rearrange(out_1, 'b  c (h w) -> b c h w', h=h, w=w)

        O1 = self.project_out1(out_1)


        return O1


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm3d(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm3d, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class eca_layer_2d(nn.Module):###对应图3b中的池化和33卷积
    def __init__(self, channel, k_size=3):
        super(eca_layer_2d, self).__init__()
        padding = k_size // 2
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=k_size, padding=padding, bias=False),
            nn.Sigmoid()
        )
        self.channel = channel
        self.k_size = k_size

    def forward(self, x):
        out = self.avg_pool(x)
        out = out.view(x.size(0), 1, x.size(1))
        out = self.conv(out)
        out = out.view(x.size(0), x.size(1), 1, 1)
        return out * x

# Complementary Feed-forward Network (CFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv3x3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features,
                                   bias=bias)
        self.dwconv5x5 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features,
                                   bias=bias)
        self.relu3 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.project_out = nn.Conv2d(hidden_features * 2, dim, kernel_size=1, bias=bias)
        self.eca = eca_layer_2d(dim)

    def forward(self, x):
        x_3,x_5 = self.project_in(x).chunk(2, dim=1)
        x1_3 = self.relu3(self.dwconv3x3(x_3))
        x1_5 = self.relu5(self.dwconv5x5(x_5))
        x = torch.cat([x1_3, x1_5], dim=1)
        x = self.project_out(x)
        x = self.eca(x)
        return x


#  Spatial-channel Cross Transformer Block (SCTB)
class Block_ViT(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Block_ViT, self).__init__()
        self.attn_norm1 = LayerNorm3d(channel_num[0], LayerNorm_type='WithBias')
        self.attn_norm2 = LayerNorm3d(channel_num[1], LayerNorm_type='WithBias')
        self.attn_norm3 = LayerNorm3d(channel_num[2], LayerNorm_type='WithBias')
        self.attn_norm4 = LayerNorm3d(channel_num[3], LayerNorm_type='WithBias')
        self.attn_norm = LayerNorm3d(config.KV_size, LayerNorm_type='WithBias')

        self.channel_attn = Attention_org(config, vis, channel_num)

        self.ffn_norm1 = LayerNorm3d(channel_num[0], LayerNorm_type='WithBias')
        self.ffn_norm2 = LayerNorm3d(channel_num[1], LayerNorm_type='WithBias')
        self.ffn_norm3 = LayerNorm3d(channel_num[2], LayerNorm_type='WithBias')
        self.ffn_norm4 = LayerNorm3d(channel_num[3], LayerNorm_type='WithBias')

        self.ffn1 = FeedForward(channel_num[0], ffn_expansion_factor=2.66, bias=False)
        self.ffn2 = FeedForward(channel_num[1], ffn_expansion_factor=2.66, bias=False)
        self.ffn3 = FeedForward(channel_num[2], ffn_expansion_factor=2.66, bias=False)
        self.ffn4 = FeedForward(channel_num[3], ffn_expansion_factor=2.66, bias=False)


    def forward(self, emb1, emb2, emb3, emb4):
        embcat = []
        org1 = emb1
        org2 = emb2
        org3 = emb3
        org4 = emb4
        for i in range(4):
            var_name = "emb" + str(i + 1)
            tmp_var = locals()[var_name]
            if tmp_var is not None:
                embcat.append(tmp_var)
        emb_all = torch.cat(embcat, dim=1)
        cx1 = self.attn_norm1(emb1) if emb1 is not None else None
        cx2 = self.attn_norm2(emb2) if emb2 is not None else None
        cx3 = self.attn_norm3(emb3) if emb3 is not None else None
        cx4 = self.attn_norm4(emb4) if emb4 is not None else None
        emb_all = self.attn_norm(emb_all)  # 1 196 960
        cx1, cx2, cx3, cx4, weights = self.channel_attn(cx1, cx2, cx3, cx4, emb_all)
        cx1 = org1 + cx1 if emb1 is not None else None#这里进行了px出输出和原始输出的add处理
        cx2 = org2 + cx2 if emb2 is not None else None
        cx3 = org3 + cx3 if emb3 is not None else None
        cx4 = org4 + cx4 if emb4 is not None else None

        org1 = cx1
        org2 = cx2
        org3 = cx3
        org4 = cx4
        x1 = self.ffn_norm1(cx1) if emb1 is not None else None##对应图3b处的归一化处理模块，所以不太严谨，他下面的ffn对应前向网络
        x2 = self.ffn_norm2(cx2) if emb2 is not None else None
        x3 = self.ffn_norm3(cx3) if emb3 is not None else None
        x4 = self.ffn_norm4(cx4) if emb4 is not None else None
        x1 = self.ffn1(x1) if emb1 is not None else None
        x2 = self.ffn2(x2) if emb2 is not None else None
        x3 = self.ffn3(x3) if emb3 is not None else None
        x4 = self.ffn4(x4) if emb4 is not None else None
        x1 = x1 + org1 if emb1 is not None else None
        x2 = x2 + org2 if emb2 is not None else None
        x3 = x3 + org3 if emb3 is not None else None
        x4 = x4 + org4 if emb4 is not None else None

        return x1, x2, x3, x4, weights


class Encoder(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm1 = LayerNorm3d(channel_num[0], LayerNorm_type='WithBias')
        self.encoder_norm2 = LayerNorm3d(channel_num[1], LayerNorm_type='WithBias')
        self.encoder_norm3 = LayerNorm3d(channel_num[2], LayerNorm_type='WithBias')
        self.encoder_norm4 = LayerNorm3d(channel_num[3], LayerNorm_type='WithBias')
        for _ in range(config.transformer["num_layers"]):
            layer = Block_ViT(config, vis, channel_num)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, emb1, emb2, emb3, emb4):
        attn_weights = []
        for layer_block in self.layer:
            emb1, emb2, emb3, emb4, weights = layer_block(emb1, emb2, emb3, emb4)
            if self.vis:
                attn_weights.append(weights)
        emb1 = self.encoder_norm1(emb1) if emb1 is not None else None
        emb2 = self.encoder_norm2(emb2) if emb2 is not None else None
        emb3 = self.encoder_norm3(emb3) if emb3 is not None else None
        emb4 = self.encoder_norm4(emb4) if emb4 is not None else None
        return emb1, emb2, emb3, emb4, attn_weights


def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(CBN(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(CBN(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class CBN(nn.Module):
    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(CBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CCA(nn.Module):
    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_x))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        avg_pool_x = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_x = self.mlp_x(avg_pool_x)
        avg_pool_g = F.avg_pool2d(g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)
        channel_att_sum = (channel_att_x + channel_att_g) / 2.0
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale
        out = self.relu(x_after_channel)
        return out


class UpBlock_attention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.coatt = CCA(F_g=in_channels // 2, F_x=in_channels // 2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = self.up(x)
        skip_x_att = self.coatt(g=up, x=skip_x)
        x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)


class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Res_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # self.fca = FCA_Layer(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


class MultiScaleTextureExtractor(nn.Module):
    """
    Multi-scale Texture Feature Extraction Module

    Mathematical Model:
    T_s(x,y) = {
        η·(σ_s²(x,y))/(σ_L²(x,y)+ε),  if σ_s²(x,y) > τ
        σ_s²(x,y),                    otherwise
    }

    Implemented as a PyTorch module to process feature maps of shape (B,C,H,W)
    """

    def __init__(self,
                 in_channels,
                 small_window_size=3,
                 large_window_size=11,
                 init_threshold=0.02,
                 init_enhance_factor=3.0,
                 epsilon=1e-7,
                 learnable=True):
        """
        Initialize the Multi-scale Texture Extractor Module

        Args:
            in_channels (int): Number of input channels
            small_window_size (int): Size of small-scale window for variance calculation
            large_window_size (int): Size of large-scale window for variance calculation
            init_threshold (float): Initial value for target detection threshold
            init_enhance_factor (float): Initial value for enhancement factor
            epsilon (float): Small constant to avoid division by zero
            learnable (bool): Whether parameters should be learnable
        """
        super(MultiScaleTextureExtractor, self).__init__()

        self.in_channels = in_channels
        self.small_window_size = small_window_size
        self.large_window_size = large_window_size
        self.epsilon = epsilon
        self.learnable = learnable

        # Define learnable parameters
        if learnable:
            # Channel-wise threshold adaptation
            self.threshold_adapt = nn.Sequential(####之前用的 组卷积
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.Sigmoid()
            )

            # Initialize weights to produce values close to init_threshold after sigmoid
            for m in self.threshold_adapt.modules():
                if isinstance(m, nn.Conv2d):
                    sigmoid_correction = torch.log(torch.tensor(init_threshold / (1 - init_threshold)))
                    nn.init.constant_(m.bias, sigmoid_correction)

            # Channel-wise enhancement factor adaptation
            self.enhance_factor_adapt = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.ReLU()
            )
            # self.enhance_factor_adapt = nn.Sequential(
            #     nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            #     nn.BatchNorm2d(in_channels),
            #     nn.ReLU()
            # )

            # Initialize weights to produce values close to init_enhance_factor
            for m in self.enhance_factor_adapt.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.constant_(m.bias, init_enhance_factor)

            # Context-aware refinement module
            self.refinement = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels, in_channels, kernel_size=1),
                nn.Sigmoid()
            )
        else:
            # Fixed parameters
            self.register_buffer('threshold', torch.ones(in_channels, 1, 1) * init_threshold)
            self.register_buffer('enhance_factor', torch.ones(in_channels, 1, 1) * init_enhance_factor)

    def uniform_filter2d(self, x, window_size):
        """
        Efficient implementation of uniform filter (mean filter) using depthwise convolution

        Args:
            x (torch.Tensor): Input tensor of shape (B,C,H,W)
            window_size (int): Size of the square window

        Returns:
            torch.Tensor: Filtered tensor of shape (B,C,H,W)
        """
        # Create averaging kernel
        kernel_size = window_size
        kernel = torch.ones(self.in_channels, 1, kernel_size, kernel_size,
                            device=x.device, dtype=x.dtype) / (kernel_size * kernel_size)

        # Apply depthwise convolution - each channel filtered separately
        padding = kernel_size // 2
        return F.conv2d(x, kernel, padding=padding, groups=self.in_channels)

    def calculate_local_variance(self, x, window_size):
        """
        Calculate local variance for each pixel in each channel

        Args:
            x (torch.Tensor): Input tensor of shape (B,C,H,W)
            window_size (int): Size of the square window

        Returns:
            torch.Tensor: Local variance tensor of shape (B,C,H,W)
        """
        # Square the input
        x_sq = x ** 2

        # Calculate local mean and local mean of squares
        mean = self.uniform_filter2d(x, window_size)  # E[X]
        mean_sq = self.uniform_filter2d(x_sq, window_size)  # E[X²]

        # Calculate variance: E[X²] - E[X]²
        var = mean_sq - (mean ** 2)###计算局部方差，局部方差能衡量一个区域内像素值的波动大小。

        # Clamp small negative values that might occur due to numerical precision
        var = torch.clamp(var, min=0.0)###这段可能需要删

        return var

    def forward(self, x):
        """
        Forward pass

        Args:
            x (torch.Tensor): Input tensor of shape (B,C,H,W)

        Returns:
            torch.Tensor: Texture features tensor of shape (B,C,H,W)
        """
        # Step 1: Normalize input to [0,1] range if needed
        # Assuming input is already normalized or in range that makes sense for feature maps

        # Step 2: Calculate multi-scale local variance
        small_var = self.calculate_local_variance(x, self.small_window_size)  # σ_s²(x,y)##小窗口的局部方差
        large_var = self.calculate_local_variance(x, self.large_window_size)  # σ_L²(x,y)##大窗口的局部方差

        if self.learnable:
            # Generate adaptive threshold based on input features
            threshold = self.threshold_adapt(x)

            # Generate adaptive enhancement factor
            enhance_factor = self.enhance_factor_adapt(x) + 1.0  # Add 1.0 to ensure it's always > 1
        else:
            # Use fixed parameters
            threshold = self.threshold
            enhance_factor = self.enhance_factor

        # Step 3: Identify potential target regions
        # Compare small window variance with threshold
        potential_targets = (small_var > threshold)

        # Step 4: Calculate variance ratio for feature enhancement
        # σ_s²(x,y) / (σ_L²(x,y) + ε)
        variance_ratio = small_var / (large_var + self.epsilon)

        # Apply enhancement factor to ratio in target regions
        enhanced_ratio = variance_ratio * enhance_factor####删掉这一块实施，感觉这样野性，对variance_var进行强化

        # Apply the mathematical model
        # T_s(x,y) = { η·(σ_s²(x,y))/(σ_L²(x,y)+ε), if σ_s²(x,y) > τ
        #             { σ_s²(x,y),                   otherwise
        texture_map = torch.where(potential_targets, enhanced_ratio, small_var)

        # Clamp values to reasonable range to prevent extreme values
        texture_map = torch.clamp(texture_map, 0.0, 5.0)###这个感觉先删除比较好

        if self.learnable:
            # Context-aware refinement
            # refinement_weights = self.refinement(x)
            # texture_map = texture_map * refinement_weights###这里之前有调整模块

            texture_map = texture_map

        # Normalize output per channel for stability
        # This is optional and can be removed if normalization is not desired
        # B, C, H, W = texture_map.shape
        # texture_map_reshaped = texture_map.view(B, C, -1)
        # min_vals, _ = torch.min(texture_map_reshaped, dim=2, keepdim=True)
        # max_vals, _ = torch.max(texture_map_reshaped, dim=2, keepdim=True)
        #
        # # Avoid division by zero
        # divisor = torch.clamp(max_vals - min_vals, min=self.epsilon)
        # normalized_map = (texture_map_reshaped - min_vals) / divisor
        # texture_map = normalized_map.view(B, C, H, W)

        return texture_map

class MultiScaleTextureExtractor_4beiout(nn.Module):
    """
    Multi-scale Texture Feature Extraction Module

    Mathematical Model:
    T_s(x,y) = {
        η·(σ_s²(x,y))/(σ_L²(x,y)+ε),  if σ_s²(x,y) > τ
        σ_s²(x,y),                    otherwise
    }

    Implemented as a PyTorch module to process feature maps of shape (B,C,H,W)
    """

    def __init__(self,
                 in_channels,
                 small_window_size=3,
                 large_window_size=11,
                 init_threshold=0.02,
                 init_enhance_factor=3.0,
                 epsilon=1e-7,
                 learnable=True):
        """
        Initialize the Multi-scale Texture Extractor Module

        Args:
            in_channels (int): Number of input channels
            small_window_size (int): Size of small-scale window for variance calculation
            large_window_size (int): Size of large-scale window for variance calculation
            init_threshold (float): Initial value for target detection threshold
            init_enhance_factor (float): Initial value for enhancement factor
            epsilon (float): Small constant to avoid division by zero
            learnable (bool): Whether parameters should be learnable
        """
        super(MultiScaleTextureExtractor_4beiout, self).__init__()

        self.in_channels = in_channels
        self.small_window_size = small_window_size
        self.large_window_size = large_window_size
        self.epsilon = epsilon
        self.learnable = learnable

        self.channel_4 = nn.Sequential(  ####之前用的 组卷积
            nn.Conv2d(in_channels, 4*in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(4*in_channels),
            nn.Sigmoid()
        )

        # Define learnable parameters
        if learnable:
            # Channel-wise threshold adaptation
            self.threshold_adapt = nn.Sequential(####之前用的 组卷积
                nn.Conv2d(in_channels*4, in_channels*4, kernel_size=3, padding=1, groups=in_channels*4),
                nn.BatchNorm2d(in_channels*4),
                nn.Sigmoid()
            )

            # Initialize weights to produce values close to init_threshold after sigmoid
            for m in self.threshold_adapt.modules():
                if isinstance(m, nn.Conv2d):
                    sigmoid_correction = torch.log(torch.tensor(init_threshold / (1 - init_threshold)))
                    nn.init.constant_(m.bias, sigmoid_correction)

            # Channel-wise enhancement factor adaptation
            self.enhance_factor_adapt = nn.Sequential(
                nn.Conv2d(in_channels*4, in_channels*4, kernel_size=3, padding=1, groups=in_channels*4),
                nn.BatchNorm2d(in_channels*4),
                nn.ReLU()
            )
            # self.enhance_factor_adapt = nn.Sequential(
            #     nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            #     nn.BatchNorm2d(in_channels),
            #     nn.ReLU()
            # )

            # Initialize weights to produce values close to init_enhance_factor
            for m in self.enhance_factor_adapt.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.constant_(m.bias, init_enhance_factor)

            # Context-aware refinement module
            self.refinement = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels, in_channels, kernel_size=1),
                nn.Sigmoid()
            )
        else:
            # Fixed parameters
            self.register_buffer('threshold', torch.ones(in_channels, 1, 1) * init_threshold)
            self.register_buffer('enhance_factor', torch.ones(in_channels, 1, 1) * init_enhance_factor)

    def uniform_filter2d(self, x, window_size):###这就是局部窗口求均值
        """
        Efficient implementation of uniform filter (mean filter) using depthwise convolution

        Args:
            x (torch.Tensor): Input tensor of shape (B,C,H,W)
            window_size (int): Size of the square window

        Returns:
            torch.Tensor: Filtered tensor of shape (B,C,H,W)
        """
        # Create averaging kernel
        kernel_size = window_size
        kernel = torch.ones(self.in_channels*4, 1, kernel_size, kernel_size,
                            device=x.device, dtype=x.dtype) / (kernel_size * kernel_size)

        # Apply depthwise convolution - each channel filtered separately
        padding = kernel_size // 2
        return F.conv2d(x, kernel, padding=padding, groups=self.in_channels*4)

    def calculate_local_variance(self, x, window_size):
        """
        Calculate local variance for each pixel in each channel

        Args:
            x (torch.Tensor): Input tensor of shape (B,C,H,W)
            window_size (int): Size of the square window

        Returns:
            torch.Tensor: Local variance tensor of shape (B,C,H,W)
        """
        # Square the input
        x_sq = x ** 2

        # Calculate local mean and local mean of squares
        mean = self.uniform_filter2d(x, window_size)  # E[X]
        mean_sq = self.uniform_filter2d(x_sq, window_size)  # E[X²]

        # Calculate variance: E[X²] - E[X]²
        var = mean_sq - (mean ** 2)

        # Clamp small negative values that might occur due to numerical precision
        var = torch.clamp(var, min=0.0)###这段可能需要删

        return var

    def forward(self, x):
        """
        Forward pass

        Args:
            x (torch.Tensor): Input tensor of shape (B,C,H,W)

        Returns:
            torch.Tensor: Texture features tensor of shape (B,C,H,W)
        """
        # Step 1: Normalize input to [0,1] range if needed
        # Assuming input is already normalized or in range that makes sense for feature maps

        # Step 2: Calculate multi-scale local variance
        x = self.channel_4(x)

        small_var = self.calculate_local_variance(x, self.small_window_size)  # σ_s²(x,y)
        large_var = self.calculate_local_variance(x, self.large_window_size)  # σ_L²(x,y)

        if self.learnable:
            # Generate adaptive threshold based on input features
            threshold = self.threshold_adapt(x)

            # Generate adaptive enhancement factor
            enhance_factor = self.enhance_factor_adapt(x) + 1.0  # Add 1.0 to ensure it's always > 1
        else:
            # Use fixed parameters
            threshold = self.threshold
            enhance_factor = self.enhance_factor

        # Step 3: Identify potential target regions
        # Compare small window variance with threshold
        potential_targets = (small_var > threshold)

        # Step 4: Calculate variance ratio for feature enhancement
        # σ_s²(x,y) / (σ_L²(x,y) + ε)
        variance_ratio = small_var / (large_var + self.epsilon)

        # Apply enhancement factor to ratio in target regions
        enhanced_ratio = variance_ratio * enhance_factor####删掉这一块实施

        # Apply the mathematical model
        # T_s(x,y) = { η·(σ_s²(x,y))/(σ_L²(x,y)+ε), if σ_s²(x,y) > τ
        #             { σ_s²(x,y),                   otherwise
        texture_map = torch.where(potential_targets, enhanced_ratio, small_var)

        # Clamp values to reasonable range to prevent extreme values
        texture_map = torch.clamp(texture_map, 0.0, 5.0)###这个感觉先删除比较好

        if self.learnable:
            # Context-aware refinement
            # refinement_weights = self.refinement(x)
            # texture_map = texture_map * refinement_weights###这里之前有调整模块

            texture_map = texture_map

        # Normalize output per channel for stability
        # This is optional and can be removed if normalization is not desired
        # B, C, H, W = texture_map.shape
        # texture_map_reshaped = texture_map.view(B, C, -1)
        # min_vals, _ = torch.min(texture_map_reshaped, dim=2, keepdim=True)
        # max_vals, _ = torch.max(texture_map_reshaped, dim=2, keepdim=True)
        #
        # # Avoid division by zero
        # divisor = torch.clamp(max_vals - min_vals, min=self.epsilon)
        # normalized_map = (texture_map_reshaped - min_vals) / divisor
        # texture_map = normalized_map.view(B, C, H, W)

        return texture_map
class MultiScaleTextureNetwork(nn.Module):
    """
    Complete network for multi-scale texture feature extraction
    Includes options for CNN-based adaptation of the core texture extraction algorithm
    """

    def __init__(self,
                 in_channels,
                 small_window_size=3,
                 large_window_size=11,
                 init_threshold=0.02,
                 init_enhance_factor=3.0,
                 epsilon=1e-7,
                 learnable=True,
                 use_context_module=True,
                 uiu_layer = 3):
        """
        Initialize the complete texture extraction network

        Args:
            in_channels (int): Number of input channels
            small_window_size (int): Size of small-scale window for variance calculation
            large_window_size (int): Size of large-scale window for variance calculation
            init_threshold (float): Initial value for target detection threshold
            init_enhance_factor (float): Initial value for enhancement factor
            epsilon (float): Small constant to avoid division by zero
            learnable (bool): Whether parameters should be learnable
            use_context_module (bool): Whether to use context-aware feature enhancement
        """
        super(MultiScaleTextureNetwork, self).__init__()

        # Feature preprocessing (optional)
        self.use_context_module = use_context_module
        if use_context_module:
            self.context_module = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU()
            )

        # Core texture extraction module
        self.texture_extractor = MultiScaleTextureExtractor(
            in_channels=in_channels,
            small_window_size=small_window_size,
            large_window_size=large_window_size,
            init_threshold=init_threshold,
            init_enhance_factor=init_enhance_factor,
            epsilon=epsilon,
            learnable=learnable
        )
        self.uiu_layer = uiu_layer
        if self.uiu_layer == 4:
            self.uiu = RSU4(in_channels, in_channels//2, in_channels)
        if self.uiu_layer == 3:
            self.uiu = RSU3(in_channels, in_channels//2, in_channels)
            self.uiu_trans = RSU3_Trans(in_channels)
        if self.uiu_layer == 2:
            self.uiu = RSU2(in_channels, in_channels//2, in_channels)
            self.uiu_trans = RSU2_Trans(in_channels)
        # self.uiu_trans_txue = RSU2_Trans(in_channels)

        # Feature post-processing (optional)
        self.feature_refiner = nn.Sequential(
            nn.Conv2d(in_channels, in_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels*2),
            nn.ReLU(),
            nn.Conv2d(in_channels*2, in_channels, kernel_size=1)
        )

        # Initialize weights
        # self._initialize_weights()
        block_res_text = Res_block
        self.channel = in_channels
        self.texture_process = self._make_layer(block_res_text, self.channel, self.channel, 1)  # 64  128

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)


    # def _initialize_weights(self):
    #     """Initialize network weights"""
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)

    def forward(self, x, o):
        """
        Forward pass

        Args:
            x (torch.Tensor): Input tensor of shape (B,C,H,W)

        Returns:
            torch.Tensor: Enhanced texture features tensor of shape (B,C,H,W)
        """
        # Apply context-aware preprocessing if enabled
        # if self.use_context_module:
        #     x = self.context_module(x)

        # Extract texture features
        texture_features = self.texture_extractor(x)###纹理图 这里必须经过卷积的！！！


        texture_features = self.uiu_trans(texture_features)
        x_uiu = self.uiu(x)

        # texture_features_trans = self.uiu_trans(texture_features)
        # x_uiu_trans = self.uiu_trans(x)


        # Refine features
        refined_features = self.feature_refiner(texture_features+x_uiu)
        # refined_features = x

        # Add residual connection
        output = refined_features

        target_size = (o.size(-2), o.size(-1))  # (H_o, W_o)
        # Upsample output to match target size
        output = F.interpolate(
            output,
            size=target_size,
            mode='bilinear',
            align_corners=True
        )

        return output

class MultiScaleTextureNetwork_out(nn.Module):
    """
    Complete network for multi-scale texture feature extraction
    Includes options for CNN-based adaptation of the core texture extraction algorithm
    """

    def __init__(self,
                 in_channels,
                 small_window_size=3,
                 large_window_size=11,
                 init_threshold=0.02,
                 init_enhance_factor=3.0,
                 epsilon=1e-7,
                 learnable=True,
                 use_context_module=True):
        """
        Initialize the complete texture extraction network

        Args:
            in_channels (int): Number of input channels
            small_window_size (int): Size of small-scale window for variance calculation
            large_window_size (int): Size of large-scale window for variance calculation
            init_threshold (float): Initial value for target detection threshold
            init_enhance_factor (float): Initial value for enhancement factor
            epsilon (float): Small constant to avoid division by zero
            learnable (bool): Whether parameters should be learnable
            use_context_module (bool): Whether to use context-aware feature enhancement
        """
        super(MultiScaleTextureNetwork_out, self).__init__()

        # Feature preprocessing (optional)
        self.use_context_module = use_context_module
        if use_context_module:
            self.context_module = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU()
            )

        # Core texture extraction module
        self.texture_extractor = MultiScaleTextureExtractor_4beiout(
            in_channels=in_channels,
            small_window_size=small_window_size,
            large_window_size=large_window_size,
            init_threshold=init_threshold,
            init_enhance_factor=init_enhance_factor,
            epsilon=epsilon,
            learnable=learnable
        )
        self.uiu = RSU3_Trans(in_channels*4)
        self.uiu_x = RSU3(in_channels*4, in_channels*2, in_channels*4)
        # self.uiu_trans_x = RSU2_Trans_out(in_channels)
        # self.uiu_trans_textture = RSU2_Trans_out(in_channels*4)

        # Feature post-processing (optional)
        self.feature_refiner = nn.Sequential(
            nn.Conv2d(in_channels*4, in_channels*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels*4),
            nn.ReLU()
        )
        self.one2four = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=1)
        )


        # Initialize weights
        # self._initialize_weights()
        block_res_text = Res_block
        self.channel = in_channels
        self.texture_process = self._make_layer(block_res_text, self.channel, self.channel, 1)  # 64  128

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)


    # def _initialize_weights(self):
    #     """Initialize network weights"""
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)

    def forward(self, x, o):
        """
        Forward pass

        Args:
            x (torch.Tensor): Input tensor of shape (B,C,H,W)

        Returns:
            torch.Tensor: Enhanced texture features tensor of shape (B,C,H,W)
        """
        # Apply context-aware preprocessing if enabled
        # if self.use_context_module:
        #     x = self.context_module(x)

        # Extract texture features
        texture_features = self.texture_extractor(x)###纹理图 这里必须经过卷积的！！！
        x_4 = self.one2four(x)
        texture_features = self.uiu(texture_features)
        x_uiu = self.uiu_x(x_4)
        # texture_features_trans = self.uiu_trans_textture(texture_features)
        # x_uiu_trans = self.uiu_trans_x(x)


        # Refine features
        refined_features = self.feature_refiner(texture_features+x_uiu)
        # refined_features = x

        # Add residual connection
        output = refined_features

        target_size = (o.size(-2), o.size(-1))  # (H_o, W_o)
        # Upsample output to match target size
        output = F.interpolate(
            output,
            size=target_size,
            mode='bilinear',
            align_corners=True
        )

        return output

class LightweightResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(LightweightResBlock, self).__init__()

        # 深度可分离卷积的实现
        self.conv1 = nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels),
            # Pointwise convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

        # 可以选择移除第二个卷积层来进一步轻量化
        # 或者同样使用深度可分离卷积

        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                # 使用1x1卷积进行通道调整
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x if self.shortcut is None else self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)

        out += residual
        out = self.relu(out)
        return out

class REBNCONV(nn.Module):
    def __init__(self,in_ch=3,out_ch=3,dirate=1):
        super(REBNCONV,self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self,x):

        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout

def _upsample_like(src,tar):

    src = F.upsample(src, size=tar.shape[2:], mode='bilinear')

    return src

class RSU4(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

class RSU3(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU3,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)


        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=2)



        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)


        hx3 = self.rebnconv3(hx)



        hx2d = self.rebnconv2d(torch.cat((hx3, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

class RSU3_Trans(nn.Module):

    def __init__(self, in_ch=3):
        super(RSU3_Trans,self).__init__()

        self.rebnconvin = Attention_org(in_ch)

        self.rebnconv1 = Attention_org(in_ch)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = Attention_org(in_ch)


        self.rebnconv3 = Attention_org(in_ch)



        self.rebnconv2d = Attention_org(in_ch)
        self.rebnconv1d = Attention_org(in_ch)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)


        hx3 = self.rebnconv3(hx)



        hx2d = self.rebnconv2d((hx3+hx2))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(hx2dup+hx1)

        return hx1d + x

class RSU2(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU2,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)

        # print(mid_ch)





        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        # hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx1)






        hx1d = self.rebnconv1d(torch.cat((hx2,hx1),1))

        return hx1d + hxin

class RSU2_Trans(nn.Module):

    def __init__(self, in_ch=3):
        super(RSU2_Trans,self).__init__()

        self.rebnconv1 = Attention_org(in_ch)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = Attention_org(in_ch)

        # print(mid_ch)





        self.rebnconv1d = Attention_org(in_ch)

    def forward(self,x):

        hx = x

        hx1 = self.rebnconv1(hx)
        # hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx1)

        hx1d = self.rebnconv1d(hx1+hx2)

        return hx1d + x

class RSU2_Trans_out(nn.Module):

    def __init__(self, in_ch=3):
        super(RSU2_Trans_out,self).__init__()

        self.rebnconv1 = Attention_org_out(in_ch)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = Attention_org_out(in_ch)

        # print(mid_ch)





        self.rebnconv1d = Attention_org_out(in_ch)

    def forward(self,x):

        hx = x

        hx1 = self.rebnconv1(hx)
        # hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx1)

        hx1d = self.rebnconv1d(hx1+hx2)

        return hx1d + hx


def generate_custom_gaussian_noise(feature_map, scale=0.01):
    """
    根据特征图生成与其像素值相关联的高斯噪声。

    Args:
        feature_map (torch.Tensor): 输入的特征图，形状为 [B, C, H, W]。
        scale (float): 用于控制噪声强度的比例因子，默认值为 0.01。

    Returns:
        torch.Tensor: 与输入特征图形状一致的高斯噪声。
    """
    # 计算特征图的标准差作为噪声的标准差
    noise_std = feature_map.abs() * scale  # 噪声标准差与特征图值相关联

    # 生成高斯噪声
    gaussian_noise = torch.randn_like(feature_map) * noise_std

    return gaussian_noise

class SCTransNet(nn.Module):
    def __init__(self, config, n_channels=1, n_classes=1, img_size=256, vis=False, mode='train', deepsuper=True):
        super().__init__()
        self.vis = vis
        self.deepsuper = deepsuper
        print('Deep-Supervision:', deepsuper)
        self.mode = mode
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = 16#config.base_channel  # basic channel 64
        block_res =  Res_block
        Light_block_res = LightweightResBlock
        self.pool = nn.MaxPool2d(2, 2)
        self.inc = self._make_layer(block_res, n_channels, in_channels)
        self.down_encoder1 = self._make_layer(block_res, in_channels, in_channels * 2, 1)  # 64  128
        self.down_encoder2 = self._make_layer(block_res, in_channels * 2, in_channels * 4, 1)  # 64  128
        self.down_encoder3 = self._make_layer(block_res, in_channels * 4, in_channels * 8, 1)  # 64  128
        self.down_encoder4 = self._make_layer(block_res, in_channels * 8, in_channels * 8, 1)  # 64  128
        print(in_channels)

        # self.mtc = ChannelTransformer(config, vis, img_size,
        #                               channel_num=[in_channels, in_channels * 2, in_channels * 4, in_channels * 8],
        #                               patchSize=config.patch_sizes)
        self.up_decoder4 = UpBlock_attention(in_channels * 16, in_channels * 4, nb_Conv=2)
        self.up_decoder3 = UpBlock_attention(in_channels * 8, in_channels * 2, nb_Conv=2)
        self.up_decoder2 = UpBlock_attention(in_channels * 4, in_channels, nb_Conv=2)
        self.up_decoder1 = UpBlock_attention(in_channels * 2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1), stride=(1, 1))

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.inc(x)  # 32 256 256
        x2 = self.down_encoder1(self.pool(x1))  # 64 128 128
        x3 = self.down_encoder2(self.pool(x2))  # 128 64  64
        x4 = self.down_encoder3(self.pool(x3))  # 256 32  32
        d5 = self.down_encoder4(self.pool(x4))  # 256 16  16

        ###
        f1 = x1
        f2 = x2
        f3 = x3
        f4 = x4
        #  CCT
        # x1, x2, x3, x4, att_weights = self.mtc(x1, x2, x3, x4)
        x1 = x1 + f1
        x2 = x2 + f2
        x3 = x3 + f3
        x4 = x4 + f4
        ###

        #  Feature fusion
        d4 = self.up_decoder4(d5, x4)###32
        d3 = self.up_decoder3(d4, x3)###64
        d2 = self.up_decoder2(d3, x2)###128
        out = self.outc(self.up_decoder1(d2, x1))###256
        return torch.sigmoid(out)










if __name__ == '__main__':
    config_vit = get_CTranS_config()
    model = SCTransNet(config_vit, mode='train', deepsuper=True)
    model = model
    inputs = torch.rand(2, 1, 256, 256)
    output = model(inputs)
    flops, params = profile(model, (inputs,))

    print("-" * 50)
    print('FLOPs = ' + str(flops / 1000 ** 3) + ' G')
    print('Params = ' + str(params / 1000 ** 2) + ' M')
