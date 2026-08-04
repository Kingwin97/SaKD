#UTransformer的改进版本

import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from model.mlp import INR as INR
from model.fusion3 import *
#############################################



from thop import profile


##################################################
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=False, norm=False, relu=True, transpose=False,
                 channel_shuffle_g=0, norm_method=nn.BatchNorm2d, groups=1):
        super(BasicConv, self).__init__()
        self.channel_shuffle_g = channel_shuffle_g
        self.norm = norm
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias,
                                   groups=groups))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias,
                          groups=groups))
        if norm:
            layers.append(norm_method(out_channel))
        elif relu:
            layers.append(nn.ReLU(inplace=True))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


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


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, BasicConv=BasicConv):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = BasicConv(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, bias=bias,
                                relu=False, groups=hidden_features * 2)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, BasicConv=BasicConv):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = BasicConv(dim * 3, dim * 3, kernel_size=3, stride=1, bias=bias, relu=False, groups=dim * 3)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, BasicConv=BasicConv):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias, BasicConv=BasicConv)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias, BasicConv=BasicConv)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        # self.relu_s1 = nn.GELU()
        self.relu_s1 = nn.ReLU(inplace=True)
        # self.relu_s1 = nn.SiLU(inplace=True)

    def forward(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x) # b,c,h,w->b,c/2,h,w->b,2c,h/2,w/2


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)   # b,c,h,w->b,2c,h,w->b,c/2,2h,2w

def _upsample_like(src, tar):
    _, _, hei, wid = tar.shape
    src = F.interpolate(src, size=[hei, wid], mode='bilinear', align_corners=True)

    return src


class U3(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(U3, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, mid_ch, dirate=1)
        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv1 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv3d = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.fuse3 = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.fuse2 = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.fuse1 = REBNCONV(mid_ch * 2, out_ch, dirate=1)

        self.out_conv = REBNCONV(out_ch, out_ch, 1)

    def forward(self, x):
        _, _, hei, wid = x.shape
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool(hx3)

        # ------------decoder-----------

        hx3d = self.rebnconv3d(hx)
        hx3dup = _upsample_like(hx3d, hx3)
        hx3f = self.fuse3(torch.cat((hx3dup, hx3), 1))

        hx2d = self.rebnconv2d(hx3f)
        hx2dup = _upsample_like(hx2d, hx2)
        hx2f = self.fuse2(torch.cat((hx2dup, hx2), 1))

        hx1d = self.rebnconv1d(hx2f)
        hx1dup = _upsample_like(hx1d, hx1)
        hx1f = self.fuse1(torch.cat((hx1dup, hx1), 1))

        return self.out_conv(hx1f)


class UT3(nn.Module):

    def __init__(self,
                 inp_channels=3,
                 dim=48,
                 out_channels=3,
                 num_blocks=[2, 3, 3],
                 heads=[1, 2, 4],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):
        super(UT3, self).__init__()

        self.rebnconvin = REBNCONV(inp_channels,dim, dirate=1)

        self.aConV = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1), nn.BatchNorm2d(out_channels), REBNCONV(out_channels, out_channels, dirate=1))

        # self.aConV = REBNCONV(out_channels, out_channels, dirate=1)

        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 1 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 1 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 1 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])


        self.latent = REBNCONV(dim, dim, dirate=1)

        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 1 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 1 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 1 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])


        self.fuse3 = REBNCONV(dim * 2, dim, dirate=1)
        self.fuse2 = REBNCONV(dim * 2, dim, dirate=1)
        self.fuse1 = REBNCONV(dim * 2, out_channels, dirate=1)

        self.out_conv = REBNCONV(out_channels, out_channels, 1)

    def forward(self, x):
        _, _, hei, wid = x.shape


        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.encoder_level1(hxin)
        hx = self.pool(hx1)

        hx2 = self.encoder_level2(hx)
        hx = self.pool(hx2)

        hx3 = self.encoder_level3(hx)
        hx = self.pool(hx3)

        hx = self.latent(hx)

        # ------------decoder-----------

        hx3d = self.decoder_level3(hx)
        hx3dup = _upsample_like(hx3d, hx3)
        hx3f = self.fuse3(torch.cat((hx3dup, hx3), 1))

        hx2d = self.decoder_level2(hx3f)
        hx2dup = _upsample_like(hx2d, hx2)
        hx2f = self.fuse2(torch.cat((hx2dup, hx2), 1))

        hx1d = self.decoder_level1(hx2f)
        hx1dup = _upsample_like(hx1d, hx1)
        hx1f = self.fuse1(torch.cat((hx1dup, hx1), 1))

        hx1f = hx1f + torch.sigmoid(self.aConV(hx1f))

        return self.out_conv(hx1f)

class UT2(nn.Module):

    def __init__(self,
                 inp_channels=3,
                 dim=48,
                 out_channels=3,
                 num_blocks=[2, 3, 3],
                 heads=[1, 2, 4],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):
        super(UT2, self).__init__()

        self.rebnconvin = REBNCONV(inp_channels,dim, dirate=1)

        self.aConV = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1), nn.BatchNorm2d(out_channels), REBNCONV(out_channels, out_channels, dirate=1))

        # self.aConV = REBNCONV(out_channels, out_channels, dirate=1)

        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 1 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 1 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])


        self.latent = REBNCONV(dim, dim, dirate=1)

        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 1 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 1 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])



        self.fuse2 = REBNCONV(dim * 2, dim, dirate=1)
        self.fuse1 = REBNCONV(dim * 2, out_channels, dirate=1)

        self.out_conv = REBNCONV(out_channels, out_channels, 1)

    def forward(self, x):
        _, _, hei, wid = x.shape


        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.encoder_level1(hxin)
        hx = self.pool(hx1)

        hx2 = self.encoder_level2(hx)
        hx = self.pool(hx2)

        hx = self.latent(hx)

        # ------------decoder-----------

        hx2d = self.decoder_level2(hx)
        hx2dup = _upsample_like(hx2d, hx2)
        hx2f = self.fuse2(torch.cat((hx2dup, hx2), 1))

        hx1d = self.decoder_level1(hx2f)
        hx1dup = _upsample_like(hx1d, hx1)
        hx1f = self.fuse1(torch.cat((hx1dup, hx1), 1))

        hx1f = hx1f + torch.sigmoid(self.aConV(hx1f))

        return self.out_conv(hx1f)


class UT1(nn.Module):

    def __init__(self,
                 inp_channels=3,
                 dim=48,
                 out_channels=3,
                 num_blocks=[2, 3, 3],
                 heads=[1, 2, 4],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):
        super(UT1, self).__init__()

        self.rebnconvin = REBNCONV(inp_channels,dim, dirate=1)

        self.aConV = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1), nn.BatchNorm2d(out_channels), REBNCONV(out_channels, out_channels, dirate=1))

        # self.aConV = REBNCONV(out_channels, out_channels, dirate=1)

        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 1 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.latent = REBNCONV(dim, dim, dirate=1)

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 1 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])



        self.fuse1 = REBNCONV(dim * 2, out_channels, dirate=1)

        self.out_conv = REBNCONV(out_channels, out_channels, 1)

    def forward(self, x):
        _, _, hei, wid = x.shape


        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.encoder_level1(hxin)
        hx = self.pool(hx1)

        hx = self.latent(hx)

        # ------------decoder-----------

        hx1d = self.decoder_level1(hx)
        hx1dup = _upsample_like(hx1d, hx1)
        hx1f = self.fuse1(torch.cat((hx1dup, hx1), 1))

        hx1f = hx1f + torch.sigmoid(self.aConV(hx1f))

        return self.out_conv(hx1f)



class PIP(nn.Module):

    def __init__(self, in_ch=1, out_ch=1, mode='train', deepsuper=True):
        super(PIP, self).__init__()
        self.mode = mode
        self.deepsuper = deepsuper

        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.pool2 = nn.MaxPool2d(4, stride=4, ceil_mode=True)
        self.pool3 = nn.MaxPool2d(8, stride=8, ceil_mode=True)
        self.pool4 = nn.MaxPool2d(16, stride=16, ceil_mode=True)

        self.inr1 = INR(in_dim=1, out_dim=1).cuda()
        self.inr2 = INR(in_dim=1, out_dim=1).cuda()
        self.inr3 = INR(in_dim=1, out_dim=1).cuda()
        self.inr4 = INR(in_dim=1, out_dim=1).cuda()

        self.stage1 = UT3(in_ch + 1, 48, 32)
        self.stage2 = UT3(32 + 1, 48, 64)
        self.stage3 = UT3(64 + 1, 48, 128)
        self.stage4 = UT3(128 + 1, 48, 256)

        self.stage4d = UT3(256, 48, 256)  # 第一个d注意改
        self.stage3d = UT3(256, 48, 128)
        self.stage2d = UT3(128, 48, 64)
        self.stage1d = UT3(64, 48, 32)

        self.fuse4 = self._fuse_layer(256, 256, 256, fuse_mode='AsymBi')
        self.fuse3 = self._fuse_layer(128, 128, 128, fuse_mode='AsymBi')
        self.fuse2 = self._fuse_layer(64, 64, 64, fuse_mode='AsymBi')
        self.fuse1 = self._fuse_layer(32, 32, 32, fuse_mode='AsymBi')

        # ------------------------PDE--------------------------

        self.side1 = nn.Conv2d(256, out_ch, 1)  # 深度监督
        self.side2 = nn.Conv2d(256, out_ch, 1)  # 这里注意改
        self.side3 = nn.Conv2d(128, out_ch, 1)
        self.side4 = nn.Conv2d(64, out_ch, 1)
        self.side5 = nn.Conv2d(32, out_ch, 1)

        self.out_conv = nn.Conv2d(64, out_ch, 1)
        self.outconv = nn.Conv2d(5 * out_ch, out_ch, 1)  # 这里注意改

    def _fuse_layer(self, in_high_channels, in_low_channels, out_channels, fuse_mode='AsymBi'):  # fuse_mode='AsymBi'

        if fuse_mode == 'AsymBi':
            fuse_layer = Fuse(in_high_channels, in_low_channels, out_channels)
            # fuse_layer = AsymBiChaFuseReduce(in_high_channels, in_low_channels, out_channels)
        else:
            NameError
        return fuse_layer

    def forward(self, x):
        _, _, hei, wid = x.shape
        hx = x

        inr_input1 = x
        inr_input2 = self.pool(inr_input1)
        inr_input3 = self.pool(inr_input2)
        inr_input4 = self.pool(inr_input3)

        # stage 1

        inr_output1 = self.inr1(inr_input1)

        hx1 = self.stage1(torch.cat([hx, inr_output1], dim=1))
        hx = self.pool(hx1)

        # stage 2
        inr_output2 = self.inr2(inr_input2)
        hx2 = self.stage2(torch.cat([hx, inr_output2], dim=1))
        hx = self.pool(hx2)

        # stage 3
        inr_output3 = self.inr3(inr_input3)
        hx3 = self.stage3(torch.cat([hx, inr_output3], dim=1))
        hx = self.pool(hx3)

        # stage 4

        inr_output4 = self.inr4(inr_input4)
        hx4 = self.stage4(torch.cat([hx, inr_output4], dim=1))
        hx = self.pool(hx4)

        # -------------------- decoder --------------------

        hx4d = self.stage4d(hx)
        hx4dup = _upsample_like(hx4d, hx4)
        hx4f = self.fuse4(hx4dup, hx4)

        hx3d = self.stage3d(hx4f)  # 这里注意改
        hx3dup = _upsample_like(hx3d, hx3)
        hx3f = self.fuse3(hx3dup, hx3)

        hx2d = self.stage2d(hx3f)
        hx2dup = _upsample_like(hx2d, hx2)
        hx2f = self.fuse2(hx2dup, hx2)

        hx1d = self.stage1d(hx2f)
        hx1dup = _upsample_like(hx1d, hx1)
        hx1f = self.fuse1(hx1dup, hx1)

        # --------------------deep supervision-------------------
        if self.deepsuper:
            d5 = F.interpolate(self.side1(hx4), size=[hei, wid])  # 这里注意改
            d4 = F.interpolate(self.side2(hx4f), size=[hei, wid])
            d3 = F.interpolate(self.side3(hx3f), size=[hei, wid])
            d2 = F.interpolate(self.side4(hx2f), size=[hei, wid])
            d1 = self.side5(hx1f)
            out = self.outconv(torch.cat((d1, d2, d3, d4, d5), 1))
            if self.mode == 'train':
                return torch.sigmoid(out)
                # return torch.sigmoid(out), inr_output1, inr_output2, inr_output3, inr_output4
            else:
                return torch.sigmoid(out)
        else:
            print("no")
            return 0


if __name__ == '__main__':
    model = PIP(1, 1, mode='train', deepsuper=True)

    inputs = torch.rand(2, 1, 256, 256)

    output = model(inputs)
    flops, params = profile(model, (inputs,))

    print("-" * 50)
    print('FLOPs = ' + str(flops / 1000 ** 3) + ' G')
    print('Params = ' + str(params / 1000 ** 2) + ' M')