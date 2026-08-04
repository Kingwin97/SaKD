import torch
import torch.nn as nn

# INR的fusion

class CONV(nn.Module):
    default_act = nn.ReLU(inplace=True)  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, d=1, act=True):
        super(CONV, self).__init__()

        self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=p, groups=g, dilation=d)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Fuse(nn.Module):
    def __init__(self, in_high_channels, in_low_channels, out_channels=64, r=4):
        super(Fuse, self).__init__()
        assert in_low_channels == out_channels

        self.high_channels = in_high_channels
        self.low_channels = in_low_channels
        self.out_channels = out_channels

        self.bottleneck_channels = int(out_channels // r)

        self.feature_high = nn.Sequential(
            nn.Conv2d(self.high_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(True),
        )  ##512

        self.feature_low = nn.Sequential(
            nn.Conv2d(self.high_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(True),
        )  ##512

        self.topdown = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.out_channels, self.bottleneck_channels, 1, 1, 0),
            nn.BatchNorm2d(self.bottleneck_channels),
            nn.ReLU(True),

            nn.Conv2d(self.bottleneck_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(self.out_channels),
            nn.Sigmoid()
        )  # 512

        ##############add spatial attention ###Cross UtU############
        self.bottomup = SpatialAttention(self.low_channels, self.high_channels)

        self.post = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(True),
        )  # 512

    def forward(self, xh, xl):
        xh = self.feature_high(xh)
        w_t = self.topdown(xh)
        xl_tau = xl * w_t

        
        theta = self.bottomup(xl_tau, xh)
        out = self.post(theta * xh + (1 - theta) * xl)
        return out

        ##############################


class SpatialAttention(nn.Module):
    def __init__(self, low_channels, high_channels):
        super(SpatialAttention, self).__init__()
        self.low_channels = low_channels
        self.high_channels = high_channels

        self.convl = nn.Conv2d(in_channels=self.low_channels,out_channels=1,kernel_size=3,stride=1,padding=1)
        self.convh = nn.Conv2d(in_channels=self.high_channels,out_channels=1,kernel_size=3,stride=1,padding=1)

        self.bn = nn.BatchNorm2d(1)

    def forward(self, xl, xh):
        attenl = self.convl(xl)
        attenh = self.convh(xh)
        theta = torch.sigmoid(attenl*attenh)

        return theta


if __name__ == "__main__":
    x = torch.randn(4, 4, 4, 4)
    y = torch.randn(4, 4, 4, 4)
    demo = Fuse(4, 4, 4)
    z = demo(x, y)
    print(z.size())