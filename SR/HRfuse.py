'''
fuse super HR features & LR features
'''
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from typing import Optional, Callable
from torch import Tensor

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class Upsampler(nn.Sequential):
    def __init__(self, conv=default_conv, scale=4, n_feats=16, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class HRfuse(nn.Module):
    def __init__(self, hr_channel=16, lr_channel=16, mid_channel=16, out_channel=3, upscale=4):
        super().__init__()
        # first, fuse at the low-level features
        self.fuse = nn.Sequential(
            nn.Conv2d(hr_channel + lr_channel, mid_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, mid_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True))
        # upsampler at high-resolution features
        self.upsampler = Upsampler(scale=upscale, n_feats=mid_channel)
        self.conv_last = nn.Conv2d(mid_channel, out_channel, 3, 1, 1)

    def forward(self, x_lr, x_hr):
        x = self.fuse(torch.cat([x_lr, x_hr], dim=1))
        x = self.upsampler(x)
        x = self.conv_last(x)
        return x

# resize 2 times
class HRfuse_x2(nn.Module):
    def __init__(self, hr_channel=16, lr_channel=16, mid_channel=16, out_channel=3, upscale=4):
        super().__init__()
        # upsampler at high-resolution features
        self.upsampler = Upsampler(scale=upscale, n_feats=mid_channel)
        # first, fuse at the low-level features
        self.fuse = nn.Sequential(
            nn.Conv2d(hr_channel + lr_channel, mid_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, mid_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True))
        self.conv_last = nn.Conv2d(mid_channel, out_channel, 3, 1, 1)
        # self.conv_last = nn.Conv2d(mid_channel, out_channel, 1)
    def forward(self, x_lr, x_hr):
        x_lr = self.upsampler(x_lr) # upsample to 4x
        x = self.fuse(torch.cat([x_lr, x_hr], dim=1))
        # x = self.upsampler(x)
        x = self.conv_last(x)
        return x


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        # downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        expansion: int = 1
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = None
        self.stride = stride
        if stride != 1 or inplanes != planes * expansion:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes * expansion, stride),
                norm_layer(planes * expansion),
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# HR information extraction from HR features, e.g., texture and spatial details.
# firstly, try two-layer resnet
class HRfeature(nn.Sequential):
    def __init__(self, in_chans, mid_chans=64, out_chans=64):
        feat = [BasicBlock(in_chans, mid_chans, stride=1),
                BasicBlock(mid_chans, mid_chans, stride=1),
                BasicBlock(mid_chans, out_chans, stride=1)]
        super(HRfeature, self).__init__(*feat)


# use residual block
class HRfuse_residual(nn.Module):
    def __init__(self, hr_chans=16, lr_chans=16, mid_chans=16, out_chans=3, upscale=4):
        super().__init__()
        # upsampler at high-resolution features
        self.upsampler = Upsampler(scale=upscale, n_feats=lr_chans)
        # first, fuse at the low-level features
        self.fuse = nn.Sequential(
             BasicBlock(hr_chans+lr_chans, mid_chans, stride=1),
             BasicBlock(mid_chans, mid_chans, stride=1),
             BasicBlock(mid_chans, mid_chans, stride=1))
        #self.conv_last = nn.Conv2d(mid_chans, out_chans, 3, 1, 1)
        self.conv_last = nn.Conv2d(mid_chans, out_chans, 3, 1, 1)
    def forward(self, x_lr, x_hr):
        x_lr = self.upsampler(x_lr) # upsample to 4x
        x = self.fuse(torch.cat([x_lr, x_hr], dim=1))
        # x = self.upsampler(x)
        x = self.conv_last(x)
        return x

# 2023.12.14: used for no-super resolution module
class HRupsample(nn.Module):
    def __init__(self, lr_chans=16, out_chans=3, upscale=4):
        super().__init__()
        # upsampler at high-resolution features
        self.upsampler = Upsampler(scale=upscale, n_feats=lr_chans)
        self.conv_last = nn.Conv2d(lr_chans, out_chans, 3, 1, 1)
    def forward(self, x):
        x = self.upsampler(x)
        x = self.conv_last(x)
        return x

# write one, process lon, lat, alt
class GeoNet(nn.Module):
    def __init__(self, in_chans=4, mid_chans=16):
        super().__init__()
        self.feat = nn.Sequential(
            BasicBlock(in_chans, mid_chans, stride=1),
            BasicBlock(mid_chans, mid_chans, stride=1),
            BasicBlock(mid_chans, mid_chans, stride=1))
    def forward(self, x):
        return self.feat(x)


class Refine_residual(nn.Module):
    def __init__(self, hr_chans=16, lr_chans=16, mid_chans=16, out_chans=3):
        super().__init__()
        self.fuse = nn.Sequential(
             BasicBlock(hr_chans+lr_chans, mid_chans, stride=1),
             BasicBlock(mid_chans, mid_chans, stride=1),
             BasicBlock(mid_chans, mid_chans, stride=1))
        #self.conv_last = nn.Conv2d(mid_chans, out_chans, 3, 1, 1)
        self.conv_last = nn.Conv2d(mid_chans, out_chans, 3, 1, 1)
    def forward(self, x_lr, x_hr):
        x = self.fuse(torch.cat([x_lr, x_hr], dim=1))
        x = self.conv_last(x)
        return x


if __name__ == '__main__':
    # model = HRfeature(in_chans=3, mid_chans=64, out_chans=64)
    # model = GeoNet(in_chans=4, mid_chans=16)
    model = HRupsample(lr_chans=4, out_chans=3, upscale=4)
    i = torch.rand((2, 4, 64, 64))
    a = model(i)
    print(a.shape)
    #
    nparas = sum([p.numel() for p in model.parameters()])
    print('nparams: %.2f M'%(nparas/1e+6))
    print(model) # 0.01 M
