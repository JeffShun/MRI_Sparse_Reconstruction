import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation
    )

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, bypass=None):
        super(BasicBlock, self).__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv_block = nn.Sequential(
            conv3x3(inplanes, planes, stride),
            nn.BatchNorm2d(planes),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            conv3x3(planes, planes),
            nn.BatchNorm2d(planes)
        )
        self.bypass = bypass
        self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        identity = x
        
        out = self.conv_block(x)

        if self.bypass is not None:
            identity = self.bypass(x)

        out += identity
        out = self.activate(out)

        return out


def make_res_layer(inplanes, planes, blocks, stride=1):
    bypass = nn.Sequential(
        conv1x1(inplanes, planes, stride),
        nn.BatchNorm2d(planes),
    )

    layers = []
    layers.append(BasicBlock(inplanes, planes, stride, bypass))
    for _ in range(1, blocks):
        layers.append(BasicBlock(planes, planes))

    return nn.Sequential(*layers)


class DoubleConv(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=int(kernel_size / 2)),
            nn.BatchNorm2d(out_ch), 
            nn.LeakyReLU(negative_slope=0.2, inplace=True), 
            nn.Conv2d(out_ch, out_ch, 3, padding=1, dilation=1),
            nn.BatchNorm2d(out_ch), 
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class FCN(nn.Module):

    def __init__(self, in_chans, out_chans, num_chans=32, n_res_blocks=2):
        super(FCN, self).__init__()

        self.layer1 = make_res_layer(in_chans, num_chans, n_res_blocks, stride=2)
        self.layer2 = make_res_layer(num_chans, num_chans * 2, n_res_blocks, stride=2)
        self.layer3 = make_res_layer(num_chans * 2, num_chans * 4, n_res_blocks, stride=2)
        self.layer4 = make_res_layer(num_chans * 4, num_chans * 8, n_res_blocks, stride=2)
      
        self.end = conv1x1(num_chans * 8, out_chans)
        self.up = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)
        
    def forward(self, x):
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        out = self.up(self.end(c4))
        
        return out