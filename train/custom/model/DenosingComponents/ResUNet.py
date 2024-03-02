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

class ResUnet(nn.Module):

    def __init__(self, in_chans, out_chans, num_chans=32, n_res_blocks=2, global_residual=True):
        super(ResUnet, self).__init__()

        self.global_residual = global_residual
        self.layer1 = make_res_layer(in_chans, num_chans, n_res_blocks, stride=1)
        self.layer2 = make_res_layer(num_chans, num_chans * 2, n_res_blocks, stride=2)
        self.layer3 = make_res_layer(num_chans * 2, num_chans * 4, n_res_blocks, stride=2)
        self.layer4 = make_res_layer(num_chans * 4, num_chans * 8, n_res_blocks, stride=2)
        self.layer5 = make_res_layer(num_chans * 8, num_chans * 16, n_res_blocks, stride=2)

        self.up5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.mconv4 = DoubleConv(num_chans * 24, num_chans * 8)
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.mconv3 = DoubleConv(num_chans * 12, num_chans * 4)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.mconv2 = DoubleConv(num_chans * 6, num_chans * 2)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.mconv1 = DoubleConv(num_chans * 3, num_chans)
        self.end = conv1x1(num_chans, out_chans)
        
    def forward(self, x):
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        c5 = self.layer5(c4)

        merge4 = self.mconv4(torch.cat([self.up5(c5), c4], dim=1))
        merge3 = self.mconv3(torch.cat([self.up4(merge4), c3], dim=1))
        merge2 = self.mconv2(torch.cat([self.up3(merge3), c2], dim=1))
        merge1 = self.mconv1(torch.cat([self.up2(merge2), c1], dim=1))
        out = self.end(merge1)
        if self.global_residual:
            out = torch.sub(x, out)
        return out