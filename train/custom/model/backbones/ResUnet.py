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
            nn.InstanceNorm2d(planes),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            conv3x3(planes, planes),
            nn.InstanceNorm2d(planes)
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
        nn.InstanceNorm2d(planes),
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
            nn.InstanceNorm2d(out_ch), 
            nn.LeakyReLU(negative_slope=0.2, inplace=True), 
            nn.Conv2d(out_ch, out_ch, 3, padding=1, dilation=1),
            nn.InstanceNorm2d(out_ch), 
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class ResUnet(nn.Module):

    def __init__(self, in_ch, channels=32, blocks=2):
        super(ResUnet, self).__init__()

        self.layer1 = make_res_layer(in_ch, channels * 2, blocks, stride=1)
        self.layer2 = make_res_layer(channels * 2, channels * 4, blocks, stride=2)
        self.layer3 = make_res_layer(channels * 4, channels * 8, blocks, stride=2)
        self.layer4 = make_res_layer(channels * 8, channels * 16, blocks, stride=2)
        self.layer5 = make_res_layer(channels * 16, channels * 32, blocks, stride=2)

        self.up5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.mconv4 = DoubleConv(channels * 48, channels * 16)
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.mconv3 = DoubleConv(channels * 24, channels * 8)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.mconv2 = DoubleConv(channels * 12, channels * 4)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.mconv1 = DoubleConv(channels * 6, channels)
        
    def forward(self, input):
        c1 = self.layer1(input)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        c5 = self.layer5(c4)

        merge4 = self.mconv4(torch.cat([self.up5(c5), c4], dim=1))
        merge3 = self.mconv3(torch.cat([self.up4(merge4), c3], dim=1))
        merge2 = self.mconv2(torch.cat([self.up3(merge3), c2], dim=1))
        merge1 = self.mconv1(torch.cat([self.up2(merge2), c1], dim=1))

        return merge1



if __name__ == '__main__':
    model = ResUnet(1).cuda()
    print(model)

