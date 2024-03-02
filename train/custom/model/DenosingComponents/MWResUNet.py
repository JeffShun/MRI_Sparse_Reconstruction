import torch
import torch.nn as nn

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return self.dwt_init(x)

    def dwt_init(self, x):

        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4

        return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return self.iwt_init(x)

    def iwt_init(self, x):
        r = 2
        in_batch, in_channel, in_height, in_width = x.size()
        out_batch, out_channel, out_height, out_width = in_batch, int(
            in_channel / (r ** 2)), r * in_height, r * in_width
        x1 = x[:, 0:out_channel, :, :] / 2
        x2 = x[:, out_channel:out_channel * 2, :, :] / 2
        x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
        x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
        
        h = torch.zeros([out_batch, out_channel, out_height, out_width]).to(x.dtype).to(x.device)

        h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
        h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
        h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
        h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
        return h

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

        if stride == 1:
            self.conv_block = nn.Sequential(
                conv3x3(inplanes, planes),
                nn.BatchNorm2d(planes),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                conv3x3(planes, planes),
                nn.BatchNorm2d(planes)
            )
        elif stride == 2:
            self.conv_block = nn.Sequential(
                DWT(),
                conv3x3(inplanes*4, planes),
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
    if stride == 1:
        bypass = nn.Sequential(
            conv1x1(inplanes, planes),
            nn.BatchNorm2d(planes),
        )        

    elif stride == 2:
        bypass = nn.Sequential(
            DWT(),
            conv1x1(inplanes*4, planes),
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

class MWResUnet(nn.Module):

    def __init__(self, in_chans, out_chans, num_chans=32, n_res_blocks=2, global_residual=True):
        super(MWResUnet, self).__init__()

        self.global_residual = global_residual
        self.layer1 = make_res_layer(in_chans, num_chans, n_res_blocks, stride=1)
        self.layer2 = make_res_layer(num_chans, num_chans * 2, n_res_blocks, stride=2)
        self.layer3 = make_res_layer(num_chans * 2, num_chans * 4, n_res_blocks, stride=2)
        self.layer4 = make_res_layer(num_chans * 4, num_chans * 8, n_res_blocks, stride=2)
        self.layer5 = make_res_layer(num_chans * 8, num_chans * 16, n_res_blocks, stride=2)

        self.up5 = nn.Sequential(
            nn.Conv2d(num_chans*16, num_chans * 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_chans * 32),
            IWT()
        )
        self.mconv4 = DoubleConv(num_chans * 16, num_chans * 8)
        self.up4 = nn.Sequential(
            nn.Conv2d(num_chans*8, num_chans * 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_chans * 16),
            IWT()
        )
        self.mconv3 = DoubleConv(num_chans * 8, num_chans * 4)
        self.up3 = nn.Sequential(
            nn.Conv2d(num_chans*4, num_chans * 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_chans * 8),
            IWT()
        )
        self.mconv2 = DoubleConv(num_chans * 4, num_chans * 2)
        self.up2 = nn.Sequential(
            nn.Conv2d(num_chans*2, num_chans * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_chans * 4),
            IWT()
        )
        self.mconv1 = DoubleConv(num_chans * 2, num_chans)
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