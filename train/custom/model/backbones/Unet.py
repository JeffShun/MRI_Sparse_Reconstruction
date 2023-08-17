import torch
import torch.nn as nn

def conv1x1(in_channels, out_channels):
    return nn.Conv2d(
        in_channels, 
        out_channels, 
        kernel_size=1,
        bias=True
        )

def conv3x3(in_channels, out_channels, stride=1, dilation=1):
    return nn.Conv2d(
        in_channels, 
        out_channels, 
        kernel_size=3,
        stride=stride,
        dilation=dilation,
        padding=dilation, 
        bias=True
        )


class DS(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DS, self).__init__()
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
    
    def forward(self, x):
        x = self.conv_down(x)
        return x

class US(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(US, self).__init__()
        self.conv_down = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
    
    def forward(self, x):
        x = self.conv_down(x)
        return x


class BBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels):
        super(BBlock, self).__init__()
        self.conv_block = nn.Sequential(
            conv3x3(in_channels, out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
    def forward(self, x):
        x = self.conv_block(x)
        return x

class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=[2, 1]):

        super(DBlock, self).__init__()
        self.conv_block = nn.Sequential(
            conv3x3(in_channels, out_channels, dilation=dilations[0]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            conv3x3(in_channels, out_channels, dilation=dilations[1]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x):
        out = self.conv_block(x)
        return out


class Unet(nn.Module):
    def __init__(self, in_ch, channels):
        super(Unet, self).__init__()

        self.head = BBlock(in_ch, channels)
        self.d_l0 = nn.Sequential(
            DBlock(channels, channels, dilations=[2, 1])
            )
        self.downsampling1 = DS(channels, channels * 4)


        self.d_l1 = nn.Sequential(
            BBlock(channels * 4, channels * 2),
            DBlock(channels * 2, channels * 2, dilations=[2, 1])
            )
        self.downsampling2 = DS(channels * 2, channels * 8)

        self.d_l2 = nn.Sequential(
            BBlock(channels * 8, channels * 4),
            DBlock(channels * 4, channels * 4, dilations=[2, 1])
            )
        self.downsampling3 = DS(channels * 4, channels * 16)

        self.pro_l3 = nn.Sequential(
            BBlock(channels * 16, channels * 8),
            DBlock(channels * 8, channels * 8, dilations=[2, 3]),
            DBlock(channels * 8, channels * 8, dilations=[3, 2]),
            BBlock(channels * 8, channels * 16)
            )

        self.upsampling1 = US(channels * 16, channels * 4)
        self.i_l2 = nn.Sequential(
            DBlock(channels * 4, channels * 4, dilations=[2, 1]),
            BBlock(channels * 4, channels * 8)
            )

        self.upsampling2 = US(channels * 8, channels * 2)
        self.i_l1 = nn.Sequential(
            DBlock(channels * 2, channels * 2, dilations=[2, 1]),
            BBlock(channels * 2, channels * 4)
            )

        self.upsampling3 = US(channels * 4, channels)
        self.i_l0 = nn.Sequential(
            DBlock(channels, channels, dilations=[2, 1])
            )


    def forward(self, x):
        x0 = self.d_l0(self.head(x))
        x1 = self.d_l1(self.downsampling1(x0))
        x2 = self.d_l2(self.downsampling2(x1))
        x_ = self.upsampling1(self.pro_l3(self.downsampling3(x2))) + x2
        x_ = self.upsampling2(self.i_l2(x_)) + x1
        x_ = self.upsampling3(self.i_l1(x_)) + x0
        out = self.i_l0(x_)
        return out

if __name__ == '__main__':
    model = Unet(30, 32).cuda()
    input = torch.rand(8,30,320,320).cuda()
    out = model(input)
    print(out.shape)
