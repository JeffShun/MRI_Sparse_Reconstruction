import torch
import torch.nn as nn
from custom.utils.fftc import *

def conv1x1(in_channels, out_channels):
    return nn.Conv2d(
        in_channels, 
        out_channels, 
        kernel_size=1,
        bias=False
        )

def conv3x3(in_channels, out_channels, dilation=1):
    return nn.Conv2d(
        in_channels, 
        out_channels, 
        kernel_size=3,
        dilation=dilation,
        padding=dilation, 
        bias=False
        )

def dwt_init(x):

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

def iwt_init(x):
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

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


class BBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels):
        super(BBlock, self).__init__()
        self.conv_block = nn.Sequential(
            conv3x3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
    def forward(self, x):
        x = self.conv_block(x)
        return x

class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=[2, 1]):

        super(DBlock, self).__init__()
        self.conv_block = nn.Sequential(
            conv3x3(in_channels, out_channels, dilation=dilations[0]),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            conv3x3(in_channels, out_channels, dilation=dilations[1]),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
    def forward(self, x):
        out = self.conv_block(x)
        return out


class MWCNN(nn.Module):
    def __init__(self, in_ch, channels):
        super(MWCNN, self).__init__()

        self.DWT = DWT()
        self.IWT = IWT()

        self.head = BBlock(in_ch, channels)
        self.d_l0 = nn.Sequential(
            DBlock(channels, channels, dilations=[2, 1])
            )

        self.d_l1 = nn.Sequential(
            BBlock(channels * 4, channels * 2),
            DBlock(channels * 2, channels * 2, dilations=[2, 1])
            )

        self.d_l2 = nn.Sequential(
            BBlock(channels * 8, channels * 4),
            DBlock(channels * 4, channels * 4, dilations=[2, 1])
            )

        self.pro_l3 = nn.Sequential(
            BBlock(channels * 16, channels * 8),
            DBlock(channels * 8, channels * 8, dilations=[2, 3]),
            DBlock(channels * 8, channels * 8, dilations=[3, 2]),
            BBlock(channels * 8, channels * 16)
            )

        self.i_l2 = nn.Sequential(
            DBlock(channels * 4, channels * 4, dilations=[2, 1]),
            BBlock(channels * 4, channels * 8)
            )

        self.i_l1 = nn.Sequential(
            DBlock(channels * 2, channels * 2, dilations=[2, 1]),
            BBlock(channels * 2, channels * 4)
            )

        self.i_l0 = nn.Sequential(
            DBlock(channels, channels, dilations=[2, 1])
            )
        
        self.end = conv3x3(channels, in_ch)
        self.bypass = conv3x3(in_ch, in_ch)

    def forward(self, x):
        x0 = self.d_l0(self.head(x))
        x1 = self.d_l1(self.DWT(x0))
        x2 = self.d_l2(self.DWT(x1))
        x_ = self.IWT(self.pro_l3(self.DWT(x2))) + x2
        x_ = self.IWT(self.i_l2(x_)) + x1
        x_ = self.IWT(self.i_l1(x_)) + x0
        out = self.end(self.i_l0(x_))
        return out + self.bypass(x)


class DCBlock(nn.Module):
    def __init__(self):
        super(DCBlock, self).__init__()

    def forward(self, x, feature, sample_mask): 
        k_feas = fft2c_new(feature)
        k_x = fft2c_new(x)
        sample_mask = sample_mask.unsqueeze(1).unsqueeze(1)
        dc_feas = k_x*sample_mask + k_feas*(1-sample_mask)
        

        # import matplotlib.pyplot as plt
        # import os
        # import numpy as np
        # os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
        # data_0 = torch.abs(k_x)[0,10,:,:].cpu().numpy() 
        # data_pre = torch.abs(k_feas)[0,10,:,:].cpu().numpy() 
        # data_now = torch.abs(dc_feas)[0,10,:,:].cpu().numpy() 

        # data_0_i = torch.abs(ifft2c_new(k_x))[0,10,:,:].cpu().numpy() 
        # data_pre_i = torch.abs(ifft2c_new(k_feas))[0,10,:,:].cpu().numpy() 
        # data_now_i = torch.abs(ifft2c_new(dc_feas))[0,10,:,:].cpu().numpy() 

        # plt.figure(1)
        # plt.subplot(2,3,1)
        # plt.imshow(np.clip(data_0,0,0.1))
        # plt.subplot(2,3,2)
        # plt.imshow(np.clip(data_pre,0,0.1))
        # plt.subplot(2,3,3)
        # plt.imshow(np.clip(data_now,0,0.1))

        # plt.subplot(2,3,4)
        # plt.imshow(data_0_i)
        # plt.subplot(2,3,5)
        # plt.imshow(data_pre_i)
        # plt.subplot(2,3,6)
        # plt.imshow(data_now_i)

        # plt.show()


        i_feas = ifft2c_new(dc_feas)
        return i_feas

class DC_MWCNN(nn.Module):
    def __init__(self, in_ch, channels, stages=4):
        super(DC_MWCNN, self).__init__()
        self.DC_MWCNN = []
        for _ in range(stages):
            self.DC_MWCNN.append(MWCNN(in_ch, channels))
            self.DC_MWCNN.append(DCBlock())
        self.dcn = nn.ModuleList(self.DC_MWCNN)
        self.stages = stages

    def forward(self, inputs):
        outputs = []
        x, sample_mask = inputs
        feas_0 = self.dcn[0](self.complex2channel(x))
        feas_1 = self.dcn[1](x, self.channel2complex(feas_0), sample_mask)
        # outputs.append(self.channel2complex(feas_0))
        outputs.append(feas_1)
        for i in range(1, self.stages):
            feas_0 = self.dcn[i*2](self.complex2channel(feas_1))
            feas_1 = self.dcn[i*2+1](x, self.channel2complex(feas_0), sample_mask)
            # outputs.append(self.channel2complex(feas_0))
            outputs.append(feas_1)
        return outputs

    def channel2complex(self, data):
        b, nc, nx, ny = data.shape
        real = data[:,:nc//2,:,:]
        imag = data[:,nc//2:,:,:]
        data_out = real + 1j*imag
        return data_out

    def complex2channel(self, data):
        real = data.real
        imag = data.imag
        data_out = torch.cat((real,imag),1)
        return data_out
    

if __name__ == '__main__':
    model = DC_MWCNN(30, 32).cuda()
    input = torch.rand(8,30,320,320).cuda()
    sample_mask = torch.rand(8,320).cuda()
    out = model(input, sample_mask)
    print(out.shape)
