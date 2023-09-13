
import torch
import torch.nn as nn
import numpy as np
from custom.utils.mri_tools import *

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

    def __init__(self, in_ch, channels=32, outchannel=2, blocks=2, global_residual=True):
        super(ResUnet, self).__init__()
        self.global_residual = global_residual

        self.layer1 = make_res_layer(in_ch, channels, blocks, stride=1)
        self.layer2 = make_res_layer(channels, channels * 2, blocks, stride=2)
        self.layer3 = make_res_layer(channels * 2, channels * 4, blocks, stride=2)
        self.layer4 = make_res_layer(channels * 4, channels * 8, blocks, stride=2)
        self.layer5 = make_res_layer(channels * 8, channels * 16, blocks, stride=2)

        self.up5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.mconv4 = DoubleConv(channels * 24, channels * 8)
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.mconv3 = DoubleConv(channels * 12, channels * 4)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.mconv2 = DoubleConv(channels * 6, channels * 2)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.mconv1 = DoubleConv(channels * 3, channels)
        self.end = conv1x1(channels, outchannel)
        
    def forward(self, x):
        residual = x
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
            out = torch.add(out, residual)

        return out


class DataGDLayer(nn.Module):
    """
        DataLayer computing the gradient on the L2 dataterm.
    """
    def __init__(self, lambda_init, learnable=True):
        """
        Args:
            lambda_init (float): Init value of data term weight lambda.
        """
        super(DataGDLayer, self).__init__()
        self.lambda_init = lambda_init
        self.data_weight = torch.nn.Parameter(torch.Tensor(1))
        self.data_weight.data = torch.tensor(
            lambda_init,
            dtype=self.data_weight.dtype,
        )
        self.data_weight.requires_grad = learnable

    def forward(self, x, y, mask):
        A_x_y = fft2c_new(x) * mask - y
        gradD_x = ifft2c_new(A_x_y * mask)
        return x - self.data_weight * gradD_x

    def __repr__(self):
        return f'DataLayer(lambda_init={self.data_weight.item():.4g})'
    
class DenoisingWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input):
        # re-shape data from [nBatch, nCoil, nFE, nPE, 2]
        # to [nBatch, nCoil*2, nFE, nPE]
        nCoil = input.shape[1]
        output = torch.cat((input[...,0], input[...,1]), 1)

        # apply denoising
        output = self.model(output)

        # re-shape data from [nBatch, nCoil*2, nFE, nPE]
        # to [nBatch, nCoil, nFE, nPE, 2]
        output = torch.stack((output[:,:nCoil], output[:,nCoil:]),-1)

        return output

class MC_DUnet(torch.nn.Module):
    """
    Sensitivity network with data term based on forward and adjoint containing
    the sensitivity maps

    """
    def __init__(
        self,
        in_ch=30,
        channels=64,
        outchannel=30,
        blocks=2,
        global_residual=True,
        num_iter=8,
        shared_params=True,
        lambda_init = 0.05
    ):
        super().__init__()

        self.shared_params = shared_params

        if self.shared_params:
            self.num_iter = 1
        else:
            self.num_iter = num_iter

        self.num_iter_total = num_iter

        self.is_trainable = [True] * num_iter

        # setup the modules
        self.gradR = torch.nn.ModuleList([
            DenoisingWrapper(ResUnet(in_ch, channels, outchannel, blocks, global_residual))
            for i in range(self.num_iter)
        ])
        self.gradD = torch.nn.ModuleList([
            DataGDLayer(lambda_init, learnable=True) for i in range(self.num_iter)
        ])

    def forward(self, inputs):
        random_sample_img, random_sample_mask, full_sampling_kspace = inputs
        x = torch.view_as_real(random_sample_img)                # [B, 15, 320, 320, 2]
        y = torch.view_as_real(full_sampling_kspace)             # [B, 15, 320, 320, 2]
        mask = random_sample_mask.unsqueeze(1).unsqueeze(-1)     # [B, 1, 320, 320, 1]
        y*=mask

        if self.shared_params:
            num_iter = self.num_iter_total
        else:
            num_iter = min(np.where(self.is_trainable)[0][-1] + 1, self.num_iter)
        
        for i in range(num_iter):
            x_thalf = x - self.gradR[i%self.num_iter](x)
            x = self.gradD[i%self.num_iter](x_thalf, y, mask)

        out = c2r(torch.view_as_complex(x), axis=1)
        return out


