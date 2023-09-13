
import torch
import torch.nn as nn
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

    def __init__(self, in_ch, out_chans, num_chans=32, n_res_blocks=2, global_residual=True):
        super(ResUnet, self).__init__()
        self.global_residual = global_residual

        self.layer1 = make_res_layer(in_ch, num_chans, n_res_blocks, stride=1)
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
    flag = "GD"
    def __init__(self, lambda_init, learnable=True):
        super(DataGDLayer, self).__init__()
        self.lam = nn.Parameter(torch.tensor(lambda_init), requires_grad=learnable)

    def forward(self, x, y, mask):
        A_x_y = fft2c_new(x) * mask - y
        gradD_x = ifft2c_new(A_x_y * mask)
        return x - self.lam * gradD_x
    

class DataPMLayer(nn.Module):
    flag = "PM"
    def __init__(self, lambda_init, learnable=True):
        super(DataPMLayer, self).__init__()
        self.lam = nn.Parameter(torch.tensor(lambda_init), requires_grad=learnable)

    def forward(self, z_k, x0, mask):
        rhs = x0 + self.lam * z_k 
        rec = self.myCG(rhs, mask, self.lam)
        return rec

    def myAtA(self, im, mask, lam):
        """
        AtA: a func. that contains csm, mask and lambda and operates forward model
        :im: complex image (B, ncoil, nrow, ncol)
        """
        k_full = torch.view_as_complex(fft2c_new(torch.view_as_real(im))) # convert into k-space 
        k_u = k_full * mask.unsqueeze(1) # undersampling
        im_u = torch.view_as_complex(ifft2c_new(torch.view_as_real(k_u))) # convert into image domain
        return im_u + lam * im        

    def myCG(self, rhs, mask, lam):
        """
        performs CG algorithm
        refer: https://lusongno1.blog.csdn.net/article/details/78550803?spm=1001.2101.3001.6650.4
        """
        rhs = r2c(rhs, axis=1) # (B, ncoil, nrow, ncol)
        x = torch.zeros_like(rhs)
        i, r, p = 0, rhs, rhs
        rTr = torch.sum(r.conj()*r,dim=(-1,-2,-3),keepdim=True).real
        while i < 10 and torch.all(rTr > 1e-10):
            Ap = self.myAtA(p, mask, lam)
            alpha = rTr / torch.sum(p.conj()*Ap,dim=(-1,-2,-3),keepdim=True).real
            alpha = alpha
            x = x + alpha * p
            r = r - alpha * Ap
            rTrNew = torch.sum(r.conj()*r,dim=(-1,-2,-3),keepdim=True).real
            beta = rTrNew / rTr
            beta = beta
            p = r + beta * p
            i += 1
            rTr = rTrNew
        return c2r(x, axis=1)

    def __repr__(self):
        return 'DataPMLayer'

class MC_DUnet(torch.nn.Module):
    def __init__(
        self,
        model,
        model_config,
        datalayer,
        datalayer_config,
        num_iter=10,
        shared_params=True,  
    ):
        super().__init__()

        self.shared_params = shared_params

        if self.shared_params:
            self.num_iter = 1
        else:
            self.num_iter = num_iter

        self.num_iter_total = num_iter
        self.dc = datalayer.flag
        
        # setup the modules
        self.gradR = torch.nn.ModuleList([
            model(**model_config)
            for i in range(self.num_iter)
        ])
        self.gradD = torch.nn.ModuleList([
            datalayer(**datalayer_config)
            for i in range(self.num_iter)
        ])

    def forward(self, inputs):
        if self.dc == "GD":
            return self.forwardGD(inputs)
        else:
            return self.forwardPM(inputs)


    def forwardGD(self, inputs):
        random_sample_img, random_sample_mask, full_sampling_kspace = inputs
        x = torch.view_as_real(random_sample_img)                # [B, 15, 320, 320, 2]
        y = torch.view_as_real(full_sampling_kspace)             # [B, 15, 320, 320, 2]
        mask = random_sample_mask.unsqueeze(1).unsqueeze(-1)     # [B, 1, 320, 320, 1]
        y*=mask
        
        for k in range(self.num_iter_total):
            x_thalf = self.reshapeWrapper(self.gradR[k%self.num_iter], x)
            x = self.gradD[k%self.num_iter](x_thalf, y, mask)

        out = c2r(torch.view_as_complex(x), axis=1)
        return out


    def forwardPM(self, inputs):
        random_sample_img, random_sample_mask, full_sampling_kspace = inputs
        
        x0 = c2r(random_sample_img, axis=1)
        mask = random_sample_mask
        """
        :x0: zero-filled reconstruction (B, ncoil*2, nrow, ncol) - float32
        :mask: sampling mask (B, nrow, ncol) - int8
        """
        x_k = x0.clone()
        for k in range(self.num_iter_total):
            #dw 
            z_k = self.gradR[k%self.num_iter](x_k) # (B, 2*ncoil, nrow, ncol)
            #dc
            x_k = self.gradD[k%self.num_iter](z_k, x0, mask)  # (B, 2*ncoil, nrow, ncol)

        return x_k
    
    def reshapeWrapper(self, model, input):
        # re-shape data from [nBatch, nCoil, nFE, nPE, 2]
        # to [nBatch, nCoil*2, nFE, nPE]
        nCoil = input.shape[1]
        output = torch.cat((input[...,0], input[...,1]), 1)

        # apply denoising
        output = model(output)

        # re-shape data from [nBatch, nCoil*2, nFE, nPE]
        # to [nBatch, nCoil, nFE, nPE, 2]
        output = torch.stack((output[:,:nCoil], output[:,nCoil:]),-1)

        return output        