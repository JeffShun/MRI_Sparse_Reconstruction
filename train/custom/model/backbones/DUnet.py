
import torch
import torch.nn as nn
import numpy as np
from custom.utils.mri_tools import *

class _Residual_Block(nn.Module):
    def __init__(self, num_chans=64):
        super(_Residual_Block, self).__init__()
        bias = True
        #res1
        self.conv1 = nn.Conv2d(num_chans, num_chans, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(num_chans, num_chans, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu4 = nn.PReLU()
        #res1
        #concat1

        self.conv5 = nn.Conv2d(num_chans, num_chans * 2, kernel_size=3, stride=2, padding=1, bias=bias)
        self.relu6 = nn.PReLU()

        #res2
        self.conv7 = nn.Conv2d(num_chans * 2, num_chans * 2, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu8 = nn.PReLU()
        #res2
        #concat2

        self.conv9 = nn.Conv2d(num_chans * 2, num_chans * 4, kernel_size=3, stride=2, padding=1, bias=bias)
        self.relu10 = nn.PReLU()

        #res3
        self.conv11 = nn.Conv2d(num_chans * 4, num_chans * 4, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu12 = nn.PReLU()
        #res3

        self.conv13 = nn.Conv2d(num_chans * 4, num_chans * 8, kernel_size=1, stride=1, padding=0, bias=bias)
        self.up14 = nn.PixelShuffle(2)

        #concat2
        self.conv15 = nn.Conv2d(num_chans * 4, num_chans * 2, kernel_size=1, stride=1, padding=0, bias=bias)
        #res4
        self.conv16 = nn.Conv2d(num_chans * 2, num_chans * 2, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu17 = nn.PReLU()
        #res4

        self.conv18 = nn.Conv2d(num_chans * 2, num_chans * 4, kernel_size=1, stride=1, padding=0, bias=bias)
        self.up19 = nn.PixelShuffle(2)

        #concat1
        self.conv20 = nn.Conv2d(num_chans * 2, num_chans, kernel_size=1, stride=1, padding=0, bias=bias)
        #res5
        self.conv21 = nn.Conv2d(num_chans, num_chans, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu22 = nn.PReLU()
        self.conv23 = nn.Conv2d(num_chans, num_chans, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu24 = nn.PReLU()
        #res5

        self.conv25 = nn.Conv2d(num_chans, num_chans, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        res1 = x
        out = self.relu4(self.conv3(self.relu2(self.conv1(x))))
        out = torch.add(res1, out)
        cat1 = out

        out = self.relu6(self.conv5(out))
        res2 = out
        out = self.relu8(self.conv7(out))
        out = torch.add(res2, out)
        cat2 = out

        out = self.relu10(self.conv9(out))
        res3 = out

        out = self.relu12(self.conv11(out))
        out = torch.add(res3, out)

        out = self.up14(self.conv13(out))

        out = torch.cat([out, cat2], 1)
        out = self.conv15(out)
        res4 = out
        out = self.relu17(self.conv16(out))
        out = torch.add(res4, out)

        out = self.up19(self.conv18(out))

        out = torch.cat([out, cat1], 1)
        out = self.conv20(out)
        res5 = out
        out = self.relu24(self.conv23(self.relu22(self.conv21(out))))
        out = torch.add(res5, out)

        out = self.conv25(out)
        out = torch.add(out, res1)

        return out


class DIDN(nn.Module):
    """
    Deep Iterative Down-Up Network, NTIRE denoising challenge winning entry

    Source: http://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Yu_Deep_Iterative_Down-Up_CNN_for_Image_Denoising_CVPRW_2019_paper.pdfp

    """
    def __init__(self, in_chans, out_chans, num_chans=64, global_residual=True, n_res_blocks=6):
        super().__init__()
        self.global_residual = global_residual
        bias=True
        self.conv_input = nn.Conv2d(in_chans, num_chans, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu1 = nn.PReLU()
        self.conv_down = nn.Conv2d(num_chans, num_chans, kernel_size=3, stride=2, padding=1, bias=bias)
        self.relu2 = nn.PReLU()

        self.n_res_blocks = n_res_blocks
        recursive = []
        for i in range(self.n_res_blocks):
            recursive.append(_Residual_Block(num_chans))
        self.recursive = torch.nn.ModuleList(recursive)

        self.conv_mid = nn.Conv2d(num_chans * self.n_res_blocks, num_chans, kernel_size=1, stride=1, padding=0, bias=bias)
        self.relu3 = nn.PReLU()
        self.conv_mid2 = nn.Conv2d(num_chans, num_chans, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu4 = nn.PReLU()

        self.subpixel = nn.PixelShuffle(2)
        self.conv_output = nn.Conv2d(num_chans // 4, out_chans, kernel_size=3, stride=1, padding=1, bias=bias)


    def forward(self, x):

        residual = x
        out = self.relu1(self.conv_input(x))
        out = self.relu2(self.conv_down(out))

        recons = []
        for i in range(self.n_res_blocks):
            out = self.recursive[i](out)
            recons.append(out)

        out = torch.cat(recons, 1)

        out = self.relu3(self.conv_mid(out))
        residual2 = out
        out = self.relu4(self.conv_mid2(out))
        out = torch.add(out, residual2)

        out= self.subpixel(out)
        out = self.conv_output(out)

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

class Dunet(torch.nn.Module):
    """
    Sensitivity network with data term based on forward and adjoint containing
    the sensitivity maps

    """
    def __init__(
        self,
        num_iter,
        model,
        model_config,
        datalayer,
        datalayer_config,
        shared_params=True,
        save_space=True,
        reset_cache=False,
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
            DenoisingWrapper(model(**model_config))
            for i in range(self.num_iter)
        ])
        self.gradD = torch.nn.ModuleList([
            datalayer(**datalayer_config)
            for i in range(self.num_iter)
        ])

        self.save_space = save_space
        if self.save_space:
            self.forward = self.forward_save_space
        self.reset_cache = reset_cache

    def forward(self, inputs):
        random_sample_img, random_sample_mask, full_sampling_kspace = inputs
        x = torch.view_as_real(random_sample_img)                # [B, 15, 320, 320, 2]
        y = torch.view_as_real(full_sampling_kspace)             # [B, 15, 320, 320, 2]
        mask = random_sample_mask.unsqueeze(1).unsqueeze(-1)     # [B, 1, 320, 320, 1]
        y*=mask

        x_all = [x]
        x_half_all = []
        if self.shared_params:
            num_iter = self.num_iter_total
        else:
            num_iter = min(np.where(self.is_trainable)[0][-1] + 1, self.num_iter)

        for i in range(num_iter):
            x_thalf = x - self.gradR[i%self.num_iter](x)
            x = self.gradD[i%self.num_iter](x_thalf, y, mask)
            x_all.append(x)
            x_half_all.append(x_thalf)
        out = torch.view_as_complex(x_all[-1])
        return out

    def forward_save_space(self, inputs):
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

            # would run out of memory at test time
            # if this is False for some cases
            if self.reset_cache:
                torch.cuda.empty_cache()
                torch.backends.cuda.cufft_plan_cache.clear()
        out = c2r(torch.view_as_complex(x), axis=1)
        return out


