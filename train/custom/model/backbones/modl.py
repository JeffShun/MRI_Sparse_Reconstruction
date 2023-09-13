import torch
import torch.nn as nn
from custom.utils.mri_tools import *

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

        self.down5 = nn.Sequential(
            DWT(),
            nn.Conv2d(num_chans*4, num_chans * 2, kernel_size=3, stride=1, padding=1, bias=bias)
        )
        # self.down5 = nn.Conv2d(num_chans, num_chans * 2, kernel_size=3, stride=2, padding=1, bias=bias)
        self.relu6 = nn.PReLU()

        #res2
        self.conv7 = nn.Conv2d(num_chans * 2, num_chans * 2, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu8 = nn.PReLU()
        #res2
        #concat2

        self.down9 = nn.Sequential(
            DWT(),
            nn.Conv2d(num_chans*8, num_chans * 4, kernel_size=3, stride=1, padding=1, bias=bias)
        )
        # self.down9 = nn.Conv2d(num_chans * 2, num_chans * 4, kernel_size=3, stride=2, padding=1, bias=bias)
        self.relu10 = nn.PReLU()

        #res3
        self.conv11 = nn.Conv2d(num_chans * 4, num_chans * 4, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu12 = nn.PReLU()
        #res3

        self.conv13 = nn.Conv2d(num_chans * 4, num_chans * 8, kernel_size=1, stride=1, padding=0, bias=bias)
        # self.up14 = nn.PixelShuffle(2)
        self.up14 = IWT()


        #concat2
        self.conv15 = nn.Conv2d(num_chans * 4, num_chans * 2, kernel_size=1, stride=1, padding=0, bias=bias)
        #res4
        self.conv16 = nn.Conv2d(num_chans * 2, num_chans * 2, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu17 = nn.PReLU()
        #res4

        self.conv18 = nn.Conv2d(num_chans * 2, num_chans * 4, kernel_size=1, stride=1, padding=0, bias=bias)
        # self.up19 = nn.PixelShuffle(2)
        self.up19 = IWT()

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

        out = self.relu6(self.down5(out))
        res2 = out
        out = self.relu8(self.conv7(out))
        out = torch.add(res2, out)
        cat2 = out

        out = self.relu10(self.down9(out))
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

# CNN denoiser ======================
class MW_DIDN(nn.Module):
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
        # self.conv_down = nn.Conv2d(num_chans, num_chans, kernel_size=3, stride=2, padding=1, bias=bias)
        self.conv_down = nn.Sequential(
            DWT(),
            nn.Conv2d(num_chans*4, num_chans, kernel_size=3, stride=1, padding=1, bias=bias)
        )
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

        # self.up = nn.PixelShuffle(2)
        self.up = IWT()

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

        out= self.up(out)
        out = self.conv_output(out)

        if self.global_residual:
            out = torch.add(out, residual)
        return out


#CNN denoiser ======================
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

class cnn_denoiser(nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        layers = []
        layers += conv_block(30, 64)

        for _ in range(n_layers-2):
            layers += conv_block(64, 64)

        layers += nn.Sequential(
            nn.Conv2d(64, 30, 3, padding=1),
            nn.BatchNorm2d(30)
        )

        self.nw = nn.Sequential(*layers)
    
    def forward(self, x):
        idt = x # (B, ncoil*2, nrow, ncol)
        dw = self.nw(x) + idt # (B, ncoil*2, nrow, ncol)
        return dw

#CG algorithm ======================
class myAtA(nn.Module):
    """
    performs DC step
    """
    def __init__(self, mask, lam):
        super(myAtA, self).__init__()
        self.mask = mask # int (B x nrow x ncol)
        self.lam = lam 

    def forward(self, im): #step for batch image
        """
        :im: complex image (B, ncoil, nrow, ncol)
        """
        k_full = torch.view_as_complex(fft2c_new(torch.view_as_real(im))) # convert into k-space 
        k_u = k_full * self.mask.unsqueeze(1) # undersampling
        im_u = torch.view_as_complex(ifft2c_new(torch.view_as_real(k_u))) # convert into image domain
        return im_u + self.lam * im

def myCG(AtA, rhs):
    """
    performs CG algorithm
    :AtA: a class object that contains csm, mask and lambda and operates forward model
    refer: https://lusongno1.blog.csdn.net/article/details/78550803?spm=1001.2101.3001.6650.4
    """
    rhs = r2c(rhs, axis=1) # (B, ncoil, nrow, ncol)
    x = torch.zeros_like(rhs)
    i, r, p = 0, rhs, rhs
    rTr = torch.sum(r.conj()*r,dim=(-1,-2,-3),keepdim=True).real
    while i < 10 and torch.all(rTr > 1e-10):
        Ap = AtA(p)
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

class data_consistency(nn.Module):
    def __init__(self):
        super().__init__()
        self.lam = nn.Parameter(torch.tensor(0.05), requires_grad=True)

    def forward(self, z_k, x0, mask):
        rhs = x0 + self.lam * z_k 
        AtA = myAtA(mask, self.lam)
        rec = myCG(AtA, rhs)
        return rec

#model =======================    
class MoDL(nn.Module):
    def __init__(self, n_layers, k_iters):
        """
        :n_layers: number of layers
        :k_iters: number of iterations
        """
        super().__init__()
        self.k_iters = k_iters
        self.dw = cnn_denoiser(n_layers)
        self.dc = data_consistency()

    def forward(self, inputs):
        random_sample_img, random_sample_mask, full_sampling_kspace = inputs
        
        x0 = c2r(random_sample_img, axis=1)
        mask = random_sample_mask
        """
        :x0: zero-filled reconstruction (B, ncoil*2, nrow, ncol) - float32
        :mask: sampling mask (B, nrow, ncol) - int8
        """
        x_k = x0.clone()
        for k in range(self.k_iters):
            #dw 
            z_k = self.dw(x_k) # (B, ncoil*2, nrow, ncol)
            #dc
            x_k = self.dc(z_k, x0, mask) # (B, ncoil*2, nrow, ncol)
        return x_k
    
#model =======================    
class UnetMoDL(nn.Module):
    def __init__(self, in_chans, out_chans, num_chans=64, global_residual=True, n_res_blocks=6, k_iters=10):
        """
        :n_layers: number of layers
        :k_iters: number of iterations
        """
        super().__init__()
        self.k_iters = k_iters
        self.dw = MW_DIDN(in_chans, out_chans, num_chans, global_residual, n_res_blocks)
        self.dc = data_consistency()

    def forward(self, inputs):
        random_sample_img, random_sample_mask, full_sampling_kspace = inputs
        
        x0 = c2r(random_sample_img, axis=1)
        mask = random_sample_mask
        """
        :x0: zero-filled reconstruction (B, 2, nrow, ncol) - float32
        :mask: sampling mask (B, nrow, ncol) - int8
        """
        x_k = x0.clone()
        for k in range(self.k_iters):
            #dw 
            z_k = self.dw(x_k) # (B, 2, nrow, ncol)
            #dc
            x_k = self.dc(z_k, x0, mask) # (B, 2, nrow, ncol)
        return x_k