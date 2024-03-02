import torch
import torch.nn as nn
from custom.utils.mri_tools import *

class DataPMLayer(nn.Module):
    flag = "PM"
    def __init__(self, lambda_init, learnable=True):
        super(DataPMLayer, self).__init__()
        self.lam = nn.Parameter(torch.tensor(lambda_init), requires_grad=learnable)

    def forward(self, z_k, x0, csm, mask):
        rhs = x0 + self.lam * z_k 
        rec = self.myCG(rhs, csm, mask)
        return rec

    def myAtA(self, im, csm, mask):
        """
        AtA: a func. that contains csm, mask and lambda and operates forward model
        :im: complex image (B, nrow, ncol)
        """
        im_coil = csm * im                                                           # split coil images (B x ncoil x nrow x ncol)
        k_full = torch.view_as_complex(fft2c_new(torch.view_as_real(im_coil)))       # convert into k-space 
        k_u = k_full * mask.unsqueeze(1)                                             # undersampling
        im_u_coil = torch.view_as_complex(ifft2c_new(torch.view_as_real(k_u)))       # convert into image domain
        im_u = torch.sum(im_u_coil * csm.conj(), axis=1, keepdim=True)               # coil combine (B x nrow x ncol)
        return im_u + self.lam * im        

    def myCG(self, rhs, csm, mask):
        """
        performs CG algorithm
        refer: https://lusongno1.blog.csdn.net/article/details/78550803?spm=1001.2101.3001.6650.4
        """
        rhs = r2c(rhs, axis=1) # (B, ncoil, nrow, ncol)
        x = torch.zeros_like(rhs)
        i, r, p = 0, rhs, rhs
        rTr = torch.sum(r.conj()*r, dim=(-1,-2),keepdim=True).real
        while i < 10 and torch.all(rTr > 1e-10):
            Ap = self.myAtA(p, csm, mask)
            alpha = rTr / torch.sum(p.conj()*Ap,dim=(-1,-2),keepdim=True).real
            alpha = alpha
            x = x + alpha * p
            r = r - alpha * Ap
            rTrNew = torch.sum(r.conj()*r,dim=(-1,-2),keepdim=True).real
            beta = rTrNew / rTr
            beta = beta
            p = r + beta * p
            i += 1
            rTr = rTrNew
        return c2r(x, axis=1, expand_dim=False)
    

class DataGDLayer(nn.Module):
    flag = "GD"
    def __init__(self, lambda_init, learnable=True):
        super(DataGDLayer, self).__init__()
        self.lam = nn.Parameter(torch.tensor(lambda_init), requires_grad=learnable)

    def forward(self, x, y, csm, mask):
        x = r2c(x, axis=1)                                                           
        im_coil = csm * x                                                            # split coil images (B x ncoil x nrow x ncol)
        k_full = torch.view_as_complex(fft2c_new(torch.view_as_real(im_coil)))       # convert into k-space 
        A_x_y = k_full * mask.unsqueeze(1) - y                                       # undersampling
        im_u_coil = torch.view_as_complex(ifft2c_new(torch.view_as_real(A_x_y)))     # convert into image domain
        gradD_x = torch.sum(im_u_coil * csm.conj(), axis=1, keepdim=True)            # coil combine (B x nrow x ncol)
        xg = x - self.lam * gradD_x
        return c2r(xg, axis=1, expand_dim=False)

"""
Multi-Channel Version 
"""

class MC_DataPMLayer(nn.Module):
    flag = "PM"
    def __init__(self, lambda_init, learnable=True):
        super(MC_DataPMLayer, self).__init__()
        self.lam = nn.Parameter(torch.tensor(lambda_init), requires_grad=learnable)

    def forward(self, z_k, x0, mask):
        rhs = x0 + self.lam * z_k 
        rec = self.myCG(rhs, mask)
        return rec

    def myAtA(self, im, mask):
        """
        AtA: a func. that contains csm, mask and lambda and operates forward model
        :im: complex image (B, nrow, ncol)
        """
        k_full = torch.view_as_complex(fft2c_new(torch.view_as_real(im)))       # convert into k-space 
        k_u = k_full * mask.unsqueeze(1)                                        # undersampling
        im_u = torch.view_as_complex(ifft2c_new(torch.view_as_real(k_u)))       # convert into image domain
        return im_u + self.lam * im        

    def myCG(self, rhs, mask):
        """
        performs CG algorithm
        refer: https://lusongno1.blog.csdn.net/article/details/78550803?spm=1001.2101.3001.6650.4
        """
        rhs = r2c(rhs, axis=1) # (B, ncoil, nrow, ncol)
        x = torch.zeros_like(rhs)
        i, r, p = 0, rhs, rhs
        rTr = torch.sum(r.conj()*r, dim=(-1,-2),keepdim=True).real
        while i < 10 and torch.all(rTr > 1e-10):
            Ap = self.myAtA(p, mask)
            alpha = rTr / torch.sum(p.conj()*Ap,dim=(-1,-2),keepdim=True).real
            alpha = alpha
            x = x + alpha * p
            r = r - alpha * Ap
            rTrNew = torch.sum(r.conj()*r,dim=(-1,-2),keepdim=True).real
            beta = rTrNew / rTr
            beta = beta
            p = r + beta * p
            i += 1
            rTr = rTrNew
        return c2r(x, axis=1,expand_dim=False)
    

class MC_DataGDLayer(nn.Module):
    flag = "GD"
    def __init__(self, lambda_init, learnable=True):
        super(MC_DataGDLayer, self).__init__()
        self.lam = nn.Parameter(torch.tensor(lambda_init), requires_grad=learnable)

    def forward(self, x, y, mask):
        x = r2c(x, axis=1)                                                      
        k_full = torch.view_as_complex(fft2c_new(torch.view_as_real(x)))           # convert into k-space 
        A_x_y = k_full * mask.unsqueeze(1) - y                                     # undersampling and grad
        gradD_x = torch.view_as_complex(ifft2c_new(torch.view_as_real(A_x_y)))     # convert into image domain
        xg = x - self.lam * gradD_x
        return c2r(xg, axis=1, expand_dim=False)


