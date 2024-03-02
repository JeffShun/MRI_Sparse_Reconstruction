import torch
from custom.utils.mri_tools import *

class My_MC_DCNet(torch.nn.Module):
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
        random_sample_img, sensemap, random_sample_mask, full_sampling_kspace = inputs

        k_mask = random_sample_mask.unsqueeze(1).unsqueeze(-1)                           # [B, 1, 320, 320, 1]
        input_x = ifft2c_new(torch.view_as_real(full_sampling_kspace) * k_mask)          # [B, 15, 320, 320, 2]
        i_mask = (torch.abs(random_sample_img.unsqueeze(1))>0).float()                   # [B, 1, 320, 320]
        x = torch.cat((input_x[...,0],input_x[...,1]), 1)                                # [B, 30, 320, 320]

        y = full_sampling_kspace * random_sample_mask.unsqueeze(1)            # [B, 15, 320, 320] -complex64
        k_mask = random_sample_mask                                           # [B, 320, 320]  -float32
        i_mask = (torch.abs(random_sample_img.unsqueeze(1))>0).float()        # [B, 1, 320, 320]
        
        x_k = x*i_mask
        for k in range(self.num_iter_total):
            #dw 
            z_k = self.gradR[k%self.num_iter](x_k)                            # (B, 30, nrow, ncol)
            #dc
            x_k = self.gradD[k%self.num_iter](z_k, y, k_mask)                 # (B, 30, nrow, ncol)
        return x_k


    def forwardPM(self, inputs):
        random_sample_img, sensemap, random_sample_mask, full_sampling_kspace = inputs
        
        full_sampling_kspace = torch.view_as_real(full_sampling_kspace)                  # [B, 15, 320, 320, 2]
        k_mask = random_sample_mask.unsqueeze(1).unsqueeze(-1)                           # [B, 1, 320, 320, 1]
        input_x = ifft2c_new(full_sampling_kspace * k_mask)                              # [B, 15, 320, 320, 2]
        i_mask = (torch.abs(random_sample_img.unsqueeze(1))>0).float()                   # [B, 1, 320, 320]
        x0 = torch.cat((input_x[...,0],input_x[...,1]), 1)                               # [B, 30, 320, 320]
        
        k_mask = random_sample_mask                                           # [B, 320, 320]  -float32
        i_mask = (torch.abs(random_sample_img.unsqueeze(1))>0).float()        # [B, 1, 320, 320]
        x0 *= i_mask

        """
        :x0: zero-filled reconstruction (B, 2, nrow, ncol) - float32
        :csm: coil sensitivity map (B, ncoil, nrow, ncol) - complex64
        :mask: sampling mask (B, nrow, ncol) - int8
        """
        x_k = x0.clone()
        for k in range(self.num_iter_total):
            #dw 
            z_k = self.gradR[k%self.num_iter](x_k)                 # (B, 2, nrow, ncol)
            #dc
            x_k = self.gradD[k%self.num_iter](z_k, x0, k_mask)  # (B, 2, nrow, ncol)

        return x_k
    

  