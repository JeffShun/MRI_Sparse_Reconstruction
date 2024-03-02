import torch
from custom.utils.mri_tools import *

class My_DCNet(torch.nn.Module):
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
        x0 = c2r(random_sample_img, axis=1)                                   # [B, 2, 320, 320]  -float32
        y = full_sampling_kspace * random_sample_mask.unsqueeze(1)            # [B, 15, 320, 320] -complex64
        csm = sensemap                                                        # [B, 15, 320, 320] -complex64
        mask = random_sample_mask                                             # [B, 320, 320]  -float32
        
        x_k = x0.clone()
        for k in range(self.num_iter_total):
            #dw 
            z_k = self.gradR[k%self.num_iter](x_k)                            # (B, 2, nrow, ncol)
            #dc
            x_k = self.gradD[k%self.num_iter](z_k, y, csm, mask)              # (B, 2, nrow, ncol)
        out = r2c(x_k, axis=1)
        return out


    def forwardPM(self, inputs):
        random_sample_img, sensemap, random_sample_mask, full_sampling_kspace = inputs
        
        x0 = c2r(random_sample_img, axis=1)
        csm = sensemap
        mask = random_sample_mask
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
            x_k = self.gradD[k%self.num_iter](z_k, x0, csm, mask)  # (B, 2, nrow, ncol)
        out = r2c(x_k, axis=1)
        return out
    

  