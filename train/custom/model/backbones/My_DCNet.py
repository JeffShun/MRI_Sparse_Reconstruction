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
        x = torch.view_as_real(random_sample_img.unsqueeze(1))                # [B, 1, 320, 320, 2]
        y = torch.view_as_real(full_sampling_kspace.unsqueeze(2))             # [B, 15, 1, 320, 320, 2]
        smaps = torch.view_as_real(sensemap.unsqueeze(2))                     # [B, 15, 1, 320, 320, 2]
        mask = random_sample_mask.unsqueeze(1).unsqueeze(2).unsqueeze(-1)     # [B, 1, 1, 320, 320, 1]
        y*=mask
        
        for k in range(self.num_iter_total):
            x_thalf = self.reshapeWrapper(self.gradR[k%self.num_iter], x)
            x = self.gradD[k%self.num_iter](x_thalf, y, smaps, mask)
            
        out = torch.view_as_complex(x.squeeze(1).squeeze(1))
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
            z_k = self.gradR[k%self.num_iter](x_k) # (B, 2, nrow, ncol)
            #dc
            x_k = self.gradD[k%self.num_iter](z_k, x0, csm, mask)  # (B, 2, nrow, ncol)
        out = r2c(x_k, axis=1)
        return out
    
    def reshapeWrapper(self, model, input):
        # re-shape data from [nBatch, nSmaps, nFE, nPE, 2]
        # to [nBatch*nSmaps, 2, nFE, nPE]
        shp = input.shape
        output = input.view(shp[0] * shp[1], *shp[2:]).permute(0, 3, 1, 2)

        # apply denoising
        output = model(output)

        # re-shape data from [nBatch*nSmaps, 2, nFE, nPE]
        # to [nBatch, nSmaps, nFE, nPE, 2]
        output = output.permute(0, 2, 3, 1).view(*shp)

        return output 