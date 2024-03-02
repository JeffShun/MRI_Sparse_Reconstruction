
import torch
from custom.utils.mri_tools import *

class MC_DCFree(torch.nn.Module):
    """
    Sensitivity network with data term based on forward and adjoint containing
    the sensitivity maps
    """
    def __init__(
        self,
        model,
        model_config,
    ):
        super().__init__()

        # setup the modules
        self.net = model(**model_config)

    def forward(self, inputs):
        random_sample_img, sensemap, random_sample_mask, full_sampling_kspace = inputs

        full_sampling_kspace = torch.view_as_real(full_sampling_kspace)                # [B, 15, 320, 320, 2]
        mask = random_sample_mask.unsqueeze(1).unsqueeze(-1)                           # [B, 1, 320, 320, 1]
        x = ifft2c_new(full_sampling_kspace * mask)                                    # [B, 15, 320, 320, 2]
                                     
        input_x = torch.cat((x[...,0],x[...,1]), 1)                                    # [B, 30, 320, 320]
        x = self.net(input_x)                                                          # [B, 2, 320, 320]
        x = torch.view_as_complex(x.unsqueeze(-1).permute(0, 4, 2, 3, 1).contiguous()) # [B, 1, 320, 320] - complex64
        out = x.squeeze(1)
        return out



