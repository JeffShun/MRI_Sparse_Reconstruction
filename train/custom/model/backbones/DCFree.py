
import torch

class DCFree(torch.nn.Module):
    """
    Sensitivity network with data term based on forward and adjoint containing
    the sensitivity maps
    """
    def __init__(
        self,
        model,
        model_config,
        use_sensemap=False
    ):
        super().__init__()

        # setup the modules
        self.net = model(**model_config)
        self.use_sensemap = use_sensemap

    def forward(self, inputs):
        random_sample_img, sensemap, random_sample_mask, full_sampling_kspace = inputs
        x = torch.view_as_real(random_sample_img.unsqueeze(1))   # [B, 1, 320, 320, 2]
 
        input_x = torch.cat((x[...,0],x[...,1]), 1)
        x = self.net(input_x)
        x = torch.view_as_complex(x.unsqueeze(-1).permute(0, 4, 2, 3, 1).contiguous()) # [B, 1, 320, 320] - complex64
        if self.use_sensemap:
            out = torch.sum(x * sensemap.conj(), axis=1) # coil combine (B x nrow x ncol)
        else:
            out = x.squeeze(1)
        return out



