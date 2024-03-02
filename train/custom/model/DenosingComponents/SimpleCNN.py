import torch
import torch.nn as nn

#CNN denoiser ======================
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

class SimpleCNN(nn.Module):
    def __init__(self, in_chans=2, out_chans=2, num_chans=64, n_layers=5, global_residual=True):
        super().__init__()
        self.global_residual = global_residual
        layers = []
        layers += conv_block(in_chans, num_chans)

        for _ in range(n_layers-2):
            layers += conv_block(num_chans, num_chans)

        layers += nn.Sequential(
            nn.Conv2d(num_chans, out_chans, 3, padding=1),
            nn.BatchNorm2d(out_chans)
        )
        self.nw = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.nw(x)
        if self.global_residual:
            out = torch.sub(x, out)
        return out