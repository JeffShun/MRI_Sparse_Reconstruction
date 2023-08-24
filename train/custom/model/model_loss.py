import torch
import torch.nn as nn
import torch.nn.functional as F

class SSIMLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super(SSIMLoss, self).__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size**2)
        NP = win_size**2
        self.cov_norm = NP / (NP - 1)

    def forward(self, Xs, Ys):
        loss = self.forward_single(Xs[0], Ys)
        for i in range(1, len(Xs)):
            loss+= self.forward_single(Xs[i], Ys)
        return loss

    def forward_single(self, X: torch.Tensor, Y: torch.Tensor):
        assert isinstance(self.w, torch.Tensor)
        B, C, W, D = Y.shape
        max_values, _ = torch.max(Y.view(B, C, -1), -1)
        max_values = max_values[:, :, None, None]
        C1 = (self.k1 * max_values) ** 2
        C2 = (self.k2 * max_values) ** 2
        self.w = self.w.to(X.device)
        ux = F.conv2d(X, self.w)  # typing: ignore
        uy = F.conv2d(Y, self.w)  #
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux**2 + uy**2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D
        return 1 - S.mean()


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        return ((X-Y)**2).mean()
    


class SmoothL1Loss(nn.Module):
    def __init__(self):
        super(SmoothL1Loss, self).__init__()
        self.smoothL1loss = nn.SmoothL1Loss(reduction="mean")

    def forward(self, Xs, Ys):
        loss = self.forward_single(Xs[0], Ys)
        for i in range(1, len(Xs)):
            loss+= self.forward_single(Xs[i], Ys)
        return loss        

    def forward_single(self, X, Y):
        return self.smoothL1loss(X, Y)
