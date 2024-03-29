
import torch
import torch.nn as nn
import torch.nn.functional as F
from custom.utils.mri_tools import *

class LossCompose(object):

    """Composes several loss together.
    Args:
        Losses: list of Losses to compose.
    """
    def __init__(self, Losses):
        self.Losses = Losses

    def __call__(self, img, label):
        loss_dict = dict()
        for loss_f in self.Losses:
            loss = loss_f(img, label)
            loss_dict[loss_f._get_name()] = loss
        return loss_dict

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for loss in self.Losses:
            format_string += '\n'
            format_string += '    {0}'.format(loss)
        format_string += '\n)'
        return format_string

class LossSamplingWrapper(nn.Module):
    def __init__(self, loss_obj, sampling_p=0.5):
        super(LossSamplingWrapper, self).__init__()
        self.sampling_p = sampling_p
        self.loss_obj = loss_obj

    def forward(self, X, Y):
        scaling = 1e-4
        loss = self.loss_obj(X, Y)
        B, C, H, W = loss.shape
        assert C == 1 
        loss_flat = loss.view(B,-1)
        # softmax
        loss_norm = F.softmax(scaling * loss_flat, dim=-1)
        sampling_prob = self.sampling_p * H * W * loss_norm
        random_map = torch.rand_like(sampling_prob)
        sampling_weight = (random_map < sampling_prob).int()
        # print(sampling_weight.sum()/(B*C*H*W))
        sampling_loss = (loss_flat * sampling_weight).sum()/(sampling_weight.sum())
        return sampling_loss

    def _get_name(self):
        return self.loss_obj._get_name()

class SSIMLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03, reduce: bool = True):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super(SSIMLoss, self).__init__()
        self.reduce = reduce
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size**2)
        NP = win_size**2
        self.cov_norm = NP / (NP - 1)

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        if len(X.shape)==3:
            X = X.unsqueeze(1)
        Y = torch.abs(Y).unsqueeze(1)
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
        loss = (1 - S)

        if self.reduce:
            return loss.mean()
        return loss


class MSELoss(nn.Module):
    def __init__(self, reduce: bool = True):
        super(MSELoss, self).__init__()
        self.reduce = reduce

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        if len(X.shape)==3:
            X = X.unsqueeze(1)
        Y = torch.abs(Y).unsqueeze(1)
        loss = ((X-Y)**2)
        if self.reduce:
            return loss.mean()
        return loss
    
class MAELoss(nn.Module):
    def __init__(self, reduce: bool = True):
        super(MAELoss, self).__init__()
        self.reduce = reduce

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        if len(X.shape)==3:
            X = X.unsqueeze(1)
        Y = torch.abs(Y).unsqueeze(1)
        loss = torch.abs(X-Y)

        import matplotlib.pylab as plt
        import seaborn as sns
        plt.figure()
        # 将损失张量展平并转换为 NumPy 数组
        loss_array = loss.view(-1).cpu().detach().numpy()

        # 使用Seaborn绘制直方图
        sns.histplot(loss_array, bins=100, kde=False)
        plt.xlabel('L1 Loss Value')
        plt.ylabel('Pixel Count')
        plt.title('Distribution of L1 Loss')
        plt.show() 

        if self.reduce:
            return loss.mean()
        return loss