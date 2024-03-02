import torch
import torch.nn as nn
import torch.nn.functional as F

class ComplexConv2d(nn.Module):
    # https://github.com/litcoderr/ComplexCNN/blob/master/complexcnn/modules.py
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super().__init__()

        ## Model components
        self.conv_re = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias, **kwargs)
        self.conv_im = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias, **kwargs)

    def forward(self, x):  # shpae of x : [batch,channel,axis1,axis2,2]
        real = self.conv_re(x[..., 0]) - self.conv_im(x[..., 1])
        imaginary = self.conv_re(x[..., 1]) + self.conv_im(x[..., 0])
        output = torch.stack((real, imaginary), dim=-1)
        return output

class ComplexUpsample(nn.Module):
    def __init__(
        self,
        size=None,
        scale_factor=None,
        mode='nearest',
        align_corners=False,
        recompute_scale_factor=False,
    ):
        """Upsample layer for complex inputs.

        Parameters
        ----------
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int], optional):
            output spatial sizes
        scale_factor (float or Tuple[float] or Tuple[float, float] or Tuple[float, float, float], optional):
            multiplier for spatial size. Has to match input size if it is a tuple.
        mode (str, optional): the upsampling algorithm: one of ``'nearest'``,
            ``'linear'``, ``'bilinear'``, ``'bicubic'`` and ``'trilinear'``.
            Default: ``'nearest'``
        align_corners (bool, optional): if ``True``, the corner pixels of the input
            and output tensors are aligned, and thus preserving the values at
            those pixels. This only has effect when :attr:`mode` is
            ``'linear'``, ``'bilinear'``, ``'bicubic'``, or ``'trilinear'``.
            Default: ``False``
        recompute_scale_factor (bool, optional): recompute the scale_factor for use in the
            interpolation calculation. If `recompute_scale_factor` is ``True``, then
            `scale_factor` must be passed in and `scale_factor` is used to compute the
            output `size`. The computed output `size` will be used to infer new scales for
            the interpolation. Note that when `scale_factor` is floating-point, it may differ
            from the recomputed `scale_factor` due to rounding and precision issues.
            If `recompute_scale_factor` is ``False``, then `size` or `scale_factor` will
            be used directly for interpolation.
        """
        super(ComplexUpsample, self).__init__()
        self.upsample = nn.Upsample(
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor
        )

    def forward(self, input):
        real, imag = torch.unbind(input, dim=-1)
        output = torch.stack((self.upsample(real), self.upsample(imag)), dim=-1)
        return output
    

class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super().__init__()

        ## Model components
        self.tconv_re = nn.ConvTranspose2d(in_channel, out_channel,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           output_padding=output_padding,
                                           groups=groups,
                                           bias=bias,
                                           dilation=dilation,
                                           **kwargs)
        self.tconv_im = nn.ConvTranspose2d(in_channel, out_channel,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           output_padding=output_padding,
                                           groups=groups,
                                           bias=bias,
                                           dilation=dilation,
                                           **kwargs)

    def forward(self, x):  # shpae of x : [batch,channel,axis1,axis2,2]
        real = self.tconv_re(x[..., 0]) - self.tconv_im(x[..., 1])
        imaginary = self.tconv_re(x[..., 1]) + self.tconv_im(x[..., 0])
        output = torch.stack((real, imaginary), dim=-1)
        return output


class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, **kwargs):
        super().__init__()
        self.bn_re = nn.BatchNorm2d(num_features=num_features, momentum=momentum, affine=affine, eps=eps, track_running_stats=track_running_stats, **kwargs)
        self.bn_im = nn.BatchNorm2d(num_features=num_features, momentum=momentum, affine=affine, eps=eps, track_running_stats=track_running_stats, **kwargs)

    def forward(self, x):
        real = self.bn_re(x[..., 0])
        imag = self.bn_im(x[..., 1])
        output = torch.stack((real, imag), dim=-1)
        return output
    

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding."""
    return ComplexConv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation
    )

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return ComplexConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, bypass=None):
        super(BasicBlock, self).__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv_block = nn.Sequential(
            conv3x3(inplanes, planes, stride),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            conv3x3(planes, planes)
        )
        self.bypass = bypass
        self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        identity = x
        
        out = self.conv_block(x)

        if self.bypass is not None:
            identity = self.bypass(x)

        out += identity
        out = self.activate(out)
        return out


def make_res_layer(inplanes, planes, blocks, stride=1):
    bypass = nn.Sequential(
        conv1x1(inplanes, planes, stride),
    )

    layers = []
    layers.append(BasicBlock(inplanes, planes, stride, bypass))
    for _ in range(1, blocks):
        layers.append(BasicBlock(planes, planes))

    return nn.Sequential(*layers)


class DoubleConv(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            ComplexConv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=int(kernel_size / 2)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True), 
            ComplexConv2d(out_ch, out_ch, 3, padding=1, dilation=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class ComplexResUnet(nn.Module):

    def __init__(self, in_chans, out_chans, num_chans=32, n_res_blocks=2, global_residual=True):
        super(ComplexResUnet, self).__init__()

        self.global_residual = global_residual
        self.layer1 = make_res_layer(in_chans, num_chans, n_res_blocks, stride=1)
        self.layer2 = make_res_layer(num_chans, num_chans * 2, n_res_blocks, stride=2)
        self.layer3 = make_res_layer(num_chans * 2, num_chans * 4, n_res_blocks, stride=2)
        self.layer4 = make_res_layer(num_chans * 4, num_chans * 8, n_res_blocks, stride=2)
        self.layer5 = make_res_layer(num_chans * 8, num_chans * 16, n_res_blocks, stride=2)

        self.up5 = ComplexUpsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.mconv4 = DoubleConv(num_chans * 24, num_chans * 8)
        self.up4 = ComplexUpsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.mconv3 = DoubleConv(num_chans * 12, num_chans * 4)
        self.up3 = ComplexUpsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.mconv2 = DoubleConv(num_chans * 6, num_chans * 2)
        self.up2 = ComplexUpsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.mconv1 = DoubleConv(num_chans * 3, num_chans)
        self.end = conv1x1(num_chans, out_chans)
        
    def forward(self, x):
        i = x.shape[1]//2
        real = x[:,:i].unsqueeze(-1)
        imag = x[:,i:].unsqueeze(-1)
        x = torch.cat((real,imag),-1)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        c5 = self.layer5(c4)

        merge4 = self.mconv4(torch.cat([self.up5(c5), c4], dim=1))
        merge3 = self.mconv3(torch.cat([self.up4(merge4), c3], dim=1))
        merge2 = self.mconv2(torch.cat([self.up3(merge3), c2], dim=1))
        merge1 = self.mconv1(torch.cat([self.up2(merge2), c1], dim=1))
        out = self.end(merge1)
        if self.global_residual:
            out = torch.sub(x, out)

        out = torch.cat((out[...,0],out[...,1]),1)
        return out

if __name__ == "__main__":
    x = torch.rand(1,2,320,320).cuda()
    model = ComplexResUnet(1,1,32,1,True).cuda()
    y = model(x)
    print(y.shape)
