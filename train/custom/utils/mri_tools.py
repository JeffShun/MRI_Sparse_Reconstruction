"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import List, Optional

import torch
import torch.fft
import numpy as np

def fft2c_new(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Apply centered 2 dimensional Fast Fourier Transform.

    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.
        norm: Normalization mode. See ``torch.fft.fft``.

    Returns:
        The FFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.fftn(  # type: ignore
            torch.view_as_complex(data), dim=(-2, -1), norm=norm
        )
    )
    data = fftshift(data, dim=[-3, -2])

    return data


def ifft2c_new(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.

    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.
        norm: Normalization mode. See ``torch.fft.ifft``.

    Returns:
        The IFFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.ifftn(  # type: ignore
            torch.view_as_complex(data), dim=(-2, -1), norm=norm
        )
    )
    data = fftshift(data, dim=[-3, -2])

    return data


# Helper functions
def roll_one_dim(x: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
    """
    Similar to roll but for only one dim.

    Args:
        x: A PyTorch tensor.
        shift: Amount to roll.
        dim: Which dimension to roll.

    Returns:
        Rolled version of x.
    """
    shift = shift % x.size(dim)
    if shift == 0:
        return x

    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)

    return torch.cat((right, left), dim=dim)


def roll(
    x: torch.Tensor,
    shift: List[int],
    dim: List[int],
) -> torch.Tensor:
    """
    Similar to np.roll but applies to PyTorch Tensors.

    Args:
        x: A PyTorch tensor.
        shift: Amount to roll.
        dim: Which dimension to roll.

    Returns:
        Rolled version of x.
    """
    if len(shift) != len(dim):
        raise ValueError("len(shift) must match len(dim)")

    for (s, d) in zip(shift, dim):
        x = roll_one_dim(x, s, d)

    return x


def fftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors

    Args:
        x: A PyTorch tensor.
        dim: Which dimension to fftshift.

    Returns:
        fftshifted version of x.
    """
    if dim is None:
        # this weird code is necessary for toch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = x.shape[dim_num] // 2

    return roll(x, shift, dim)


def ifftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors

    Args:
        x: A PyTorch tensor.
        dim: Which dimension to ifftshift.

    Returns:
        ifftshifted version of x.
    """
    if dim is None:
        # this weird code is necessary for toch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = (x.shape[dim_num] + 1) // 2

    return roll(x, shift, dim)


def c2r(complex_img, axis=0, expand_dim=True):
    """
    :input shape: row x col (complex64)
    :output shape: 2 x row x col (float32)
    """
    if isinstance(complex_img, np.ndarray):
        if expand_dim:
            real_img = np.stack((complex_img.real, complex_img.imag), axis=axis)
        else:
            real_img = np.concatenate((complex_img.real, complex_img.imag), axis=axis)
    elif isinstance(complex_img, torch.Tensor):
        if expand_dim:
            real_img = torch.stack((complex_img.real, complex_img.imag), axis=axis)
        else:
            real_img = torch.cat((complex_img.real, complex_img.imag), axis=axis)
    else:
        raise NotImplementedError
    return real_img

def r2c(real_img, axis=0):
    """
    :input shape: 2c x row x col (float32)
    :output shape: c x row x col (complex64)
    """
    c = real_img.shape[axis]//2
    if axis == 0: 
        complex_img = real_img[:c] + 1j*real_img[c:]
    elif axis == 1:
        complex_img = real_img[:,:c] + 1j*real_img[:,c:]
    else:
        raise NotImplementedError
    return complex_img

def mriAdjointOpNoShift(kspace, smaps, mask, coil_axis=-3):
    """ Compute Cartesian MRI adjoint operation (2D) without (i)fftshifts
    :param kspace: input kspace (pre-shifted) (np.array)
    :param smaps: precomputed sensitivity maps (np.array)
    :param mask: undersampling mask (pre-shifted)
    :param coil_axis: defines the axis of the coils (and extended coil sensitivity maps if softSENSE is used)
    :return: reconstructed image (np.array)
    """
    assert kspace.ndim >= 3
    assert kspace.ndim == smaps.ndim
    assert kspace.ndim == mask.ndim or mask.ndim == 2
    mask_kspace = kspace * mask
    return torch.sum(torch.view_as_complex(ifft2c_new(torch.view_as_real(mask_kspace)))*torch.conj(smaps), axis=coil_axis)

def GenRandomMask(n_frequencies=368, center_fraction=0.08, acceleration=4, seed=1000):
    rng = np.random.RandomState(seed)
    num_low_frequencies = round(n_frequencies * center_fraction)
    prob = (n_frequencies / acceleration - num_low_frequencies) / (n_frequencies - num_low_frequencies) 
    acceleration_mask = (rng.uniform(size=n_frequencies) < prob).astype(np.int32)
    center_mask = np.zeros(n_frequencies, dtype=np.int32)
    pad = (n_frequencies - num_low_frequencies + 1) // 2
    center_mask[pad : pad + num_low_frequencies] = 1
    mask = (1 - center_mask) * acceleration_mask + center_mask
    return mask.astype(np.float32)


def GenEquiSpacedMask(n_frequencies=368, center_fraction=0.08, acceleration=4, offset=0): 
    num_low_frequencies = round(n_frequencies * center_fraction)
    # determine acceleration rate by adjusting for the number of low frequencies
    adjusted_acceleration = (acceleration * (num_low_frequencies - n_frequencies)) / (num_low_frequencies * acceleration - n_frequencies)
    acceleration_mask = np.zeros(n_frequencies, dtype=np.int32)
    accel_samples = np.arange(offset, n_frequencies - 1, adjusted_acceleration)
    accel_samples = np.around(accel_samples).astype(np.int32)
    acceleration_mask[accel_samples] = 1
    center_mask = np.zeros(n_frequencies, dtype=np.int32)
    pad = (n_frequencies - num_low_frequencies + 1) // 2
    center_mask[pad : pad + num_low_frequencies] = 1
    mask = (1 - center_mask) * acceleration_mask + center_mask  
    return mask.astype(np.float32)


def center_crop(image, crop_size):
    """
    对图像进行中心裁剪。
    """

    # 获取输入图像的尺寸
    _, _, image_height, image_width = image.shape

    # 获取裁剪尺寸
    crop_height, crop_width = crop_size

    # 计算裁剪起始位置
    start_row = (image_height - crop_height) // 2
    start_col = (image_width - crop_width) // 2

    # 进行裁剪
    cropped_image = image[:, :, start_row:start_row+crop_height, start_col:start_col+crop_width]

    return cropped_image

# 复数图像模值归一化
def normlize_complex(data):
    data_modulus = torch.abs(data)
    data_angle = torch.angle(data)
    ori_shape = data_modulus.shape
    modulus_flat = data_modulus.reshape(ori_shape[0], -1)
    modulus_min, modulus_max = torch.min(modulus_flat, -1, keepdim=True)[0], torch.max(modulus_flat, -1, keepdim=True)[0]
    modulus_norm = (modulus_flat - modulus_min)/(modulus_max - modulus_min)
    modulus_norm = modulus_norm.reshape(ori_shape)
    return torch.polar(modulus_norm, data_angle)