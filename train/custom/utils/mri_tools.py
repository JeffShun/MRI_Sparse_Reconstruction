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


def c2r(complex_img, axis=0):
    """
    :input shape: row x col (complex64)
    :output shape: 2 x row x col (float32)
    """
    if isinstance(complex_img, np.ndarray):
        real_img = np.stack((complex_img.real, complex_img.imag), axis=axis)
    elif isinstance(complex_img, torch.Tensor):
        real_img = torch.stack((complex_img.real, complex_img.imag), axis=axis)
    else:
        raise NotImplementedError
    return real_img

def r2c(real_img, axis=0):
    """
    :input shape: 2 x row x col (float32)
    :output shape: row x col (complex64)
    """
    if axis == 0:
        complex_img = real_img[0] + 1j*real_img[1]
    elif axis == 1:
        complex_img = real_img[:,0] + 1j*real_img[:,1]
    else:
        raise NotImplementedError
    return complex_img

def complex_mult(data1, data2, dim=-1):
    """
    Element-wise complex matrix multiplication X^T Y
    Params:
      data1: tensor
      data2: tensor
      dim: dimension that represents the complex values
    """
    assert data1.size(dim) == 2
    assert data2.size(dim) == 2
    re1, im1 = torch.unbind(data1, dim=dim)
    re2, im2 = torch.unbind(data2, dim=dim)
    return torch.stack([re1 * re2 - im1 * im2, im1 * re2 + re1 * im2], dim = dim)

def complex_mult_conj(data1, data2, dim=-1):
    """
    Element-wise complex matrix multiplication with conjugation X^H Y
    Params:
      data1: tensor
      data2: tensor
      dim: dimension that represents the complex values
    """
    assert data1.size(dim) == 2
    assert data2.size(dim) == 2
    re1, im1 = torch.unbind(data1, dim=dim)
    re2, im2 = torch.unbind(data2, dim=dim)
    return torch.stack([re1 * re2 + im1 * im2, im1 * re2 - re1 * im2], dim = -1)


def forwardSoftSenseOpNoShift(th_img, th_smaps, th_mask):
    data_mul = complex_mult(th_img.unsqueeze(1), th_smaps)
    th_kspace = fft2c_new(data_mul) * th_mask
    th_kspace = torch.sum(th_kspace, dim=-4, keepdim=True)
    return th_kspace

def adjointSoftSenseOpNoShift(th_kspace, th_smaps, th_mask):
    data_mul = ifft2c_new(th_kspace * th_mask)
    th_img = complex_mult_conj(data_mul, th_smaps)
    th_img = torch.sum(th_img, dim=(-5))
    return th_img

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