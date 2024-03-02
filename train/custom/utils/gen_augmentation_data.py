import argparse
import os
import sys
import numpy as np
import torch
import h5py
from tqdm import tqdm
import pathlib
import matplotlib.pylab as plt
import random
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
work_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(work_dir)
from custom.utils.mri_tools import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--org_data_path', type=str, default='./train_data/processed_data/train')
    parser.add_argument('--aug_data_path', type=str, default='./train_data/aug_data')
    args = parser.parse_args()
    return args

def add_gaussian_noise(datas, mean = 0, stddev = 0.05):
    full_sampling_img, full_sampling_kspace, random_sample_img, random_sample_mask, sensemap = datas
    full_sampling_kspace_cuda = torch.from_numpy(full_sampling_kspace).cuda()
    # 生成均值为0，方差为0.2的正态分布随机数据，每个通道独立生成
    num_channels, image_height, image_width = sensemap.shape

    noise = torch.empty(num_channels, image_height, image_width, dtype=torch.complex64)
    for channel in range(num_channels):
        random_data_real = torch.randn(image_height, image_width) * stddev + mean
        random_data_img = torch.randn(image_height, image_width) * stddev + mean
        noise[channel, :, :] = random_data_real + 1j*random_data_img
    noise_cuda = noise.cuda()
    noise_kspace_cuda = torch.view_as_complex(fft2c_new(torch.view_as_real(noise_cuda)))
    full_sampling_kspace_with_noise = noise_kspace_cuda + full_sampling_kspace_cuda

    # 计算欠采图像
    sensemap_cuda = torch.from_numpy(sensemap).cuda()
    random_sample_mask_cuda = torch.from_numpy(random_sample_mask).cuda()[None]
    random_sample_img_noised = mriAdjointOpNoShift(full_sampling_kspace_with_noise, sensemap_cuda, random_sample_mask_cuda)

    # 计算满采图像
    full_sampling_img_noised = mriAdjointOpNoShift(full_sampling_kspace_with_noise, sensemap_cuda, torch.ones_like(full_sampling_kspace_with_noise))
    return full_sampling_img_noised, full_sampling_kspace_with_noise, random_sample_img_noised, random_sample_mask, sensemap


def flip(datas, axis=1):
    full_sampling_img, full_sampling_kspace, random_sample_img, random_sample_mask, sensemap = datas
    full_sampling_kspace_cuda = torch.from_numpy(full_sampling_kspace).cuda()
    full_sampling_img_cuda = torch.view_as_complex(ifft2c_new(torch.view_as_real(full_sampling_kspace_cuda)))
    full_sampling_img_flipped_cuda = torch.flip(full_sampling_img_cuda, [axis])
    full_sampling_kspace_flipped_cuda = torch.view_as_complex(fft2c_new(torch.view_as_real(full_sampling_img_flipped_cuda)))
    sensemap_cuda = torch.from_numpy(sensemap).cuda()
    sensemap_flipped_cuda = torch.flip(sensemap_cuda, [axis])
    random_sample_mask_cuda = torch.from_numpy(random_sample_mask).cuda()[None]
    random_sample_img_flipped = mriAdjointOpNoShift(full_sampling_kspace_flipped_cuda, sensemap_flipped_cuda, random_sample_mask_cuda)
    full_sampling_img_flipped = mriAdjointOpNoShift(full_sampling_kspace_flipped_cuda, sensemap_flipped_cuda, torch.ones_like(full_sampling_kspace_flipped_cuda))
    return full_sampling_img_flipped, full_sampling_kspace_flipped_cuda, random_sample_img_flipped, random_sample_mask, sensemap_flipped_cuda


if __name__ == '__main__':
    args = parse_args()
    org_data_path = pathlib.Path(args.org_data_path)
    aug_data_path = pathlib.Path(args.aug_data_path)
    os.makedirs(aug_data_path, exist_ok=True)

    for f_name in tqdm(os.listdir(org_data_path)):     
        f_path = org_data_path / f_name
        with h5py.File(f_path, 'r') as f:
            full_sampling_img = f['full_sampling_img'][:]                # 320*320 -complex64
            full_sampling_kspace = f['full_sampling_kspace'][:]          # 15*320*320 -complex64
            random_sample_img = f['random_sample_img'][:]                # 320*320 -complex64
            random_sample_mask = f['random_sample_mask'][:]              # 320*320 -int
            sensemap = f['sensemap'][:]                                  # 15*320*320 -complex64

        datas = [full_sampling_img, full_sampling_kspace, random_sample_img, random_sample_mask, sensemap]

        """添加高斯噪声，数据增强"""
        if random.random() < 1:
            out_datas = add_gaussian_noise(datas)

        """随机翻转，数据增强"""
        if random.random() < 0:
            out_datas = flip(datas, 1)

        if random.random() < 0:
            out_datas = flip(datas, 1)

        plt.figure()
        plt.subplot(2,2,1)
        plt.imshow(abs(full_sampling_img),cmap="gray")
        plt.subplot(2,2,2)
        plt.imshow(abs(random_sample_img),cmap="gray")
        plt.subplot(2,2,3)
        plt.imshow(abs(out_datas[0].cpu().numpy()),cmap="gray")
        plt.subplot(2,2,4)
        plt.imshow(abs(out_datas[2].cpu().numpy()),cmap="gray")
        plt.show()