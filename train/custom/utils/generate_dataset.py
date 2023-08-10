"""生成模型输入数据."""

import argparse
import glob
import os
from tqdm import tqdm
from multiprocessing import Pool
import h5py
import matplotlib.pylab as plt
from scipy.linalg import svd
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./train_data/origin_data')
    parser.add_argument('--tgt_path', type=str, default='./train_data/processed_data')
    args = parser.parse_args()
    return args

def load_h5(h5_path):
    with h5py.File(h5_path, "r") as hf:
        kspace = np.array(hf["kspace"])
        reconstruction_rss = np.array(hf["reconstruction_rss"])
    return kspace, reconstruction_rss

def RandomMask(n_frequencies=368, center_fraction=0.08, acceleration=4, seed=1000):
    rng = np.random.RandomState(seed)
    num_low_frequencies = round(n_frequencies * center_fraction)
    prob = (n_frequencies / acceleration - num_low_frequencies) / (n_frequencies - num_low_frequencies) 
    acceleration_mask = (rng.uniform(size=n_frequencies) < prob).astype(np.int32)
    center_mask = np.zeros(n_frequencies, dtype=np.int32)
    pad = (n_frequencies - num_low_frequencies + 1) // 2
    center_mask[pad : pad + num_low_frequencies] = 1
    mask = (1 - center_mask) * acceleration_mask + center_mask
    return mask

def EquiSpacedMask(n_frequencies=368, center_fraction=0.08, acceleration=4, offset=0): 
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
    return mask


def sos(im, axes=-1):
    '''Root sum of squares combination along given axes.

    Parameters
    ----------
    im : array_like
        Input image.
    axes : tuple
        Dimensions to sum across.

    Returns
    -------
    array_like
        SOS combination of image.
    '''
    return np.sqrt(np.sum(np.abs(im)**2, axis=axes))


def AdaptiveCoilCombine(data_in):
    """
    Adaptive Combination using Temporal Phase information (ACTP) coil combine.

    Inputs:
        - data_in: a 3D array containing multi-channel complex images ([nx, ny, nc])

    Outputs:
        - data_out: composited complex images ([nx, ny])

    Author: Beck Zhu, 2022-12-26, initial version
    """

    nx, ny, nc = data_in.shape
    data_out = np.zeros((nx, ny), dtype=np.complex128)

    mat_corr = np.zeros((nx, ny, nc, nc), dtype=np.complex128)

    # Step-01
    for idx_ch1 in range(nc):
        for idx_ch2 in range(nc):
            mat_corr[:, :, idx_ch1, idx_ch2] = mat_corr[:, :, idx_ch1, idx_ch2] + data_in[:, :, idx_ch1] * np.conj(data_in[:, :, idx_ch2])

    # Step-02
    for idx in range(nx):
        for idy in range(ny):
            U, S, _ = svd(mat_corr[idx, idy, :, :])
            vfilter = U[:, 0]

            data_out[idx, idy] = np.dot(vfilter.conj(), data_in[idx, idy, :])

    return data_out


def center_crop(image, crop_size):
    """
    对输入的二维图像进行中心裁剪。
    """

    # 获取输入图像的尺寸
    n_coils, image_height, image_width = image.shape

    # 获取裁剪尺寸
    _, crop_height, crop_width = crop_size

    # 计算裁剪起始位置
    start_row = (image_height - crop_height) // 2
    start_col = (image_width - crop_width) // 2

    # 进行裁剪
    cropped_image = image[:, start_row:start_row+crop_height, start_col:start_col+crop_width]

    return cropped_image

def get_mask_img(mask,shape):
    mask_arr = np.zeros(shape,dtype=np.uint8)
    for i,flag in enumerate(mask):
        if flag:
            mask_arr[:,i] = 1
    return mask_arr


def kspace2img(ksapce):
    kspace_shift = np.fft.ifftshift(ksapce, axes=(-2,-1))
    img = np.fft.ifft2(kspace_shift)
    img = np.fft.fftshift(img)
    return img     


def gen_lst(tgt_path, task, processed_pids):
    save_file = os.path.join(tgt_path, task+'.txt')
    data_list = glob.glob(os.path.join(tgt_path, '*.npz'))
    data_list = [file.replace("\\","/") for file in data_list]
    num = 0
    with open(save_file, 'w') as f:
        for data in data_list:
            data = data.replace("\\","/")
            if data.split("/")[-1].split("_")[0] in processed_pids:
                num+=1
                f.writelines(data + '\r')
    print('num of data: ', num)


def process_single(input):
    data_path, tgt_path, pid = input
    kspace, reconstruction_rss = load_h5(data_path)
    n_slices, n_coils, n_rows, n_frequencies = kspace.shape
    target_size = (15, 320, 320)
    for i in range(n_slices):
        try:
            ####################################### 计算满采图像 ######################################
            # 对每个kspace切片转到图像域
            slice_kspace = kspace[i]    # size: [n_coils, n_rows, n_frequencies]
            slice_img = kspace2img(slice_kspace)
            slice_img = center_crop(slice_img, target_size)
            assert(slice_img.shape==target_size)
            # 多通道图像合并
            slice_img = np.transpose(slice_img, axes=(1, 2, 0))
            # acc算法通道合并后的图像
            acc_img = AdaptiveCoilCombine(slice_img)
            # sos算法通道合并后的图像
            sos_img = sos(slice_img)
            
            ####################################### 计算欠采图像 ######################################
            # 欠采k空间数据
            random_sample_mask_4 = RandomMask(n_frequencies=n_frequencies, center_fraction=0.08, acceleration=4, seed=1000)
            random_sample_mask_8 = RandomMask(n_frequencies=n_frequencies, center_fraction=0.04, acceleration=8, seed=1000)
            eqs_sample_mask_4 = EquiSpacedMask(n_frequencies, center_fraction=0.08, acceleration=4, offset=0)
            eqs_sample_mask_8 = EquiSpacedMask(n_frequencies, center_fraction=0.04, acceleration=8, offset=0)

            random_sample_kspace_4 = get_mask_img(random_sample_mask_4, (n_rows, n_frequencies))[np.newaxis,:,:] * slice_kspace
            random_sample_kspace_8 = get_mask_img(random_sample_mask_8, (n_rows, n_frequencies))[np.newaxis,:,:] * slice_kspace
            eqs_sample_kspace_4 = get_mask_img(eqs_sample_mask_4, (n_rows, n_frequencies))[np.newaxis,:,:] * slice_kspace
            eqs_sample_kspace_8 = get_mask_img(eqs_sample_mask_8, (n_rows, n_frequencies))[np.newaxis,:,:] * slice_kspace

            # 欠采图像数据
            random_sample_img_4 = center_crop(kspace2img(random_sample_kspace_4), target_size)
            random_sample_img_8 = center_crop(kspace2img(random_sample_kspace_8), target_size)
            eqs_sample_img_4 = center_crop(kspace2img(eqs_sample_kspace_4), target_size)
            eqs_sample_img_8 = center_crop(kspace2img(eqs_sample_kspace_8), target_size)

            np.savez_compressed(os.path.join(tgt_path, f'{pid}_{i}.npz'), 
                                acc_img=acc_img, 
                                sos_img=sos_img,
                                random_sample_img_4=random_sample_img_4,
                                random_sample_img_8=random_sample_img_8,
                                eqs_sample_img_4=eqs_sample_img_4,
                                eqs_sample_img_8=eqs_sample_img_8
                                )
        except Exception as e:
            print(f"Error: {e}")
            continue 


if __name__ == '__main__':
    args = parse_args()
    src_path = args.src_path
    for task in ["train", "val"]:
        print("\nBegin gen %s data!"%(task))
        src_data_path = os.path.join(args.src_path, "multicoil_"+task)
        tgt_path = args.tgt_path
        os.makedirs(tgt_path, exist_ok=True)
        # inputs = []
        # for f_name in tqdm(os.listdir(src_data_path)):     
        #     pid = f_name.replace(".h5", "").replace("file","")
        #     data_path = os.path.join(src_data_path, f_name)
        #     inputs.append([data_path, tgt_path, pid])
        processed_pids = [pid.replace(".h5", "").replace("file","") for pid in os.listdir(src_data_path)]
        # pool = Pool(8)
        # pool.map(process_single, inputs)
        # pool.close()
        # pool.join()
        # 生成Dataset所需的数据列表
        gen_lst(tgt_path, task, processed_pids)
