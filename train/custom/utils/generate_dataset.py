"""生成模型输入数据."""

import argparse
import glob
import os
from tqdm import tqdm
from multiprocessing import Pool
import h5py
import numpy as np
import pathlib
import torch
import sys
work_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(work_dir)
from custom.utils.fftc import *
from fftc import *
import time

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
    return mask.astype(np.float32)


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
    return mask.astype(np.float32)


def sos(im, axes=-1):
    '''Root sum of squares combination along given axes.
    '''
    return torch.sqrt(torch.sum(torch.abs(im)**2, axis=axes))


def AdaptiveCoilCombine(data_tensor):
    k = 6
    chunks = data_tensor.chunk(k, dim=0)  # 在第 0 维度进行等分
    out_chunks = []
    for i, chunk in enumerate(chunks):
        chunk = chunk.cuda()
        n_slice, nx, ny, nc = chunk.shape
        # Step-01
        data_flatten1 = chunk.reshape(-1, nc, 1)
        data_flatten2 = chunk.reshape(-1, 1, nc)
        mat_corr = torch.bmm(data_flatten1, data_flatten2)

        # Step-02
        U, S, _ = torch.svd(mat_corr)
        vfilter = U[:, :, 0].unsqueeze(1)
        data_out = torch.bmm(vfilter.conj(), data_flatten1)
        data_out = data_out.reshape(n_slice, nx, ny)
        out_chunks.append(data_out)
    return torch.cat(out_chunks,0).cpu()


def center_crop(image, crop_size):
    """
    对图像进行中心裁剪。
    """

    # 获取输入图像的尺寸
    n_slices, n_coils, image_height, image_width = image.shape

    # 获取裁剪尺寸
    crop_height, crop_width = crop_size

    # 计算裁剪起始位置
    start_row = (image_height - crop_height) // 2
    start_col = (image_width - crop_width) // 2

    # 进行裁剪
    cropped_image = image[:, :, start_row:start_row+crop_height, start_col:start_col+crop_width]

    return cropped_image


def kspace2img(ksapce):
    n_slice, nc, nx, ny = ksapce.shape
    ksapce = torch.view_as_real(ksapce)
    img = fft2c_new(ksapce).reshape(n_slice, nc, nx, ny, 2)
    return torch.view_as_complex(img)    


def gen_lst(tgt_path, task, processed_pids):
    save_file = tgt_path.parent / (task + '.txt')
    data_list = glob.glob(os.path.join(tgt_path, '*.h5'))
    data_list = [file.replace("\\","/") for file in data_list]
    num = 0
    with open(save_file, 'w') as f:
        for data in data_list:
            data = str(pathlib.Path(data))
            data = data.replace("\\","/")
            if data.split("/")[-1].split("_")[0] in processed_pids:
                num+=1
                f.writelines(data + '\r')
    print('num of data: ', num)


def process_single(input):
    data_path, tgt_path, pid = input
    kspace, reconstruction_rss = load_h5(data_path)
    n_slices, n_coils, n_rows, n_frequencies = kspace.shape
    target_size = (320, 320)
    kspace = torch.from_numpy(kspace)

    ####################################### 计算满采图像 ######################################
    # 对每个kspace切片转到图像域
    img = kspace2img(kspace)
    crop_img = center_crop(img, target_size)
    assert(tuple(crop_img.shape[-2:])==target_size)
    crop_img = crop_img.permute(0, 2, 3, 1)
    # acc算法通道合并后的图像
    acc_img = AdaptiveCoilCombine(crop_img).numpy()
    # sos算法通道合并后的图像
    sos_img = sos(crop_img).numpy()

    ####################################### 计算欠采图像 ######################################
    # 欠采k空间数据
    random_sample_mask_4 = RandomMask(n_frequencies=n_frequencies, center_fraction=0.08, acceleration=4, seed=1000)
    random_sample_mask_8 = RandomMask(n_frequencies=n_frequencies, center_fraction=0.04, acceleration=8, seed=1000)
    eqs_sample_mask_4 = EquiSpacedMask(n_frequencies, center_fraction=0.08, acceleration=4, offset=0)
    eqs_sample_mask_8 = EquiSpacedMask(n_frequencies, center_fraction=0.04, acceleration=8, offset=0)

    random_sample_kspace_4 = torch.from_numpy(random_sample_mask_4)[None,None,None] * kspace
    random_sample_kspace_8 = torch.from_numpy(random_sample_mask_8)[None,None,None] * kspace
    eqs_sample_kspace_4 = torch.from_numpy(eqs_sample_mask_4)[None,None,None] * kspace
    eqs_sample_kspace_8 = torch.from_numpy(eqs_sample_mask_8)[None,None,None] * kspace

    # 欠采图像数据
    random_sample_img_4 = center_crop(kspace2img(random_sample_kspace_4), target_size).numpy()
    random_sample_img_8 = center_crop(kspace2img(random_sample_kspace_8), target_size).numpy()
    eqs_sample_img_4 = center_crop(kspace2img(eqs_sample_kspace_4), target_size).numpy()
    eqs_sample_img_8 = center_crop(kspace2img(eqs_sample_kspace_8), target_size).numpy()

    for i in range(n_slices):
        # 将数据存储到 .h5 文件
        f =  h5py.File(os.path.join(tgt_path, f'{pid}_{i}.h5'),  'w')
        f.create_dataset('acc_img', data=acc_img[i])
        f.create_dataset('sos_img', data=sos_img[i])
        f.create_dataset('random_sample_img_4', data=random_sample_img_4[i])
        f.create_dataset('random_sample_img_8', data=random_sample_img_8[i])
        f.create_dataset('eqs_sample_img_4', data=eqs_sample_img_4[i])
        f.create_dataset('eqs_sample_img_8', data=eqs_sample_img_8[i])
        f.close()


if __name__ == '__main__':
    time_start = time.time()
    args = parse_args()
    src_path = args.src_path
    for task in ["train", "val", "test"]:
        print("\nBegin gen %s data!"%(task))
        src_data_path = pathlib.Path(args.src_path) / "multicoil_{}".format(task)
        if not src_data_path.exists():
            print(str(src_data_path) + " does not exist!")
            continue
        tgt_path = pathlib.Path(args.tgt_path) / task
        os.makedirs(tgt_path, exist_ok=True)
        inputs = []
        for f_name in tqdm(os.listdir(src_data_path)):     
            pid = f_name.replace(".h5", "").replace("file","")
            data_path = src_data_path / f_name
            inputs.append([data_path, tgt_path, pid])
            process_single([data_path,tgt_path,pid])
        processed_pids = [pid.replace(".h5", "").replace("file","") for pid in os.listdir(src_data_path)]
        pool = Pool(4)
        pool.map(process_single, inputs)
        pool.close()
        pool.join()
        # 生成Dataset所需的数据列表
        gen_lst(tgt_path, task, processed_pids)
    time_cost = time.time() - time_start
    print("time_cost: ", time_cost)

