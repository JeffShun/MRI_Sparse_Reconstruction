"""生成模型输入数据."""

import argparse
import glob
import os
from tqdm import tqdm
import h5py
import numpy as np
import pathlib
import torch
import sys
import time
work_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(work_dir)
from custom.utils.mri_tools import *
from custom.utils.common_tools import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./train_data/origin_data')
    parser.add_argument('--tgt_path', type=str, default='./train_data/processed_data')
    args = parser.parse_args()
    return args

def load_h5(h5_path):
    with h5py.File(h5_path, "r") as hf:
        kspace = hf["kspace"][:]
        reconstruction_rss = hf["reconstruction_rss"][:]
    return kspace, reconstruction_rss

def gen_sampling_mask(n_frequencies, save_path):
    random_sample_mask = GenRandomMask(n_frequencies=n_frequencies, center_fraction=0.08, acceleration=4, seed=1000)
    if not os.path.exists(save_path):
        with h5py.File(save_path,  'w') as f:
            f.create_dataset('random_sample_mask', data=random_sample_mask)

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
    data_path, espirit_path, task_tgt_path, pid = input
    kspace, reconstruction_rss = load_h5(data_path)
    n_slices, n_coils, n_rows, n_frequencies = kspace.shape
    kspace = torch.from_numpy(kspace).cuda()

    """计算满采图像"""
    img = torch.view_as_complex(ifft2c_new(torch.view_as_real(kspace)))
    img = center_crop(img, target_size)
    img = normlize_complex(img)
    kspace = torch.view_as_complex(fft2c_new(torch.view_as_real(img)))
    kspace_arr = kspace.cpu().numpy()

    """计算欠采mask"""
    with h5py.File(mask_path, 'r') as f:
        random_sample_mask = f['random_sample_mask'][:]
        random_sample_mask = torch.from_numpy(random_sample_mask).cuda()
        random_sample_mask = random_sample_mask[None][None][None] * torch.ones(kspace.shape).cuda()
        random_sample_mask_arr = random_sample_mask.cpu().numpy()[0,0]

    """读取sensemap"""
    # sensemap需要在服务器上通过bart生成
    with h5py.File(espirit_path, 'r') as file:
        sensemap = file['smaps_acl30'][:].squeeze()
        sensemap = center_crop(sensemap, target_size)

    """计算欠采图像"""
    sensemap_th = torch.from_numpy(sensemap).cuda()
    random_sample_img = mriAdjointOpNoShift(kspace, sensemap_th, random_sample_mask).cpu().numpy()

    # 满采图像数据
    full_sampling_img = mriAdjointOpNoShift(kspace, sensemap_th, torch.ones_like(kspace).cuda()).cpu().numpy()

    # os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    # import matplotlib.pylab as plt
    # plt.figure()
    # plt.subplot(1,5,1)
    # plt.imshow(np.abs(full_sampling_img[10]), cmap="gray")
    # plt.subplot(1,5,2)
    # plt.imshow(np.abs(kspace_arr[10,10]), cmap="gray")
    # plt.subplot(1,5,3)
    # plt.imshow(np.abs(random_sample_img[10]), cmap="gray")
    # plt.subplot(1,5,4)
    # plt.imshow(random_sample_mask_arr, cmap="gray")
    # plt.subplot(1,5,5)
    # plt.imshow(np.abs(sensemap[10,10]), cmap="gray")
    # plt.show()

    for i in range(n_slices):
        # 将数据存储到 .h5 文件
        with h5py.File(os.path.join(task_tgt_path, f'{pid}_{i}.h5'),  'w') as f:
            f.create_dataset('full_sampling_img', data=full_sampling_img[i])     # 320*320 -complex64
            f.create_dataset('full_sampling_kspace', data=kspace_arr[i])         # 15*320*320 -complex64
            f.create_dataset('random_sample_img', data=random_sample_img[i])     # 320*320 -complex64
            f.create_dataset('random_sample_mask', data=random_sample_mask_arr)  # 320*320 -int
            f.create_dataset('sensemap', data=sensemap[i])                       # 15*320*320 -complex64
   

if __name__ == '__main__':
    time_start = time.time()
    target_size = (320, 320)
    args = parse_args()
    src_path = pathlib.Path(args.src_path)
    tgt_path = pathlib.Path(args.tgt_path)

    # 生成下采样Mask
    os.makedirs(tgt_path / "sampling_masks", exist_ok=True)
    mask_path = tgt_path / "sampling_masks" / "mask.h5"
    if not os.path.exists(mask_path):
        gen_sampling_mask(target_size[-1], mask_path)

    # 生成训练数据和验证数据
    for task in ["train", "val", "test"]:
        print("\nBegin gen %s data!"%(task))
        src_data_path = pathlib.Path(args.src_path) / "multicoil_{}".format(task)
        if not src_data_path.exists():
            print(str(src_data_path) + " does not exist!")
            continue

        src_espirit_path = pathlib.Path(args.src_path) / "multicoil_{}_espirit".format(task)
        if not src_espirit_path.exists():
            print(str(src_espirit_path) + " does not exist!")
            continue

        task_tgt_path = tgt_path / task
        os.makedirs(task_tgt_path, exist_ok=True)
        inputs = []
        for f_name in tqdm(os.listdir(src_data_path)):     
            pid = f_name.replace(".h5", "").replace("file","")
            data_path = src_data_path / f_name
            espirit_path = src_espirit_path / f_name
            process_single([data_path, espirit_path, task_tgt_path, pid])
        processed_pids = set([pid.replace(".h5", "").replace("file","") for pid in os.listdir(src_data_path)])
        # 生成Dataset所需的数据列表
        gen_lst(task_tgt_path, task, processed_pids)
    time_cost = time.time() - time_start
    print("time_cost: ", time_cost)


