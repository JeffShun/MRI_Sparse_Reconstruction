"""data loader."""

import random
import numpy as np
from torch.utils import data
from custom.utils.common_tools import *

class MyDataset(data.Dataset):
    def __init__(
            self,
            dst_list_file,
            transforms
    ):
        self.data_lst = self._load_files(dst_list_file)
        self._transforms = transforms

    def _load_files(self, file):
        data_list = []
        with open(file, 'r') as f:
            for line in f:
                data_list.append(line.strip())
        return data_list

    def __getitem__(self, idx):
        source_data = self._load_source_data(self.data_lst[idx])
        return source_data

    def __len__(self):
        return len(self.data_lst)

    def _load_source_data(self, file_name):
        data = np.load(file_name, allow_pickle=True)

        acc_img = data['acc_img']
        sos_img = data['sos_img']
        random_sample_img_4 = data['random_sample_img_4']
        random_sample_img_8 = data['random_sample_img_8']
        eqs_sample_img_4 = data['eqs_sample_img_4']
        eqs_sample_img_8 = data['eqs_sample_img_8']

        img = random_sample_img_4
        label = sos_img[np.newaxis,:,:]
        # transform前，数据必须转化为[C,H,D]的形状
        if self._transforms:
            img, label = self._transforms(img, label)

        ##################### Debug ##########################
        # # 多通道图像合并
        # random_sample_img_4 = np.transpose(random_sample_img_4, axes=(1, 2, 0))
        # random_sample_img_4 = sos(random_sample_img_4)

        # random_sample_img_8 = np.transpose(random_sample_img_8, axes=(1, 2, 0))
        # random_sample_img_8 = sos(random_sample_img_8)

        # eqs_sample_img_4 = np.transpose(eqs_sample_img_4, axes=(1, 2, 0))
        # eqs_sample_img_4 = sos(eqs_sample_img_4)

        # eqs_sample_img_8 = np.transpose(eqs_sample_img_8, axes=(1, 2, 0))
        # eqs_sample_img_8 = sos(eqs_sample_img_8)
        
        # plt.figure(1)
        # plt.subplot(241)
        # plt.title("acc_img")
        # plt.imshow(np.abs(acc_img),cmap="gray")    
        # plt.subplot(242)
        # plt.title("sos_img")
        # plt.imshow(sos_img,cmap="gray")  
        # plt.subplot(243)
        # plt.title("random_sample_img_4")
        # plt.imshow(random_sample_img_4,cmap="gray")   
        # plt.subplot(244)
        # plt.title("random_sample_img_8")
        # plt.imshow(random_sample_img_8,cmap="gray")  
        # plt.subplot(245)
        # plt.title("eqs_sample_img_4")
        # plt.imshow(eqs_sample_img_4,cmap="gray")  
        # plt.subplot(246)
        # plt.title("eqs_sample_img_8")
        # plt.imshow(eqs_sample_img_8,cmap="gray")  
        # plt.show()
        ##################### Debug ##########################
        return img.float(), label.float()



