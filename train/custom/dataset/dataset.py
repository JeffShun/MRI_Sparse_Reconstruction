"""data loader."""
import numpy as np
from torch.utils import data
from custom.utils.common_tools import *
import copy
import h5py

class MyDataset(data.Dataset):
    def __init__(
            self,
            sample_list_file,
            mask_file,
            transforms
    ):
        self.data_lst = self._load_files(sample_list_file)
        self.mask_file = mask_file
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

    def sos(self, data):
        return torch.sqrt(torch.sum(torch.abs(data)**2, axis=0, keepdim=True))

    def _load_source_data(self, file_name):
        with h5py.File(file_name, 'r') as f:
            full_sampling_img = f['full_sampling_img'][:]  # [320, 320]   
            random_sample_img_4 = f['random_sample_img_4'][:] # [15, 320, 320]

        with h5py.File(self.mask_file, 'r') as f:
            random_sample_mask_4 = f['random_sample_mask_4'][:] 

        img = copy.deepcopy(random_sample_img_4)  
        label = copy.deepcopy(full_sampling_img)
        sample_mask = torch.from_numpy(random_sample_mask_4)

        # transform前，数据必须转化为[C,H,D]的形状
        if self._transforms:
            img, label = self._transforms(img, label)
            
        label = self.sos(label)
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
        return img, label, sample_mask



