"""data loader."""


from torch.utils import data
from custom.utils.common_tools import *
from custom.utils.mri_tools import *
import h5py

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
        with h5py.File(file_name, 'r') as f:
            full_sampling_img = f['full_sampling_img'][:]                # 320*320 -complex64
            full_sampling_kspace = f['full_sampling_kspace'][:]          # 15*320*320 -complex64
            random_sample_img = f['random_sample_img'][:]                # 320*320 -complex64
            random_sample_mask = f['random_sample_mask'][:]              # 320*320 -int
            sensemap = f['sensemap'][:]                                  # 15*320*320 -complex64

        full_sampling_img = torch.from_numpy(full_sampling_img)
        full_sampling_kspace = torch.from_numpy(full_sampling_kspace)
        random_sample_img = torch.from_numpy(random_sample_img)
        random_sample_mask = torch.from_numpy(random_sample_mask)
        sensemap = torch.from_numpy(sensemap)

        return random_sample_img, sensemap, random_sample_mask, full_sampling_img, full_sampling_kspace 



