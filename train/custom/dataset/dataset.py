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

        if self._transforms:
            datas = [full_sampling_img, full_sampling_kspace, random_sample_img, random_sample_mask, sensemap]
            """添加高斯噪声，数据增强"""
            if random.random() < 0.1:
                datas = self._add_gaussian_noise(datas)

            """随机x轴翻转，数据增强"""
            if random.random() < 0.1:
                datas = self._flip(datas, 1)

            """随机y轴翻转，数据增强"""
            if random.random() < 0.1:
                datas = self._flip(datas, 2)

            full_sampling_img, full_sampling_kspace, random_sample_img, random_sample_mask, sensemap = datas

        full_sampling_img = torch.from_numpy(full_sampling_img)
        full_sampling_kspace = torch.from_numpy(full_sampling_kspace)
        random_sample_img = torch.from_numpy(random_sample_img)
        random_sample_mask = torch.from_numpy(random_sample_mask)
        sensemap = torch.from_numpy(sensemap)

        return random_sample_img, sensemap, random_sample_mask, full_sampling_img, full_sampling_kspace 


    def _add_gaussian_noise(self, datas, mean = 0, stddev = 0.05):
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
        return full_sampling_img_noised.cpu().numpy(), full_sampling_kspace_with_noise.cpu().numpy(), random_sample_img_noised.cpu().numpy(), random_sample_mask, sensemap


    def _flip(self, datas, axis=1):
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
        return full_sampling_img_flipped.cpu().numpy(), full_sampling_kspace_flipped_cuda.cpu().numpy(), random_sample_img_flipped.cpu().numpy(), random_sample_mask, sensemap_flipped_cuda.cpu().numpy()


