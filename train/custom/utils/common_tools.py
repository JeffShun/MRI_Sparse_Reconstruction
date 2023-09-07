import random
import torch
import logging
from bisect import bisect_right
import os
import tarfile
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

"""
数据预处理工具
1、所有数据预处理函数都包含两个输入: img 、label
2、img、label的输入维度为3维[C,H,W]，第一个维度是通道数
"""

class TransformCompose(object):

    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, label):
        for t in self.transforms:
            img, label = t(img, label)
        return img, label

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class to_tensor(object):
    def __call__(self, img, label):
        img_o = torch.from_numpy(img)
        label_o = torch.from_numpy(label)
        return img_o, label_o

class normlize(object):
    @ staticmethod
    def _normlize_real(data):
        ori_shape = data.shape
        data_o = data.reshape(-1)
        data_o = (data_o - data_o.min())/(data_o.max() - data_o.min())
        data_o = data_o.reshape(ori_shape)
        return data_o

    @ staticmethod
    def _normlize_complex(data):
        data_modulus = torch.abs(data)
        data_angle = torch.angle(data)
        modulus_norm = normlize._normlize_real(data_modulus)
        data_o = torch.polar(modulus_norm, data_angle)
        return data_o
    
    @ staticmethod
    def _normlize_real2(data):
        ori_shape = data.shape
        data_o = data.reshape(ori_shape[0], -1)
        data_min = data_o.min(dim=-1,keepdim=True)[0]
        data_max = data_o.max(dim=-1,keepdim=True)[0]
        data_o = (data_o - data_min)/(data_max - data_min)
        data_o = data_o.reshape(ori_shape)
        return data_o

    @ staticmethod
    def _normlize_complex2(data):
        real = data.real
        imag = data.imag
        real = normlize._normlize_real2(real)
        imag = normlize._normlize_real2(imag)
        data_o = real + 1j * imag
        return data_o

    def __call__(self, img, label):
        if torch.is_complex(img):
            img_o = self._normlize_complex(img)
        else:
            img_o = self._normlize_real(img)

        if torch.is_complex(label):
            label_o = self._normlize_complex(label)
        else:
            label_o = self._normlize_real(label)
        return img_o, label_o

class complex_to_multichannel(object):
    @ staticmethod
    def _complex_to_multichannel(data):
        data_o = torch.concat((data.real, data.imag), 0)
        return data_o
    def __call__(self, img, label):
        img_o, label_o = img, label
        if torch.is_complex(img_o):
            img_o = self._complex_to_multichannel(img)
        if torch.is_complex(label_o):
            label_o = self._complex_to_multichannel(label_o)
        return img_o, label_o


class random_flip(object):
    def __init__(self, axis=1, prob=0.5):
        assert isinstance(axis, int) and axis in [1,2]
        self.axis = axis
        self.prob = prob

    def __call__(self, img, label):
        img_o, label_o = img, label
        if random.random() < self.prob:
            img_o = torch.flip(img, [self.axis])
            label_o = torch.flip(label, [self.axis])
        return img_o, label_o

class random_contrast(object):
    def __init__(self, alpha_range=[0.8, 1.2], prob=0.5):
        self.alpha_range = alpha_range
        self.prob = prob
    def __call__(self, img, label):
        img_o, label_o = img, label
        if random.random() < self.prob:
            alpha = random.uniform(self.alpha_range[0], self.alpha_range[1])
            mean_val = torch.mean(img, (1,2,3), keepdim=True)
            img_o = mean_val + alpha * (img - mean_val)
            img_o = torch.clip(img_o, 0.0, 1.0)
        return img_o, label_o

class random_gamma_transform(object):
    """
    input must be normlized before gamma transform
    """
    def __init__(self, gamma_range=[0.8, 1.2], prob=0.5):
        self.gamma_range = gamma_range
        self.prob = prob

    def __call__(self, img, label):
        img_o, label_o = img, label
        if random.random() < self.prob:
            gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
            img_o = img**gamma
            label_o = label**gamma
        return img_o, label_o


class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info' : logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }

    def __init__(self,filename, level='info',
                fmt='%(asctime)s - %(levelname)s : %(message)s'):
        #create a logger
        self.logger = logging.getLogger()
        self.logger.setLevel(self.level_relations.get(level))
        format_str = logging.Formatter(fmt)

        # create a handler to input
        ch = logging.StreamHandler()
        ch.setLevel(self.level_relations.get(level))
        ch.setFormatter(format_str)

        #create a handler to filer
        fh = logging.FileHandler(filename=filename, mode='w')
        fh.setLevel(self.level_relations.get(level))
        fh.setFormatter(format_str)

        self.logger.addHandler(ch)
        self.logger.addHandler(fh)

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=0.1,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [base_lr* warmup_factor*self.gamma ** bisect_right(self.milestones, self.last_epoch)  for base_lr in self.base_lrs]
 
class LossCompose(object):

    """Composes several loss together.
    Args:
        Losses: list of Losses to compose.
    """
    def __init__(self, Losses):
        self.Losses = Losses

    def __call__(self, img, label):
        loss_dict = dict()
        for loss_f in self.Losses:
            loss = loss_f(img, label)
            loss_dict[loss_f._get_name()] = loss
        return loss_dict

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for loss in self.Losses:
            format_string += '\n'
            format_string += '    {0}'.format(loss)
        format_string += '\n)'
        return format_string


def create_tar_archive(source_folder, output_filename):
    with tarfile.open(output_filename, "w") as tar:
        for root, dirs, files in os.walk(source_folder):
            # 忽略 __pycache__ 文件夹
            if "__pycache__" in dirs:
                dirs.remove("__pycache__")
            if "train_data" in dirs:
                dirs.remove("train_data")
            
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    tar.add(file_path, arcname=os.path.relpath(file_path, source_folder))

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
    

