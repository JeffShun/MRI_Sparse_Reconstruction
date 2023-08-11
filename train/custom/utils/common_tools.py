import random
import torch
from torch.nn import functional as F
import math
import numpy as np
import logging
from bisect import bisect_right
import torch.nn as nn


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
    def _normlize_real(self, data):
        ori_shape = data.shape
        data_o = data.reshape(ori_shape[0], -1)
        data_min = data_o.min(dim=-1,keepdim=True)[0]
        data_max = data_o.max(dim=-1,keepdim=True)[0]
        data_o = (data_o - data_min)/(data_max - data_min)
        data_o = data_o.reshape(ori_shape)
        return data_o

    def _normlize_complex(self, data):
        ori_shape = data.shape
        data_o = data.reshape(ori_shape[0], -1)
        data_modulus = torch.abs(data_o)
        modulus_max = data_modulus.max(dim=-1,keepdim=True)[0]
        data_o = data_o / modulus_max
        data_o = data_o.reshape(ori_shape)
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
    def _complex_to_multichannel(self, data):
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
