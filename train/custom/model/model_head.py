import torch
import torch.nn as nn

class Model_Head(nn.Module):

    def __init__(
        self,
        in_channels: int,
        num_class: int
    ):
        super(Model_Head, self).__init__()
        # TODO: 定制Head模型
        self.conv = nn.Conv2d(in_channels, num_class, kernel_size=1)

    def forward(self, inputs):
        # TODO: 定制forward网络
        output = self.conv(inputs)
        return output


