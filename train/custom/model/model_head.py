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
        # self.conv = nn.Conv2d(in_channels, num_class, 1)

    def forward(self, inputs):
        # TODO: 定制forward网络
        outputs = []
        for input in inputs:
            outputs.append(self.sos(input))
        return outputs

    def sos(self, data):
        return torch.sqrt(torch.sum(torch.abs(data)**2, axis=1, keepdim=True))
