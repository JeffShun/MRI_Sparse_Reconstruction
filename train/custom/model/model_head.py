import torch
import torch.nn as nn

class Model_Head(nn.Module):

    def forward(self, input):
        # TODO: 定制forward网络
        output = torch.abs(input)
        return output



