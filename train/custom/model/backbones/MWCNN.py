import torch
import torch.nn as nn

def conv(in_channels, out_channels, kernel_size=3, bias=True, dilation=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2)+dilation-1, bias=bias, dilation=dilation)

def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    #print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    
    h = torch.zeros([out_batch, out_channel, out_height, out_width]).to(x.dtype).to(x.device)

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
    return h

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


class BBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, bias=True, bn=False, act=nn.ReLU(inplace=True)):
        super(BBlock, self).__init__()
        m = []
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        self.pipe = nn.Sequential(*m)

    def forward(self, x):
        x = self.pipe(x)
        return x

class DBlock_com(nn.Module):
    def __init__(
        self, in_channels, out_channels, dilations=[2, 1], kernel_size=3, bias=True, bn=False, act=nn.ReLU(inplace=True)):

        super(DBlock_com, self).__init__()
        m = []
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=dilations[0]))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=dilations[1]))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        self.pipe = nn.Sequential(*m)

    def forward(self, x):
        x = self.pipe(x)
        return x

class DBlock_inv(nn.Module):
    def __init__(
        self, in_channels, out_channels, dilations=[2, 1], kernel_size=3, bias=True, bn=False, act=nn.ReLU(inplace=True)):

        super(DBlock_inv, self).__init__()
        m = []
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=dilations[0]))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=dilations[1]))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        self.pipe = nn.Sequential(*m)

    def forward(self, x):
        x = self.pipe(x)
        return x


class MWCNN(nn.Module):
    def __init__(self, in_ch, channels):
        super(MWCNN, self).__init__()

        self.DWT = DWT()
        self.IWT = IWT()

        m_head = [BBlock(in_ch, channels)]
        d_l0 = []
        d_l0.append(DBlock_com(channels, channels, dilations=[2, 1]))


        d_l1 = [BBlock(channels * 4, channels * 2)]
        d_l1.append(DBlock_com(channels * 2, channels * 2, dilations=[2, 1]))

        d_l2 = []
        d_l2.append(BBlock(channels * 8, channels * 4))
        d_l2.append(DBlock_com(channels * 4, channels * 4, dilations=[2, 1]))
        pro_l3 = []
        pro_l3.append(BBlock(channels * 16, channels * 8))
        pro_l3.append(DBlock_com(channels * 8, channels * 8, dilations=[2, 3]))
        pro_l3.append(DBlock_inv(channels * 8, channels * 8, dilations=[3, 2]))
        pro_l3.append(BBlock(channels * 8, channels * 16))

        i_l2 = [DBlock_inv(channels * 4, channels * 4, dilations=[2, 1])]
        i_l2.append(BBlock(channels * 4, channels * 8))

        i_l1 = [DBlock_inv(channels * 2, channels * 2, dilations=[2, 1])]
        i_l1.append(BBlock(channels * 2, channels * 4))

        i_l0 = [DBlock_inv(channels, channels, dilations=[2, 1])]

        self.head = nn.Sequential(*m_head)
        self.d_l2 = nn.Sequential(*d_l2)
        self.d_l1 = nn.Sequential(*d_l1)
        self.d_l0 = nn.Sequential(*d_l0)
        self.pro_l3 = nn.Sequential(*pro_l3)
        self.i_l2 = nn.Sequential(*i_l2)
        self.i_l1 = nn.Sequential(*i_l1)
        self.i_l0 = nn.Sequential(*i_l0)

    def forward(self, x):
        x0 = self.d_l0(self.head(x))
        x1 = self.d_l1(self.DWT(x0))
        x2 = self.d_l2(self.DWT(x1))
        x_ = self.IWT(self.pro_l3(self.DWT(x2))) + x2
        x_ = self.IWT(self.i_l2(x_)) + x1
        x_ = self.IWT(self.i_l1(x_)) + x0
        out = self.i_l0(x_)

        return out

if __name__ == '__main__':
    model = MWCNN(30, 32).cuda()
    input = torch.rand(8,30,320,320).cuda()
    out = model(input)
    print(out.shape)
