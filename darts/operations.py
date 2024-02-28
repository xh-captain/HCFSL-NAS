
import torch
import torch.nn as nn
import numpy as np

OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'Mc_sepConv_3x3': lambda C, stride, affine: Mc_sepConv(C, C, 3, stride, 1, affine=affine),
    'Mc_sepConv_5x5': lambda C, stride, affine: Mc_sepConv1(C, C, 5, stride, 2, affine=affine),
    'Mc_sepConv_7x7': lambda C, stride, affine: Mc_sepConv2(C, C, 7, stride, 3, affine=affine),
}

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)




class Mc_sepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(Mc_sepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Dropout(0.5),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=(kernel_size,1), stride=(stride,1), padding=(padding,0), groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=(1,kernel_size), stride=(1,stride), padding=(0,padding), groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
           
        )
        self.ca = ChannelAttention(C_in)
        self.ca1 = ChannelAttention(C_out)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.op(x)
        x = self.ca1(x) * x
        return x


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, :, :])], dim=1)
        out = self.bn(out)
        return out



class Mc_sepConv1(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(Mc_sepConv1, self).__init__()
        self.op = nn.Sequential(
            nn.Dropout(0.5),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=(kernel_size,1), stride=(stride,1), padding=(padding,0), groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=(1,kernel_size), stride=(1,stride), padding=(0,padding), groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
           
        )
        self.ca = ChannelAttention(C_in)
        self.ca1 = ChannelAttention(C_out)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.op(x)
        x = self.ca1(x) * x
        return x



class Mc_sepConv2(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(Mc_sepConv2, self).__init__()
        self.op = nn.Sequential(
            nn.Dropout(0.5),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=(kernel_size,1), stride=(stride,1), padding=(padding,0), groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=(1,kernel_size), stride=(1,stride), padding=(0,padding), groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
           
        )
        self.ca = ChannelAttention(C_in)
        self.ca1 = ChannelAttention(C_out)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.op(x)
        x = self.ca1(x) * x
        return x
