
import torch
import torch.nn as nn
import numpy as np
# from .cupy import *

OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'Mcs_sepConv_3x3': lambda C, stride, affine: Mcs_sepConv(C, C, 3, stride, 1, affine=affine),
    'Mcs_sepConv_5x5': lambda C, stride, affine: Mcs_sepConv1(C, C, 5, stride, 2, affine=affine),
    'Mcs_sepConv_7x7': lambda C, stride, affine: Mcs_sepConv2(C, C, 7, stride, 3, affine=affine),
}

def convolve2d_vector(arr, kernel, stride=1, padding='same'):
    kernel=[kernel,kernel,kernel]
    h, w, channel = arr.shape[0],arr.shape[1],arr.shape[2]
    k, n = kernel.shape[0], kernel.shape[2]
    r = int(k/2)
    #重新排列kernel为左乘矩阵，通道channel前置以便利用高维数组的矩阵乘法
    matrix_l = kernel.reshape((1,k*k,n)).transpose((2,0,1))
    padding_arr = np.zeros([h+k-1,w+k-1,channel])
    padding_arr[r:h+r,r:w+r] = arr
    #重新排列image为右乘矩阵，通道channel前置
    matrix_r = np.zeros((channel,k*k,h*w))
    for i in range(r,h+r,stride):
        for j in range(r,w+r,stride): 
            roi = padding_arr[i-r:i+r+1,j-r:j+r+1].reshape((k*k,1,channel)).transpose((2,0,1))
            matrix_r[:,:,(i-r)*w+j-r:(i-r)*w+j-r+1] = roi[:,::-1,:]        
    result = np.matmul(matrix_l, matrix_r)
    out = result.reshape((channel,h,w)).transpose((1,2,0))
    return out[::stride,::stride]

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




class Mcs_sepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(Mcs_sepConv, self).__init__()
        # self.op = involution(C_in,kernel_size, stride=stride) #convolve2d_vector
        self.kernel_size=kernel_size
        self.stride=stride
        
           

    def forward(self, x):
        # x = self.op(x)#convolve2d_vector
        x = convolve2d_vector(x,self.kernel_size,self.stride)#convolve2d_vector
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



class Mcs_sepConv1(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(Mcs_sepConv1, self).__init__()
        # self.op = involution(C_in,kernel_size, stride=stride)
        self.kernel_size=kernel_size
        self.stride=stride
        
           

    def forward(self, x):
        # x = self.op(x)#convolve2d_vector
        x = convolve2d_vector(x,self.kernel_size,self.stride)#convolve2d_vector
        return x



class Mcs_sepConv2(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(Mcs_sepConv2, self).__init__()
        # self.op = involution(C_in,kernel_size, stride=stride)
        self.kernel_size=kernel_size
        self.stride=stride
        
           

    def forward(self, x):
        # x = self.op(x)#convolve2d_vector
        x = convolve2d_vector(x,self.kernel_size,self.stride)#convolve2d_vector
        return x
