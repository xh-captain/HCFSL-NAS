import os
import sys
import time
import glob
import logging
import time
import datetime

import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import scipy.io as sio
import PIL.Image as Image
import torch.nn.functional as F
# from tensorboardX import SummaryWriter
# from torchsummary import summary
from thop import profile	
from ptflops import get_model_complexity_info

from torch.autograd import Variable
from model import NetworkHSI as Network
from sklearn.metrics import confusion_matrix
from color_dict import color_dict
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv
from checkpoint import Checkpointer
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_str,parameter_count_table
from fvcore.nn import flop_count_table


from data_prepare import readdata
import matplotlib.pyplot as plt
import tensorflow as tf
import time
print(torch.cuda.is_available())

# os.environ["OMP_NUM_THREADS"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


parser = argparse.ArgumentParser("HSI")
parser.add_argument('--num_class', type=int, default=15, help='classes of HSI dataset')  #每个数据集类别数
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--infer_batch_size', type=int, default=32, help='infer batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=500, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')

parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')

parser.add_argument('--layers', type=int, default=3, help='total number of layers')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='HSI', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()




torch.cuda.set_device(args.gpu)
cudnn.benchmark = True
torch.manual_seed(args.seed)
cudnn.enabled=True
torch.cuda.manual_seed(args.seed)

nBand=144   #光谱通道数144-HOUSTON--103--PAVIAU---102--paviaC

focal_loss = FocalLoss(gamma=2)


# criterion = nn.CrossEntropyLoss()
# criterion = criterion.cuda()
focal_loss=focal_loss.cuda()


genotype = eval('genotypes.{}'.format(args.arch))
model = Network(nBand, args.init_channels, args.num_class, args.layers,args.auxiliary, genotype)
model = model.cuda()
model.drop_path_prob = args.drop_path_prob
logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
logging.info("genotype = %s", genotype)
print('----------------model')
              # summary(model, input_size=(144, 27, 27))
device = torch.device("cuda")
input = torch.randn(1,144, 27, 27).to(device)
#               # summary(unet, (1, 256, 256))
#               #   flop = FlopCountAnalysis(model, input)
#               # # 分析parameters
#               #   print('---------flops')
#               #   print(parameter_count_table(model))
flops, params = profile(model, inputs=(input,))
print('flops',flops)			## 打印计算量
print('params',params)	
flops, params = get_model_complexity_info(model, (144, 27, 27), as_strings=True, print_per_layer_stat=True)
print('Flops:  ' + flops)
print('Params: ' + params)
tensor = (torch.rand(1, 144, 27, 27),)

# 分析FLOPs
flops = FlopCountAnalysis(model, tensor)
print('---------')
print("FLOPs: ", flops)
print("FLOPs: ", flop.total())
              # print(flop_count_table(flop, show_param_shapes=True))
print(flop_count_str(flops))
              # print("Total", flop.total() / 1e9)

              # criterion = nn.CrossEntropyLoss()
              # criterion = criterion.cuda()




