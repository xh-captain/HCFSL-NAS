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
from tensorboardX import SummaryWriter
from thop import profile	

from torch.autograd import Variable
from model import NetworkHSI as Network
from sklearn.metrics import confusion_matrix
from color_dict import color_dict
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv
from checkpoint import Checkpointer

from data_all import readdata
import matplotlib.pyplot as plt
import tensorflow as tf

os.environ["OMP_NUM_THREADS"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


parser = argparse.ArgumentParser("HSI")
parser.add_argument('--num_class', type=int, default=15, help='classes of HSI dataset')  #每个数据集类别数
parser.add_argument('--batch_size', type=int, default=32, help='batch size')   #64
parser.add_argument('--infer_batch_size', type=int, default=32, help='infer batch size')
parser.add_argument('--test_batch_size', type=int, default=100, help='infer batch size')
parser.add_argument('--learning_rate', type=float, default=0.004, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=1000, help='num of training epochs')  #300 500-11-13
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')

parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')

parser.add_argument('--layers', type=int, default=3, help='total number of layers')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='HSI', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()


WDs=1
sts=15029   #664845
way='per'
kcount=2 #2  67
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler('./result/log_test.txt')  #paviaU
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)



image_file = r'data/houston/houston.mat'  #145*145*200
label_file = r'data/houston/houston_gt_sum.mat'  #最大值都是16  sum-10249
dataset1='houston'   #salinas  botswana


def main(genotype, seed):

  data, shuffle_number = readdata(image_file, label_file, train_nsamples=0, validation_nsamples=0,
                                  windowsize=WDs, istraining=True, rand_seed=seed)
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  torch.set_num_threads(1)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)

  nBand=144   #光谱通道数144-HOUSTON--103--PAVIAU---102--paviaC

  


  model = Network(nBand, args.init_channels, args.num_class, args.layers,args.auxiliary, genotype)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
  model.drop_path_prob = args.drop_path_prob
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
  logging.info("genotype = %s", genotype)
  print('----------------model')
              # summary(model, input_size=(144, 27, 27))
  device = torch.device("cuda")
  input = torch.randn(1,144, WDs, WDs).to(device)
  flops, params = profile(model, inputs=(input,))
  print('flops',flops)			## 打印计算量
  print('params',params)	

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()

  optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs // 5, gamma=0.5)
  model.drop_path_prob = args.drop_path_prob

  model.drop_path_prob = args.drop_path_prob 
  trained_model_dir='models/model-'+str(WDs)+'W/model_best.pth'
  model_state_dict = torch.load(trained_model_dir).pop("model")
  model.load_state_dict(model_state_dict)


  model.eval()

  matrix= test_all_model(model, shuffle_number, seed)



  return matrix




def test_all_model(model, shuffle_number, seed):
  model.eval()
  total_tar = np.array([])
  total_pre = np.array([])

  i = 0
  test_nsamples = 0
  while 1:
    data = readdata(image_file, label_file, train_nsamples=0, validation_nsamples=0,
                    windowsize=WDs, istraining=False, shuffle_number=shuffle_number, times=i, rand_seed=seed)
    if data == None:
      matrix = confusion_matrix(total_tar, total_pre)
      print('test_nsamples:',test_nsamples)
      if(test_nsamples==sts):   #42276----15029-----148152  ---5211
        sio.savemat("index/houston_map/total_pre.mat", {"total_pre":total_pre})
      return matrix

    test_nsamples += data.test.num_examples

    x_test, y_test = data.test.images, data.test.labels
    add_samples = args.batch_size - data.test.num_examples % args.batch_size
    x_test = np.concatenate((x_test[0:add_samples, :, :, :], x_test), axis=0)
    y_test = np.concatenate((y_test[0:add_samples, :], y_test), axis=0)

    tar = np.array([])
    pre = np.array([])
    total_batch = (data.test.num_examples + add_samples) // args.batch_size
    for j in range(total_batch):
      input, target = x_test[j*args.batch_size:(j+1)*args.batch_size, :, :, :], y_test[j*args.batch_size:(j+1)*args.batch_size, :]
      input = Variable(torch.from_numpy(input), volatile=True).cuda()
      target = Variable(torch.from_numpy(np.argmax(target, axis=1)), volatile=True).cuda(async=True)
      # print(input.shape)
      logits = model(input)

      _, t, p = utils.accuracy(logits, target, topk=(1,))
      # print(p)
      tar = np.append(tar, t.data.cpu().numpy())
      pre = np.append(pre, p.data.cpu().numpy())


    total_tar = np.append(total_tar, tar[add_samples:])
    total_pre = np.append(total_pre, pre[add_samples:])
    i = i + 1


def labelmap_2_img(color_list, label_map):
    h, w =349,1905      #349,1905 #512,217          #1096,715
    img = np.zeros((h, w, 3))
    for i in range(h):
        for j in range(w):
            R,G,B = color_list[str(label_map[i, j])]
            img[i,j] = [R, G, B]
    return np.array(img, np.uint8)



if __name__ == '__main__':
  torch.set_num_threads(1)
  genotype = eval('genotypes.{}'.format(args.arch))
  matrix= main(genotype=genotype, seed=0)
  
  windowsize=WDs
  PATCH_SIZE=windowsize/2
 

  outputs = np.zeros((349,1905))   #结果图---1096*715
  data_path = os.path.join(os.getcwd(), 'index/houston_map/')  # houston_map
  total_pre = sio.loadmat(os.path.join(data_path,'total_pre.mat'))['total_pre']   #42776个像素点的预测结果   1*42776

 

  print(total_pre.shape)
  k=0
  for i in range(0,kcount):  #2  32  15029
      index_r = sio.loadmat(os.path.join(data_path,str(i)+'index_r.mat'))['index_r']   #42776个像素点行的位置------第i个批次的
      index_c = sio.loadmat(os.path.join(data_path,str(i)+'index_c.mat'))['index_c']   #42776个像素点列的位置
      for j in range(index_r.shape[0]-1):
        r=int(index_r[j+1][0]-PATCH_SIZE)
        c=int(index_c[j+1][0]-PATCH_SIZE)
        outputs[r][c]=total_pre[0][k]+1
        k+=1


  label_data = sio.loadmat(label_file)
  label = label_data['houston_gt_sum']  # pavia


  data_set='HoustonU'
  result_save_dir='result/'
  outputs=outputs.astype(int)
  img_result = labelmap_2_img(color_dict[data_set], np.array(outputs))
  img = Image.fromarray(img_result)
  img.save(os.path.join(result_save_dir, str(WDs)+way+'{}.png'.format(data_set)))
  # test_map=test_map.astype(int)
  gt_result = labelmap_2_img(color_dict[data_set], np.array(label))
  img_gt = Image.fromarray(gt_result)
  img_gt.save(os.path.join(result_save_dir, 'gt{}.png'.format(data_set)))
