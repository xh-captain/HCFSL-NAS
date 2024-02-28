import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import scipy.io as sio
from torch.autograd import Variable
from model_search import Network
from architect import Architect
from sklearn.metrics import confusion_matrix
from data_prepare import readdata
import random

import datetime

from tensorboardX import SummaryWriter

os.environ["OMP_NUM_THREADS"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = '4'


parser = argparse.ArgumentParser("HSI")
parser.add_argument('--num_class', type=int, default=15, help='classes of HSI dataset')  #9,15,16,13每个数据集的类别
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=4e-3, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=2, help='total number of layers')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--unrolled', action='store_true', default=True, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=3e-4, help='weight decay for arch encoding')
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler('./result/houston/result/log.txt')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


# nband=144, nclass=15
image_file = r'data/houston/houston.mat'  #349*1905*344
label_file = r'data/houston/houston_gt_sum.mat'  #最大值都是15  tr-2832,te-12197-sum-15029
# image_file = r'data/paviaC/Pavia.mat'  #349*1905*344
# label_file = r'data/paviaC/Pavia_gt.mat'  #最大值都是15  tr-2832,te-12197-sum-15029
#image_file = r'data/paviaU/PaviaU.mat'  #349*1905*344
#label_file = r'data/paviaU/PaviaU_gt.mat'  #最大值都是15  tr-2832,te-12197-sum-15
# image_file = r'data/salinas/Salinas_corrected.mat'  #512*217*204
# label_file = r'data/salinas/Salinas_gt.mat'  #最大值都是9  sum-54129
# image_file = r'data/indian/Indian_pines_corrected.mat'  #145*145*200
# label_file = r'data/indian/Indian_pines_gt.mat'  #最大值都是16  sum-10249
dataset='paviaU'   #salinas  indian  paviaC


import torchvision
# import torchvision.transforms as F
import torch.nn.functional as F


from IPython.display import display
criterion = nn.CrossEntropyLoss()
criterion = criterion.cuda()  #-------def focal_loss(y_true, y_pred):



class FocalLoss(nn.Module):

    def __init__(self, weight=None, reduction='mean', gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

def one_hot(x, class_count):
    return torch.eye(class_count)[x,:]


class CELoss(nn.Module):
    ''' Cross Entropy Loss with label smoothing '''
    def __init__(self, label_smooth=None, class_num=16):
        super().__init__()
        self.label_smooth = label_smooth
        self.class_num = class_num

    def forward(self, pred, target):
      ''' 
      Args:
      pred: prediction of model output    [N, M]
      target: ground truth of sampler [N]
      '''
      eps = 1e-12
      
      if self.label_smooth is not None:
        # cross entropy loss with label smoothing
        logprobs = F.log_softmax(pred, dim=1)	# softmax + log
        # print(logprobs.shape,target.shape)
        
        target = one_hot(target,self.class_num)
        logprobs= logprobs.cuda()
        target= target.cuda()

        
        target = torch.clamp(target.float(), min=self.label_smooth/(self.class_num-1), max=1.0-self.label_smooth)
        loss = -1*torch.sum(target*logprobs, 1)
        
        
      else:
        # standard cross entropy loss
        loss = -1.*pred.gather(1, target.unsqueeze(-1)) + torch.log(torch.exp(pred+eps).sum(dim=1))

      return loss.mean()

smooth_loss = CELoss(label_smooth=0.05, class_num=15)
smooth_loss=smooth_loss.cuda()

def main(seed):

  data, shuffle_number = readdata(image_file, label_file, train_nsamples=20, validation_nsamples=10,
                                  windowsize=9, istraining=True, rand_seed=seed)
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  torch.set_num_threads(1)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(random.randint(1, 10000))
  cudnn.enabled=True
  torch.cuda.manual_seed(random.randint(1, 10000))

  nBand=144  #200-indian  #204--salinas      #光谱通道数-144-houston,103-paviaU,102-paviaC,salinas-204

  focal_loss = FocalLoss(gamma=2)


  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  focal_loss=focal_loss.cuda()
  model = Network(nBand,args.init_channels, args.num_class, args.layers, criterion)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs // 5, gamma=0.25)

  architect = Architect(model, args)

  min_valid_obj = 100

  genotype = model.genotype()
  print('genotype = ', genotype)

  writer = SummaryWriter('models/log_search', comment=dataset)
  start_training_time = time.time()

  for epoch in range(args.epochs):
    tic = time.time()
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %03d lr %e', epoch+1, lr)

    # training
    train_acc, train_obj, tar, pre = train(data.train, data.validation, model, architect, focal_loss, optimizer, lr)

    # validation
    valid_acc, valid_obj, tar_v, pre_v = infer(data.validation, model, focal_loss)

    toc = time.time()

    logging.info('Epoch %03d: train_loss = %f, train_acc = %f, val_loss = %f, val_acc = %f, time = %f', epoch + 1, train_obj, train_acc, valid_obj, valid_acc, toc-tic)
    writer.add_scalars('Search_acc1', { 'train_acc': train_acc}, epoch+1)
    writer.add_scalars('Search_loss1', {'train_loss': train_obj}, epoch+1)
    writer.add_scalars('Search_acc', { 'val_acc': valid_acc}, epoch+1)
    writer.add_scalars('Search_loss', {'val_loss': valid_obj}, epoch+1)
   

    if valid_obj < min_valid_obj:
      genotype = model.genotype()
      logging.info('genotype=%s', genotype)
      print('genotype = ', genotype)
      min_valid_obj = valid_obj

  total_training_time = time.time() - start_training_time
  total_time_str = str(datetime.timedelta(seconds=total_training_time))
  logging.info("Total training time: {}".format(total_time_str)) 

  return genotype


def train(train_data, valid_data, model, architect, focal_loss, optimizer, lr):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  tar = np.array([])
  pre = np.array([])

  total_batch = int(train_data.num_examples / args.batch_size)
  for i in range(total_batch):
    input, target = train_data.next_batch(args.batch_size)

    model.train()
    n = input.shape[0]

    input = Variable(torch.from_numpy(input), requires_grad=False).cuda()
    target = Variable(torch.from_numpy(np.argmax(target, axis=1)), requires_grad=False).cuda(async=True)

    input_search, target_search = valid_data.next_batch(args.batch_size)
    input_search = Variable(torch.from_numpy(input_search), requires_grad=False).cuda()
    target_search = Variable(torch.from_numpy(np.argmax(target_search, axis=1)), requires_grad=False).cuda(async=True)

    architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

    optimizer.zero_grad()
    logits = model(input)
    # loss = criterion(logits, target)
    loss1 = focal_loss(logits, target)
    loss2 = smooth_loss(logits, target)
    # loss=focal_loss+criterion
    loss=loss1+loss2

    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, t, p = utils.accuracy(logits, target, topk=(1, ))
    objs.update(loss.item(), n)
    top1.update(prec1[0].item(), n)
    tar = np.append(tar, t.data.cpu().numpy())
    pre = np.append(pre, p.data.cpu().numpy())

  return top1.avg, objs.avg, tar, pre


def infer(valid_data, model, focal_loss):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  model.eval()
  tar = np.array([])
  pre = np.array([])

  total_batch = valid_data.num_examples // args.batch_size
  for i in range(total_batch):
    input, target = valid_data.next_batch(args.batch_size)
    n = input.shape[0]

    input = Variable(torch.from_numpy(input), volatile=True).cuda()
    target = Variable(torch.from_numpy(np.argmax(target, axis=1)), volatile=True).cuda(async=True)

    logits = model(input)
    # loss = criterion(logits, target)
    loss1 = focal_loss(logits, target)  
    loss2 = smooth_loss(logits, target)
    # loss=focal_loss+criterion
    loss=loss1+loss2

    prec1, t, p = utils.accuracy(logits, target, topk=(1, ))

    objs.update(loss.item(), n)
    top1.update(prec1[0].item(), n)
    tar = np.append(tar, t.data.cpu().numpy())
    pre = np.append(pre, p.data.cpu().numpy())

  return top1.avg, objs.avg, tar, pre




if __name__ == '__main__':
  genotype = main(seed=np.random.randint(low=0, high=10000, size=1))
  print('Searched Neural Architecture:')
  print(genotype)

