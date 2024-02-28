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
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import scipy.io as sio
from torch.autograd import Variable
from model import NetworkHSI as Network
from sklearn.metrics import confusion_matrix
from data_prepare import readdata

os.environ['CUDA_VISIBLE_DEVICES'] = '7'


parser = argparse.ArgumentParser("HSI")
parser.add_argument('--num_class', type=int, default=9, help='classes of HSI dataset')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--infer_batch_size', type=int, default=100, help='infer batch size')
parser.add_argument('--learning_rate', type=float, default=0.004, help='init learning rate')
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


log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler('./result/log_infe.txt')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


# nband=103, nclass=9
image_file = r'pavia/PaviaU.mat'
label_file = r'pavia/PaviaU_gt.mat'

def main(genotype, seed):

  data, shuffle_number = readdata(image_file, label_file, train_nsamples=200, validation_nsamples=100,
                                  windowsize=27, istraining=True, rand_seed=seed)
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)

  nBand=103   #光谱通道数

  model = Network(nBand, args.init_channels, args.num_class, args.layers,args.auxiliary, genotype)  #定义模型
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()

  optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs // 5, gamma=0.5)


  scheduler.step()
  model.drop_path_prob = args.drop_path_prob



    

  # utils.load(model, './result/weights.pt')           #调用训练好的模型
  print('shuffle_number:  ',shuffle_number)
  trained_model_dir='models/model/model_best.pth'
  model_state_dict = torch.load(trained_model_dir).pop("model")
  try:
    model.load_state_dict(model_state_dict)
  except:
    model.module.load_state_dict(model_state_dict)

  model.eval()
  test_model(model, shuffle_number, seed)

  return matrix




def test_model(model, shuffle_number, seed):
  model.eval()
  total_tar = np.array([])
  total_pre = np.array([])

  i = 0
  test_nsamples = 0    #所有测试样本数，做记录，最后的值为总的分类图的样本数
  while 1:                                      #用所有的样本，做分类图的可视化
    data = readdata(image_file, label_file, train_nsamples=0, validation_nsamples=0,
                    windowsize=27, istraining=False, shuffle_number=shuffle_number, times=i, rand_seed=seed)
    if data == None:
    #   matrix = confusion_matrix(total_tar, total_pre)
    #   return 
        print('--------------------------------------------------总共的处理样本数量')
        print(test_nsamples)
        print('--------------------------------------------------预测出来的值')
        print(total_pre)
        sio.savemat("./index/total_pre.mat", {"total_pre":total_pre})   #对应位置预测出来的值
        print('test pre map index ')
        break

    test_nsamples += data.test.num_examples

    x_test, y_test = data.test.images, data.test.labels
    print('--------------------111--------x_test.shape-------------------y_test.shape-----')
    print(x_test.shape)   #10016*103*27*27
    print(y_test.shape)   #10016*9
    add_samples = args.batch_size - data.test.num_examples % args.batch_size   #一次做10000个，最后的那次，不足10000个的时候
    x_test = np.concatenate((x_test[0:add_samples, :, :, :], x_test), axis=0)
    y_test = np.concatenate((y_test[0:add_samples, :], y_test), axis=0)


    print('----------------------------x_test.shape-------------------y_test.shape-----')
    print(x_test.shape)   #10016*103*27*27
    print(y_test.shape)   #10016*9

    tar = np.array([])
    pre = np.array([])
    total_batch = (data.test.num_examples + add_samples) // args.batch_size           #bs--32
    for j in range(total_batch):
      input, target = x_test[j*args.batch_size:(j+1)*args.batch_size, :, :, :], y_test[j*args.batch_size:(j+1)*args.batch_size, :]
      input = Variable(torch.from_numpy(input), volatile=True).cuda()
      target = Variable(torch.from_numpy(np.argmax(target, axis=1)), volatile=True).cuda(async=True)

      logits = model(input)

      _, t, p = utils.accuracy(logits, target, topk=(1,))
      tar = np.append(tar, t.data.cpu().numpy())
      pre = np.append(pre, p.data.cpu().numpy())

    total_tar = np.append(total_tar, tar[add_samples:][add_samples:])
    total_pre = np.append(total_pre, pre[add_samples:][add_samples:])
    print('-----------------------------------------------total_tar------------')
    print(total_tar.shape)
    print('----------------------------------------------这个是最后42476个点的类别记录-------------total_pre-')
    print(total_pre.shape)   #42776-300=42476  （10000,10000,10000,10000,2476）
    i = i + 1



if __name__ == '__main__':
  genotype = eval('genotypes.{}'.format(args.arch))
  main(genotype=genotype, seed=np.random.randint(low=0, high=10000, size=1))

#   OA, AA_mean, Kappa, AA = cal_results(matrix)
  print(OA)
