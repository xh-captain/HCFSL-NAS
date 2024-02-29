import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
import numpy as np
import os
import math
import argparse
import scipy as sp
import scipy.stats
import pickle
import random
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time
import utils
# import models
# import spectral
import logging
import sys
from torchsummary import summary
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist
from torch.autograd import Variable
from darts.model_search import Network as darts_encoder
from darts.architect import Architect
import darts.genotypes as genotypes
import task_generator_mini as tg
from darts.checkpoint import Checkpointer
from tensorboardX import SummaryWriter
from utils import AdaSPLoss    

parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 256)
parser.add_argument("-c","--src_input_dim",type = int, default = 128)   
parser.add_argument("-c2","--mini_input_dim",type = int, default = 3)   
parser.add_argument("-n","--n_dim",type = int, default = 100)   
parser.add_argument("-w","--class_num",type = int, default = 9)
parser.add_argument("-s","--shot_num_per_class",type = int, default = 1)
parser.add_argument("-b","--query_num_per_class",type = int, default = 9)  
parser.add_argument("-e","--episode",type = int, default= 500)
parser.add_argument("-t","--test_episode", type = int, default = 600)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)  
parser.add_argument("-u","--hidden_unit",type=int,default=10)
parser.add_argument("-m","--test_class_num",type=int, default=9)
parser.add_argument('--tempsum', type=float, default=0.2, help='tempsum')
parser.add_argument("-z","--test_lsample_num_per_class",type=int,default=5, help='5 4 3 2 1')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--layers', type=int, default=2, help='total number of layers')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--seeds', type=int, default=1337, help='random seed')
parser.add_argument('--arch', type=str, default='HSI', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=0.001, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=3e-4, help='weight decay for arch encoding')
args = parser.parse_args(args=[])

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler('./result/search.txt')  #paviaU
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
# dataname='PU' gpu

# Hyper Parameters
FEATURE_DIM = args.feature_dim
SRC_INPUT_DIMENSION = args.src_input_dim
MINI_INPUT_DIMENSION=args.mini_input_dim
# TAR_INPUT_DIMENSION = args.tar_input_dim
N_DIMENSION = args.n_dim
CLASS_NUM = args.class_num
SHOT_NUM_PER_CLASS = args.shot_num_per_class
QUERY_NUM_PER_CLASS = args.query_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
# GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit

TEST_CLASS_NUM = args.test_class_num # the number of class
TEST_LSAMPLE_NUM_PER_CLASS = args.test_lsample_num_per_class # the number of labeled samples per class 5 4 3 2 1
genotype = eval('genotypes.{}'.format(args.arch))
nBand=100
utils.same_seeds(0)
def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('classificationMap'):
        os.makedirs('classificationMap')
_init_()

# load source domain data set
with open(os.path.join('datasets',  'Chikusei_imdb_128.pickle'), 'rb') as handle:
    source_imdb = pickle.load(handle)
logging.info(source_imdb.keys())
logging.info(source_imdb['Labels'])

# process source domain data set  ---CK-data set 2517*2335*128  19类  77592个样本
data_train = source_imdb['data'] # (77592, 9, 9, 128)
labels_train = source_imdb['Labels'] # 77592
logging.info(data_train.shape)
logging.info(labels_train.shape)
keys_all_train = sorted(list(set(labels_train)))  # class [0,...,18] batch
logging.info(keys_all_train) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
label_encoder_train = {}  
for i in range(len(keys_all_train)):
    label_encoder_train[keys_all_train[i]] = i
logging.info(label_encoder_train)

train_set = {}
for class_, path in zip(labels_train, data_train):
    if label_encoder_train[class_] not in train_set:
        train_set[label_encoder_train[class_]] = []
    train_set[label_encoder_train[class_]].append(path)
logging.info(train_set.keys())
data = train_set
del train_set
del keys_all_train
del label_encoder_train

logging.info("Num classes for source domain datasets: " + str(len(data)))
logging.info(data.keys())
data = utils.sanity_check(data) # 200 labels samples per class
logging.info("Num classes of the number of class larger than 200: " + str(len(data)))

for class_ in data:
    for i in range(len(data[class_])):
        image_transpose = np.transpose(data[class_][i], (2, 0, 1))  # （9,9,100）-> (100,9,9)
        data[class_][i] = image_transpose

# source few-shot classification data
metatrain_data = data
logging.info(len(metatrain_data.keys()), metatrain_data.keys())
del data

# source domain adaptation data
logging.info(source_imdb['data'].shape) # (77592, 9, 9, 100) size
source_imdb['data'] = source_imdb['data'].transpose((1, 2, 3, 0)) #(9, 9, 100, 77592)
logging.info(source_imdb['data'].shape) # (77592, 9, 9, 100)
logging.info(source_imdb['Labels'])
source_dataset = utils.matcifar(source_imdb, train=True, d=3, medicinal=0)
source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=32, shuffle=True, num_workers=0)
del source_dataset, source_imdb

class Mapping(nn.Module):
    def __init__(self, in_dimension, out_dimension):
        super(Mapping, self).__init__()
        self.preconv = nn.Conv2d(in_dimension, out_dimension, 1, 1, bias=False)
        self.preconv_bn = nn.BatchNorm2d(out_dimension)

    def forward(self, x):
        x = self.preconv(x)
        x = self.preconv_bn(x)
        return x


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data = torch.ones(m.bias.data.size())



def euclidean_metric(a, b):    
    n = a.shape[0]  #171
    m = b.shape[0]  #9
    a = a.unsqueeze(1).expand(n, m, -1) 
    b = b.unsqueeze(0).expand(n, m, -1)  
    logits = -((a - b)**2).sum(dim=2)  
    return logits

nDataSet = 1
acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, CLASS_NUM])
k = np.zeros([nDataSet, 1])
best_predict_all = []
best_acc_all = 0.0
# best_G,best_RandPerm,best_Row, best_Column,best_nTrain = None,None,None,None,None

seeds = args.seeds
for iDataSet in range(nDataSet):
    np.random.seed(seeds)
    # train_loader, test_loader, target_da_metatrain_data, target_loader,G,RandPerm,Row, Column,nTrain = get_target_dataset(
    #     Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth,class_num=TEST_CLASS_NUM, shot_num_per_class=TEST_LSAMPLE_NUM_PER_CLASS)
    
    metatrain_folders, metatest_folders = tg.mini_imagenet_folders()

    # model
    feature_encoder =  darts_encoder(args.n_dim, args.init_channels, args.class_num, args.layers, crossEntropy)#Network()   #网络降维提取的特征
    
    feature_encoder.cuda()
    feature_encoder.apply(weights_init)   

    feature_encoder.train()
    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=args.learning_rate,weight_decay=args.weight_decay)
    '''neural architecture   architecture params and weight of encoder network'''
    scheduler = torch.optim.lr_scheduler.StepLR(feature_encoder_optim, step_size=args.episode // 5, gamma=0.25)
    architect = Architect(feature_encoder, args)
    logging.info("Training...")


    last_accuracy = 0.0
    best_episdoe = 0
    train_loss = []
    test_acc = []
    running_D_loss, running_F_loss = 0.0, 0.0
    running_label_loss = 0
    running_domain_loss = 0
    total_hit, total_num = 0.0, 0.0
    test_acc_list = []
    min_valid_obj = 100

    source_iter = iter(source_loader)
    train_start = time.time()
    for episode in range(500):   
        '''Few-shot claification for ck domain data set'''
        task = utils.Task(metatrain_data, CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)  
        support_dataloader = utils.get_HBKC_data_loader(task, num_per_class=SHOT_NUM_PER_CLASS, split="train", shuffle=False)
        query_dataloader = utils.get_HBKC_data_loader(task, num_per_class=QUERY_NUM_PER_CLASS, split="test", shuffle=True)

        '''Few-shot claification for mini-imagenet data set'''
        task = tg.MiniImagenetTask(metatrain_folders, CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)   
        minisupport_dataloader = tg.get_mini_imagenet_data_loader(task, num_per_class=SHOT_NUM_PER_CLASS, split="train", shuffle=False)
        miniquery_dataloader = tg.get_mini_imagenet_data_loader(task, num_per_class=QUERY_NUM_PER_CLASS, split="test", shuffle=True)
        
        # sample datas
        supports, support_labels = support_dataloader.__iter__().next()   
        querys, query_labels = query_dataloader.__iter__().next()  
        supports = Variable(supports, requires_grad=False).cuda()
        querys = Variable(querys, requires_grad=False).cuda()


        minisupports, minisupport_labels = minisupport_dataloader.__iter__().next()  
        miniquerys, miniquery_labels = miniquery_dataloader.__iter__().next()
        minisupports = Variable(minisupports, requires_grad=False).cuda()
        miniquerys = Variable(miniquerys, requires_grad=False).cuda()

        # Update parameters
        architect.step(minisupports, miniquerys, miniquery_labels, args.learning_rate, feature_encoder_optim, unrolled=args.unrolled)
      
        feature_encoder_optim.zero_grad()
        support_features = feature_encoder(supports,domain='ck')   
        query_features = feature_encoder(querys,domain='ck')   
         # Prototype network
        if SHOT_NUM_PER_CLASS > 1:
            support_proto = support_features.reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)   
        else:
            support_proto = support_features

        logits = euclidean_metric(query_features, support_proto)   
        loss = crossEntropy(logits, query_labels.cuda()) + adaploss(logits, query_labels.cuda())
        

        loss.backward()
        feature_encoder_optim.step()

        total_hit += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels).item()
        total_num += querys.shape[0]
       
        train_loss.append(loss.item())
        logging.info('episode {:>3d}:  fsl loss: {:6.4f}, acc {:6.4f}, loss: {:6.4f}'.format(episode + 1, \
                                                            loss.item(), total_hit / total_num,loss.item()))
        if loss < min_valid_obj:
            torch.save(feature_encoder.state_dict(),str( "checkpoints/search/searched_encoder_pu_iter_" + str(episode) +"shot.pth"))
            genotype = feature_encoder.genotype()
            logging.info('genotype=%s', genotype)
            min_valid_obj = loss
    torch.save(feature_encoder.state_dict(),str( "checkpoints/search/searched_encoder_pu_iter_final_shot.pth"))  
    logging.info('Searched Neural Architecture:')
    logging.info('genotype=%s', genotype)
      
