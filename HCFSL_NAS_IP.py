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
from darts.model import NetworkHSI as darts_encoder
import darts.genotypes as genotypes
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist
import task_generator_mini as tg
from darts.checkpoint import Checkpointer
from tensorboardX import SummaryWriter
from torchsummary import summary
from thop import profile	
 
from utils import AdaSPLoss    
 

os.environ["OMP_NUM_THREADS"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 256)
parser.add_argument("-c","--src_input_dim",type = int, default = 128)  
parser.add_argument("-c2","--mini_input_dim",type = int, default = 3)   
parser.add_argument("-d","--tar_input_dim",type = int, default = 200)  
parser.add_argument("-n","--n_dim",type = int, default = 100)   
parser.add_argument("-w","--class_num",type = int, default = 16)
parser.add_argument("-s","--shot_num_per_class",type = int, default = 1)
parser.add_argument("-b","--query_num_per_class",type = int, default = 9)  
parser.add_argument("-e","--episode",type = int, default= 20000) 
parser.add_argument("-t","--test_episode", type = int, default = 600)
parser.add_argument("-l","--learning_rate", type = float, default = 0.004)  
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
# target
parser.add_argument('--tempsum', type=float, default=0.2, help='tempsum')
parser.add_argument("-m","--test_class_num",type=int, default=16)
parser.add_argument("-z","--test_lsample_num_per_class",type=int,default=5, help='5 4 3 2 1') 
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--layers', type=int, default=2, help='total number of layers')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--arch', type=str, default='HSI', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args(args=[])
seeds = args.seed
 
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler('./result/log_test-train.txt')  #paviaU
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
genotype = eval('genotypes.{}'.format(args.arch))
logging.info('genotype=%s', genotype)
dataname='ip'

# Hyper Parameters
FEATURE_DIM = args.feature_dim
SRC_INPUT_DIMENSION = args.src_input_dim
MINI_INPUT_DIMENSION=args.mini_input_dim
TAR_INPUT_DIMENSION = args.tar_input_dim
N_DIMENSION = args.n_dim
CLASS_NUM = args.class_num
SHOT_NUM_PER_CLASS = args.shot_num_per_class
QUERY_NUM_PER_CLASS = args.query_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit

TEST_CLASS_NUM = args.test_class_num  
TEST_LSAMPLE_NUM_PER_CLASS = args.test_lsample_num_per_class  
utils.same_seeds(0)
def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('classificationMap'):
        os.makedirs('classificationMap')
_init_()

# load ck domain data set
with open(os.path.join('datasets',  'Chikusei_imdb_128.pickle'), 'rb') as handle:
    source_imdb = pickle.load(handle)
print(source_imdb.keys())
print(source_imdb['Labels'])


# process ck domain data set   
data_train = source_imdb['data']  
labels_train = source_imdb['Labels']  
print(data_train.shape)
print(labels_train.shape)
keys_all_train = sorted(list(set(labels_train)))   
print(keys_all_train)  
label_encoder_train = {}  
for i in range(len(keys_all_train)):
    label_encoder_train[keys_all_train[i]] = i
print(label_encoder_train)

train_set = {}
for class_, path in zip(labels_train, data_train):
    if label_encoder_train[class_] not in train_set:
        train_set[label_encoder_train[class_]] = []
    train_set[label_encoder_train[class_]].append(path)
print(train_set.keys())
data = train_set
del train_set
del keys_all_train
del label_encoder_train

print("Num classes for source domain datasets: " + str(len(data)))
logging.info("Num classes for source domain datasets: {:f}".format(len(data)))

logging.info(data.keys())
data = utils.sanity_check(data) # 200 labels samples per class

print("Num classes of the number of class larger than 200: " + str(len(data)))
logging.info("Num classes of the number of class larger than 200:  {:f}".format(len(data)))

for class_ in data:
    for i in range(len(data[class_])):
        image_transpose = np.transpose(data[class_][i], (2, 0, 1))  
        data[class_][i] = image_transpose

# source few-shot classification data
metatrain_data = data
print(len(metatrain_data.keys()), metatrain_data.keys())
del data


test_data = 'datasets/IP/Indian_pines_corrected.mat'
test_label = 'datasets/IP/Indian_pines_gt.mat'


Data_Band_Scaler, GroundTruth = utils.load_data(test_data, test_label,dataname)

# get train_loader and test_loader
def get_train_test_loader(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    print(Data_Band_Scaler.shape) # (610, 340, 103)
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape

    '''label start'''
    num_class = int(np.max(GroundTruth))  #9
    data_band_scaler = utils.flip(Data_Band_Scaler)
    groundtruth = utils.flip(GroundTruth)
    del Data_Band_Scaler
    del GroundTruth

    HalfWidth = 4
    G = groundtruth[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth]    
    data = data_band_scaler[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth,:]   
    [Row, Column] = np.nonzero(G)   
    logging.info('number of sample{:f}'.format(len(Row)))
    del data_band_scaler
    del groundtruth

    nSample = np.size(Row)
    logging.info('number of sample'.format(nSample)) 

    # Sampling samples
    train = {}
    test = {}
    da_train = {}  
    m = int(np.max(G))   
    logging.info('class:{:f}'.format(m))   

    nlabeled =TEST_LSAMPLE_NUM_PER_CLASS
    logging.info('labeled number per class for test dataset:{:f}'.format(nlabeled))   
    logging.info('number of sample:{:f}'.format((200 - nlabeled) / nlabeled + 1))   
    logging.info('number of sample math.ceil:{:f}'.format(math.ceil((200 - nlabeled) / nlabeled) + 1))   

    for i in range(m):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if G[Row[j], Column[j]] == i + 1]
        np.random.shuffle(indices)
        nb_val = shot_num_per_class    
        train[i] = indices[:nb_val]
        da_train[i] = []
        for j in range(math.ceil((200 - nlabeled) / nlabeled) + 1):
            da_train[i] += indices[:nb_val]
        test[i] = indices[nb_val:]

    train_indices = []
    test_indices = []
    da_train_indices = []
    for i in range(m):   
        train_indices += train[i]
        test_indices += test[i]
        da_train_indices += da_train[i]
    np.random.shuffle(test_indices)

    logging.info('the number of train_indices{:}:'.format(len(train_indices)))  
    logging.info('the number of test_indices:{:}'.format(len(test_indices))) 
    logging.info('the number of train_indices after data argumentation:{:}'.format(len(da_train_indices)))   
    logging.info('labeled sample indices:{:}'.format(train_indices))  

    nTrain = len(train_indices)   
    nTest = len(test_indices)   
    da_nTrain = len(da_train_indices)   

    imdb = {}
    imdb['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, nTrain + nTest], dtype=np.float32)   
    imdb['Labels'] = np.zeros([nTrain + nTest], dtype=np.int64)  
    imdb['set'] = np.zeros([nTrain + nTest], dtype=np.int64)

    RandPerm = train_indices + test_indices

    RandPerm = np.array(RandPerm)

    for iSample in range(nTrain + nTest):
        imdb['data'][:, :, :, iSample] = data[Row[RandPerm[iSample]] - HalfWidth:  Row[RandPerm[iSample]] + HalfWidth + 1,
                                        Column[RandPerm[iSample]] - HalfWidth: Column[RandPerm[iSample]] + HalfWidth + 1, :]
        imdb['Labels'][iSample] = G[Row[RandPerm[iSample]], Column[RandPerm[iSample]]].astype(np.int64)

    imdb['Labels'] = imdb['Labels'] - 1  # 1-16 0-15
    imdb['set'] = np.hstack((np.ones([nTrain]), 3 * np.ones([nTest]))).astype(np.int64)
    logging.info('Data is OK.')

    train_dataset = utils.matcifar(imdb, train=True, d=3, medicinal=0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=class_num * shot_num_per_class,shuffle=False, num_workers=0)
    del train_dataset

    test_dataset = utils.matcifar(imdb, train=False, d=3, medicinal=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)
    del test_dataset
    del imdb

    # Data Augmentation for target domain for training
    imdb_da_train = {}
    imdb_da_train['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, da_nTrain],  dtype=np.float32)   
    imdb_da_train['Labels'] = np.zeros([da_nTrain], dtype=np.int64)    
    imdb_da_train['set'] = np.zeros([da_nTrain], dtype=np.int64)

    da_RandPerm = np.array(da_train_indices)
    for iSample in range(da_nTrain):  # radiation_noise，flip_augmentation
        imdb_da_train['data'][:, :, :, iSample] = utils.radiation_noise(
            data[Row[da_RandPerm[iSample]] - HalfWidth:  Row[da_RandPerm[iSample]] + HalfWidth + 1,
            Column[da_RandPerm[iSample]] - HalfWidth: Column[da_RandPerm[iSample]] + HalfWidth + 1, :])
        imdb_da_train['Labels'][iSample] = G[Row[da_RandPerm[iSample]], Column[da_RandPerm[iSample]]].astype(np.int64)

    imdb_da_train['Labels'] = imdb_da_train['Labels'] - 1  
    imdb_da_train['set'] = np.ones([da_nTrain]).astype(np.int64)
    logging.info('ok')

    return train_loader, test_loader, imdb_da_train ,G,RandPerm,Row, Column,nTrain


def get_target_dataset(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    train_loader, test_loader, imdb_da_train,G,RandPerm,Row, Column,nTrain = get_train_test_loader(Data_Band_Scaler=Data_Band_Scaler,  GroundTruth=GroundTruth, \
                                                                    class_num=class_num,shot_num_per_class=shot_num_per_class)  # 9 classes and 5 labeled samples per class
    train_datas, train_labels = train_loader.__iter__().next()
    logging.info('train labels:{:}'.format(train_labels))  #45=9*5
    logging.info('size of train datas:{}{}{}{}'.format(train_datas.shape[0],train_datas.shape[1],train_datas.shape[2],train_datas.shape[3])) # size of train datas: torch.Size([45, 103, 9, 9])
    del Data_Band_Scaler, GroundTruth
    # target data with data augmentation
    target_da_datas = np.transpose(imdb_da_train['data'], (3, 2, 0, 1))  
    target_da_labels = imdb_da_train['Labels'] 
    logging.info('target data augmentation label:{}'.format(target_da_labels))
    # metatrain data for few-shot classification
    target_da_train_set = {}
    for class_, path in zip(target_da_labels, target_da_datas):
        if class_ not in target_da_train_set:
            target_da_train_set[class_] = []
        target_da_train_set[class_].append(path)
    target_da_metatrain_data = target_da_train_set
    # target domain : batch samples for domian adaptation
    target_dataset = utils.matcifar(imdb_da_train, train=True, d=3, medicinal=0)
    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=128, shuffle=True, num_workers=0)
    del target_dataset

    return train_loader, test_loader, target_da_metatrain_data, target_loader,G,RandPerm,Row, Column,nTrain




class Mapping(nn.Module):
    def __init__(self, in_dimension, out_dimension):
        super(Mapping, self).__init__()
        self.preconv = nn.Conv2d(in_dimension, out_dimension, 1, 1, bias=False)
        self.preconv_bn = nn.BatchNorm2d(out_dimension)

    def forward(self, x):
        x = self.preconv(x)
        x = self.preconv_bn(x)
        return x

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # self.feature_encoder = darts_encoder(100,16,9,2,arg.auxiliary,)#D_Res_3d_CNN(1,8,16)
        self.feature_encoder = darts_encoder(args.n_dim, args.init_channels, args.class_num, args.layers,args.auxiliary, genotype)
        self.feature_encoder.drop_path_prob = args.drop_path_prob
        self.final_feat_dim = FEATURE_DIM  # 64+32
        #         self.bn = nn.BatchNorm1d(self.final_feat_dim)
        self.classifier = nn.Linear(in_features=self.final_feat_dim, out_features=CLASS_NUM)
        self.target_mapping = Mapping(TAR_INPUT_DIMENSION, N_DIMENSION)
        self.source_mapping = Mapping(SRC_INPUT_DIMENSION, N_DIMENSION)
        self.mini_mapping = Mapping(MINI_INPUT_DIMENSION, N_DIMENSION)
        # self.classifier = nn.Linear(256,9)


    def forward(self, x, domain='target'):  # x
        if domain == 'target':
            x = self.target_mapping(x)  # (45, 100,9,9)  把光谱维度统一
        elif domain == 'source':
            x = self.source_mapping(x)  # (45, 100,9,9)
        elif domain == 'mini':
            x = self.mini_mapping(x)  # (45, 100,9,9)

        feature = self.feature_encoder(x)  # (45, 64)

        return feature


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

crossEntropy = nn.CrossEntropyLoss().cuda() 
adaploss = AdaSPLoss(temp = args.tempsum, loss_type = 'adasp')#.cuda()  tempsum

def euclidean_metric(a, b):  
    n = a.shape[0]  
    m = b.shape[0] 
    a = a.unsqueeze(1).expand(n, m, -1) 
    b = b.unsqueeze(0).expand(n, m, -1) 
    logits = -((a - b)**2).sum(dim=2) 
    return logits

# run 10 times
nDataSet = 1
acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, CLASS_NUM])
k = np.zeros([nDataSet, 1])
best_predict_all = []
best_acc_all = 0.0
best_G,best_RandPerm,best_Row, best_Column,best_nTrain = None,None,None,None,None

for iDataSet in range(nDataSet):
    np.random.seed(seeds)
    logging.info('seed---------------- {:f}'.format(seeds))
    train_loader, test_loader, target_da_metatrain_data, target_loader,G,RandPerm,Row, Column,nTrain = get_target_dataset(
        Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth,class_num=TEST_CLASS_NUM, shot_num_per_class=TEST_LSAMPLE_NUM_PER_CLASS)
    metatrain_folders, metatest_folders = tg.mini_imagenet_folders()
    # model
    feature_encoder = Network()    
    feature_encoder.apply(weights_init)   
    feature_encoder.cuda()
    
    device = torch.device("cuda")
    input = torch.randn(1,200, 9, 9).to(device)
    flops, params = profile(feature_encoder, inputs=(input,))
    print('flops',flops)			## 打印计算量
    print('params',params)	
    logging.info('flops {:f}'.format(flops))			## 打印计算量
    logging.info('params {:f}'.format(params))	

    feature_encoder.train()
    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=args.learning_rate) #,weight_decay=args.weight_decay
    logging.info("Training...")
    last_accuracy = 0.0
    best_episdoe = 0
    train_loss = []
    test_acc = []
    running_D_loss, running_F_loss = 0.0, 0.0
    running_label_loss = 0
    running_domain_loss = 0
    total_hit, total_num = 0.0, 0.0
    target_total_hit, target_total_num = 0.0, 0.0

    test_acc_list = []
    minitotal_hit, minitotal_num = 0.0, 0.0
    min_acc=0.0
    max_loss=100
    source_time=0.0
    target_time=0.0

    '''source 断点保存，以及训练过程的监督'''
    scheduler = torch.optim.lr_scheduler.StepLR(feature_encoder_optim, step_size=args.episode // 5, gamma=0.5)
    checkpointer = Checkpointer(
            feature_encoder, feature_encoder_optim, scheduler, 'checkpoints/model', save_to_disk=True)
    writer = SummaryWriter('checkpoints/model/log', comment='pu')
    arguments = {}
    arguments["epoch"] = 0
    extra_checkpoint_data = checkpointer.load("")
    arguments.update(extra_checkpoint_data)
    checkpointer_period =1000
    max_epoch = args.episode 
    start_epoch = arguments["epoch"]

    for episode in range(start_epoch, max_epoch):  
        arguments["epoch"]=episode
        scheduler.step()
        if episode % 2 == 0:
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
            minisupports, minisupport_labels = minisupport_dataloader.__iter__().next()   
            miniquerys, miniquery_labels = miniquery_dataloader.__iter__().next()
            # calculate features
            train_start = time.time()
            support_features = feature_encoder(supports.cuda(), domain='source')  
            query_features = feature_encoder(querys.cuda(), domain='source')  
            minisupport_features = feature_encoder(minisupports.cuda(),  domain='mini')  
            miniquery_features = feature_encoder(miniquerys.cuda(), domain='mini')
            train_end=time.time()
            source_time += (train_end-train_start)

            # Prototype network
            if SHOT_NUM_PER_CLASS > 1:
                support_proto = support_features.reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)   
                minisupport_proto = minisupport_features.reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)   

            else:
                support_proto = support_features
                minisupport_proto = minisupport_features

            logits = euclidean_metric(query_features, support_proto)   
            minilogits = euclidean_metric(miniquery_features, minisupport_proto)   
            f_loss1 = adaploss(logits, query_labels.cuda())  
            f_loss2 = crossEntropy(logits, query_labels.cuda())  
            f_loss = f_loss1+f_loss2
            minif_loss1 = adaploss(minilogits, miniquery_labels.cuda())
            minif_loss2 = crossEntropy(minilogits, miniquery_labels.cuda())
            minif_loss = minif_loss1+minif_loss2
            loss = f_loss + minif_loss  

            # Update parameters
            feature_encoder.zero_grad()
            loss.backward()
            feature_encoder_optim.step()

            total_hit += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels).item()
            minitotal_hit += torch.sum(torch.argmax(minilogits, dim=1).cpu() == miniquery_labels).item()

            total_num += querys.shape[0]
            minitotal_num += miniquerys.shape[0]
            mini_acc= minitotal_hit / minitotal_num
            ck_acc=total_hit / total_num
            sum_acc=mini_acc+ck_acc
            logging.info('source episode {:>3d}:  sp ck: {:6.4f}, sp mini: {:6.4f} '.format(episode + 1, \
                                                            f_loss1.item(),minif_loss1.item() ))
            logging.info('source episode {:>3d}:  celoss ck: {:6.4f}, celoss mini: {:6.4f} '.format(episode + 1, \
                                                                f_loss2.item(),minif_loss2.item()))
            logging.info('source episode {:>3d}:  ckfsl loss: {:6.4f}, minifsl loss: {:6.4f}, ck acc {:6.4f}, mini acc {:6.4f},loss: {:6.4f}'.format(episode + 1, \
                                                                f_loss.item(),minif_loss.item(), ck_acc, mini_acc,loss.item()))              
            
            if episode % checkpointer_period == 0:
                    checkpointer.save("source_model_{:03d}".format(episode), **arguments)
            if episode == max_epoch:
                checkpointer.save("source_model_final", **arguments)
        # target domain few-shot + domain adaptation
        else:
            '''Few-shot classification for target domain data set'''
            task = utils.Task(target_da_metatrain_data, TEST_CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)   
            support_dataloader = utils.get_HBKC_data_loader(task, num_per_class=SHOT_NUM_PER_CLASS, split="train", shuffle=False)
            query_dataloader = utils.get_HBKC_data_loader(task, num_per_class=QUERY_NUM_PER_CLASS, split="test", shuffle=True)

            # sample datas
            supports, support_labels = support_dataloader.__iter__().next()   
            querys, query_labels = query_dataloader.__iter__().next()   
        

            # calculate features
            target_train_start = time.time()
            support_features = feature_encoder(supports.cuda(),  domain='target')   
            query_features = feature_encoder(querys.cuda(), domain='target')   
            target_train_end = time.time()
            target_time +=(target_train_end-target_train_start)
        
            # Prototype network
            if SHOT_NUM_PER_CLASS > 1:
                support_proto = support_features.reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)   

            else:
                support_proto = support_features

            logits = euclidean_metric(query_features, support_proto)   
            print(logits.max())
            print(logits.min())
            f_loss1 = adaploss(logits, query_labels.cuda())  
            f_loss2 = crossEntropy(logits, query_labels.cuda())  
            f_loss = f_loss1+f_loss2  

            loss = f_loss  
            # Update parameters
            feature_encoder.zero_grad()
            loss.backward()
            feature_encoder_optim.step()

            target_total_hit += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels).item()
            target_total_num += querys.shape[0]

            # if (episode + 1) % 10 == 0:  # display
            train_loss.append(loss.item())
            logging.info('target episode {:>3d}:  fsl loss: {:6.4f}, acc {:6.4f}'.format(episode + 1, loss.item(), target_total_hit / target_total_num))
            writer.add_scalars('train_acc', { 'target_train': target_total_hit / target_total_num}, episode)
            writer.add_scalars('train_loss', {'target_loss': loss}, episode)
            writer.add_scalars('fusion_loss', {'adap_loss': f_loss1}, episode)
            writer.add_scalars('fusion_loss', {'ce_loss': f_loss2}, episode)


                

        if (episode + 1) % 1000 == 0 or episode == 0:

            logging.info("Testing ...")
            feature_encoder.eval()
            total_rewards = 0
            counter = 0
            test_time_sum=0.0
            knn_time_sum=0.0
            knn_test_time_sum=0.0
            accuracies = []
            predict = np.array([], dtype=np.int64)
            labels = np.array([], dtype=np.int64)


            train_datas, train_labels = train_loader.__iter__().next()
            logging.info(train_datas.shape)
            train_features = feature_encoder(Variable(train_datas).cuda(), domain='target')  # (45, 160)
            logging.info(train_features.shape)


            max_value = train_features.max()  # 89.67885
            min_value = train_features.min()  # -57.92479
            logging.info(max_value.item())
            logging.info(min_value.item())
            train_features = (train_features - min_value) * 1.0 / (max_value - min_value) #归一化

            KNN_classifier = KNeighborsClassifier(n_neighbors=1)  #K近邻算法将提取到的特征分类--特征度量空间
            knn_train_star=time.time()
            KNN_classifier.fit(train_features.cpu().detach().numpy(), train_labels)  # .cpu().detach().numpy()
            knn_train_end=time.time()
            knn_time_sum=knn_train_end-knn_train_star
            for test_datas, test_labels in test_loader:
                batch_size = test_labels.shape[0]

                test_train_star = time.time()
                test_features = feature_encoder(Variable(test_datas).cuda(), domain='target')  # (100, 160)
                test_train_end = time.time()
                test_time_sum +=(test_train_end-test_train_star)

                test_features = (test_features - min_value) * 1.0 / (max_value - min_value)  #归一化
                knn_test_star=time.time()
                predict_labels = KNN_classifier.predict(test_features.cpu().detach().numpy())  #KNN根据训练数据形成的特征空间，对测试数据分类
                knn_test_end=time.time()
                knn_test_time_sum +=(knn_test_end-knn_test_star)
                test_labels = test_labels.numpy()
                rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(batch_size)]

                total_rewards += np.sum(rewards)
                counter += batch_size

                predict = np.append(predict, predict_labels)
                labels = np.append(labels, test_labels)

                accuracy = total_rewards / 1.0 / counter  #
                accuracies.append(accuracy)

            test_accuracy = 100. * total_rewards / len(test_loader.dataset)

            logging.info('\t\tAccuracy: {}/{} ({:.2f}%)\n'.format( total_rewards, len(test_loader.dataset),
                100. * total_rewards / len(test_loader.dataset)))
            test_end = time.time()

            # Training mode
            feature_encoder.train()
            if test_accuracy > last_accuracy:
                # save networks
                torch.save(feature_encoder.state_dict(),str( "checkpoints/IP_best_encoder_" + "IP_" +str(iDataSet) +"iter_" + str(episode) +"shot.pth"))
                logging.info("save networks for episode:",episode+1)
                last_accuracy = test_accuracy
                best_episdoe = episode

                acc[iDataSet] = 100. * total_rewards / len(test_loader.dataset)
                OA = acc
                C = metrics.confusion_matrix(labels, predict)
                A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=np.float)

                k[iDataSet] = metrics.cohen_kappa_score(labels, predict)

            logging.info('best episode:[{}], best accuracy={}'.format(best_episdoe + 1, last_accuracy))

    if test_accuracy > best_acc_all:
        best_predict_all = predict
        best_G,best_RandPerm,best_Row, best_Column,best_nTrain = G, RandPerm, Row, Column, nTrain
    logging.info('iter:{} best episode:[{}], best accuracy={}'.format(iDataSet, best_episdoe + 1, last_accuracy))
    logging.info('***********************************************************************************')

AA = np.mean(A, 1)

AAMean = np.mean(AA,0)
AAStd = np.std(AA)

AMean = np.mean(A, 0)
AStd = np.std(A, 0)

OAMean = np.mean(acc)
OAStd = np.std(acc)

kMean = np.mean(k)
kStd = np.std(k)
file_name = "map/classification_map.txt"
# print(each_acc)
with open(file_name, 'w') as x_file:
    x_file.write(' Overall accuracy (%):{}'.format(OAMean))
    x_file.write('\n')
    x_file.write(' Average accuracy (%):{}'.format(100 * AAMean))
    x_file.write('\n')
    x_file.write(' Kappa accuracy (%):{}'.format(100 *kMean))
    x_file.write('\n')
    for i in range(len(AMean)):
            x_file.write('class {} acc: {}\n'.format(i+1, 100 * AMean[i]))
logging.info ("train time source DataSet(s): " + "{:.5f}".format(source_time))
logging.info ("train time target DataSet(s): " + "{:.5f}".format(target_time))
logging.info("knn train time target DataSet(s): " + "{:.5f}".format(knn_time_sum)) 
logging.info("knn test time target DataSet(s): " + "{:.5f}".format(knn_test_time_sum)) 
logging.info("test time target DataSet(s): " + "{:.5f}".format(test_time_sum)) 
logging.info ("average OA: " + "{:.2f}".format( OAMean) + " +- " + "{:.2f}".format( OAStd))
logging.info ("average AA: " + "{:.2f}".format(100 * AAMean) + " +- " + "{:.2f}".format(100 * AAStd))
logging.info ("average kappa: " + "{:.4f}".format(100 *kMean) + " +- " + "{:.4f}".format(100 *kStd))
logging.info ("accuracy for each class: ")
for i in range(CLASS_NUM):
    logging.info ("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i]))


best_iDataset = 0
for i in range(len(acc)):
    logging.info('{}:{}'.format(i, acc[i]))
    if acc[i] > acc[best_iDataset]:
        best_iDataset = i
logging.info('best acc all={}'.format(acc[best_iDataset]))
logging.getLogger().removeHandler(fh)

# logging.removeHandler(fh)

#################classification map################################

for i in range(len(best_predict_all)):  # predict ndarray <class 'tuple'>: (9729,)
    best_G[best_Row[best_RandPerm[best_nTrain + i]]][best_Column[best_RandPerm[best_nTrain + i]]] = best_predict_all[i] + 1

hsi_pic = np.zeros((best_G.shape[0], best_G.shape[1], 3))
for i in range(best_G.shape[0]):
    for j in range(best_G.shape[1]):
        if best_G[i][j] == 0:
            hsi_pic[i, j, :] = [0, 0, 0]
        if best_G[i][j] == 1:
            hsi_pic[i, j, :] = [0, 0, 1]
        if best_G[i][j] == 2:
            hsi_pic[i, j, :] = [0, 1, 0]
        if best_G[i][j] == 3:
            hsi_pic[i, j, :] = [0, 1, 1]
        if best_G[i][j] == 4:
            hsi_pic[i, j, :] = [1, 0, 0]
        if best_G[i][j] == 5:
            hsi_pic[i, j, :] = [1, 0, 1]
        if best_G[i][j] == 6:
            hsi_pic[i, j, :] = [1, 1, 0]
        if best_G[i][j] == 7:
            hsi_pic[i, j, :] = [0.5, 0.5, 1]
        if best_G[i][j] == 8:
            hsi_pic[i, j, :] = [0.65, 0.35, 1]
        if best_G[i][j] == 9:
            hsi_pic[i, j, :] = [0.75, 0.5, 0.75]
        if best_G[i][j] == 10:
            hsi_pic[i, j, :] = [0.75, 1, 0.5]
        if best_G[i][j] == 11:
            hsi_pic[i, j, :] = [0.5, 1, 0.65]
        if best_G[i][j] == 12:
            hsi_pic[i, j, :] = [0.65, 0.65, 0]
        if best_G[i][j] == 13:
            hsi_pic[i, j, :] = [0.75, 1, 0.65]
        if best_G[i][j] == 14:
            hsi_pic[i, j, :] = [0, 0, 0.5]
        if best_G[i][j] == 15:
            hsi_pic[i, j, :] = [0, 1, 0.75]
        if best_G[i][j] == 16:
            hsi_pic[i, j, :] = [0.5, 0.75, 1]
utils.classification_map(hsi_pic[4:-4, 4:-4, :], best_G[4:-4, 4:-4], 24,  "classificationMap/IP_"+str(TEST_LSAMPLE_NUM_PER_CLASS)+"shot.png")

