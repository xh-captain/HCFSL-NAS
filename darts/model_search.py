import torch
import torch.nn as nn
import torch.nn.functional as F
# from drop_operations import *
from .operations import *
from torch.autograd import Variable
from .genotypes import PRIMITIVES
from .genotypes import Genotype
from .utils import AdaSPLoss    


class MixedOp(nn.Module):

  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, True)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=True))
      self._ops.append(op)

  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    self.reduction = reduction

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=True)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=True)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=True)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride)
        self._ops.append(op)

  def forward(self, s0, s1, weights):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    offset_t = 0
    for i in range(self._steps):
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)


def euclidean_metric(a, b):   
  n = a.shape[0]   
  m = b.shape[0]   
  a = a.unsqueeze(1).expand(n, m, -1) 
  b = b.unsqueeze(0).expand(n, m, -1)  
  logits = -((a - b)**2).sum(dim=2)  
  return logits

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

  def __init__(self, nBand, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
    super(Network, self).__init__()
    self._nBand=nBand
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._adaploss = AdaSPLoss(loss_type = 'adasp')#.cuda()  
    self._steps = steps
    self._multiplier = multiplier

    C_curr = self._nBand#stem_multiplier*C
     
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    # self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas()
    self.target_mapping = Mapping(103, nBand)
    self.source_mapping = Mapping(128, nBand)
    self.mini_mapping = Mapping(3, nBand)
    # self.domain=domain

  def new(self):
    model_new = Network(self._nBand, self._C, self._num_classes, self._layers, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input,domain='mini'): #mini
    if domain == 'target':
        input = self.target_mapping(input)  # (45, 100,9,9)  把光谱维度统一
    elif domain == 'ck':
        input = self.source_mapping(input)  # (45, 100,9,9)
    elif domain == 'mini':
        input = self.mini_mapping(input)  # (45, 100,9,9)
    s0 = s1 = input
    # print('模型输入进行spectral pooling后的大小')
    # print(input.shape)
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        weights = F.softmax(self.alphas_reduce,dim=-1)
      else:
        weights = F.softmax(self.alphas_normal,dim=-1)
      s0, s1 = s1, cell(s0, s1, weights)
    out = self.global_pooling(s1)
    logits = out.view(out.size(0), -1)
    return logits

  def _loss(self, input, target,target_lable):
    input = self(input)
    target = self(target) #171*128
    distance = euclidean_metric(target,input)  #欧氏距离
    loss1 = self._criterion(distance, target_lable.cuda())  #171*9  171adaploss
    loss2 = self._adaploss(distance, target_lable.cuda())  #171*9  171 adaploss
    loss = loss1 + loss2
    loss=torch.unsqueeze(loss,0)
    return loss 

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)

    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):

    def _parse(weights):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene

    gene_normal = _parse(F.softmax(self.alphas_normal,dim=-1).data.cpu().numpy())
    gene_reduce = _parse(F.softmax(self.alphas_reduce,dim=-1).data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

