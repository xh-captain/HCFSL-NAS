import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data
from torch import nn
  

class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt

  # _, t, p = utils.accuracy(logits, target, topk=(1,))
def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res,target,pred.squeeze()



class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for v in model.parameters())/1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)


def cutout(img, length, num_band):

    img = np.transpose(img, (2, 0, 1))

    c, h, w = np.shape(img)[0], np.shape(img)[1], np.shape(img)[2]

    data = img
    RandPerm = np.random.permutation(c)
    for i in range(len(RandPerm)//num_band):
        img_c = img[RandPerm[i], :, :]
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0

        img_c *= mask
        img_c = img_c[np.newaxis, :, :]
        data[RandPerm[i], :, :] = img_c

    img = np.transpose(data, (1, 2, 0))

    return img


 
class AdaSPLoss(object):
  """
  Adaptive sparse pairwise (AdaSP) loss
  """

  def __init__(self, temp=0.2, loss_type = 'adasp'):
    self.temp = temp
    self.loss_type = loss_type

  def __call__(self, feats, targets):
      
    feats_n = nn.functional.normalize(feats, dim=1)
    
    bs_size = feats_n.size(0)
    N_id = len(torch.unique(targets))
    N_ins = bs_size // N_id

    scale = 1./self.temp

    sim_qq = torch.matmul(feats_n, feats_n.T)
    sf_sim_qq = sim_qq*scale

    right_factor = torch.from_numpy(np.kron(np.eye(N_id),np.ones((N_ins,1)))).cuda()
    pos_mask = torch.from_numpy(np.kron(np.eye(N_id),np.ones((N_ins,1)))).cuda()
    left_factor = torch.from_numpy(np.kron(np.eye(N_id), np.ones((1,N_ins)))).cuda()
    
    ## hard-hard mining for pos
    mask_HH = torch.from_numpy(np.kron(np.eye(N_id),-1.*np.ones((N_ins,N_ins)))).cuda()
    mask_HH[mask_HH==0]=1.

    ID_sim_HH = torch.exp(sf_sim_qq.mul(mask_HH))
    ID_sim_HH = ID_sim_HH.mm(right_factor)
    ID_sim_HH = left_factor.mm(ID_sim_HH)

    pos_mask_id = torch.eye(N_id).cuda()
    pos_sim_HH = ID_sim_HH.mul(pos_mask_id)
    pos_sim_HH[pos_sim_HH==0]=1.
    pos_sim_HH = 1./pos_sim_HH
    ID_sim_HH = ID_sim_HH.mul(1-pos_mask_id) + pos_sim_HH.mul(pos_mask_id)
    
    ID_sim_HH_L1 = nn.functional.normalize(ID_sim_HH,p = 1, dim = 1)   
    
    ## hard-easy mining for pos
    mask_HE = torch.from_numpy(np.kron(np.eye(N_id),-1.*np.ones((N_ins,N_ins)))).cuda()
    mask_HE[mask_HE==0]=1.

    ID_sim_HE = torch.exp(sf_sim_qq.mul(mask_HE))
    ID_sim_HE = ID_sim_HE.mm(right_factor)

    pos_sim_HE = ID_sim_HE.mul(pos_mask)
    pos_sim_HE[pos_sim_HE==0]=1.
    pos_sim_HE = 1./pos_sim_HE
    ID_sim_HE = ID_sim_HE.mul(1-pos_mask) + pos_sim_HE.mul(pos_mask)

    # hard-hard for neg
    ID_sim_HE = left_factor.mm(ID_sim_HE)

    ID_sim_HE_L1 = nn.functional.normalize(ID_sim_HE,p = 1, dim = 1)
    

    l_sim = torch.log(torch.diag(ID_sim_HH))
    s_sim = torch.log(torch.diag(ID_sim_HE))

    weight_sim_HH = torch.log(torch.diag(ID_sim_HH)).detach()/scale
    weight_sim_HE = torch.log(torch.diag(ID_sim_HE)).detach()/scale
    wt_l = 2*weight_sim_HE.mul(weight_sim_HH)/(weight_sim_HH + weight_sim_HE)
    wt_l[weight_sim_HH < 0] = 0
    both_sim = l_sim.mul(wt_l) + s_sim.mul(1-wt_l) 

    adaptive_pos = torch.diag(torch.exp(both_sim))

    pos_mask_id = torch.eye(N_id).cuda()
    adaptive_sim_mat = adaptive_pos.mul(pos_mask_id) + ID_sim_HE.mul(1-pos_mask_id)

    adaptive_sim_mat_L1 = nn.functional.normalize(adaptive_sim_mat,p = 1, dim = 1)

    loss_sph = -1*torch.log(torch.diag(ID_sim_HH_L1)).mean()
    loss_splh = -1*torch.log(torch.diag(ID_sim_HE_L1)).mean()
    loss_adasp = -1*torch.log(torch.diag(adaptive_sim_mat_L1)).mean()
    
    if self.loss_type == 'sp-h':
        loss = loss_sph.mean()
    elif self.loss_type == 'sp-lh':
        loss = loss_splh.mean()
    elif self.loss_type == 'adasp':
        loss = loss_adasp
        
    return loss


