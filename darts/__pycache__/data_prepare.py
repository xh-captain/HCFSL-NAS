'''
read HSI dataset, split training, validation, and test dataset
'''
from __future__ import absolute_import, division
import torch
import h5py
import os
import utils
import logging
import argparse
from tensorflow.python.framework import dtypes
from tensorflow.contrib.learn.python.learn.datasets import base
import scipy.io as sio
import numpy as np

parser = argparse.ArgumentParser("HSI")

parser.add_argument('--data_root', type=str, default='data/h5')
args = parser.parse_args()
def load_data(image_file, label_file,windowsize):
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)
    image = image_data['hsi']   #610*340*103 salinas_corrected indian_pines_corrected

    image = image.astype(np.float32 )     #放一下，看归一化，放哪里合适--放扩充前，还是扩充后
    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    shape = np.shape(image)    #填充数据，方便取块
    imag1=np.zeros((shape[0]+windowsize,shape[1]+windowsize,shape[2]), dtype=np.float32)
    half=windowsize//2
    imag1[half:half+shape[0] ,half:half+shape[1] ,:]=image
    # print('填充后的数据立方体：',imag1.shape)

   
    label = label_data['houston_gt_sum']  # pavia  paviaU_gt  houston_gt_sum  indian_pines_gt
    # print('标签数据类型',type(label))
    print(label.dtype)
    shape1 = np.shape(label)    #填充数据，方便取块
    label1=np.zeros((shape1[0]+windowsize,shape1[1]+windowsize),dtype=np.uint8)
    half=windowsize//2
    label1[half:half+shape1[0] ,half:half+shape1[1]]=label
    # print('填充后的数据立方体：',imag1.shape)
    return imag1, label1


class DataSet(object):

    def __init__(self, images, labels, dtype=dtypes.float32, reshape=False):

        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint16, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %dtype)

        self._num_examples = images.shape[0]

        if reshape:
            images = images.reshape(images.shape[0], images.shape[1] * images.shape[2], images.shape[3])

        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""

        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]


def one_hot_transform(x, length):
    ont_hot_array = np.zeros([1, length])
    ont_hot_array[0, int(x)-1] = 1
    return ont_hot_array


def h5_dist_loader(data_dir):
    with h5py.File(data_dir, 'r') as f:
        train_map, val_map, test_map = f['train_label_map'][0], f['val_label_map'][0], f['test_label_map'][0]

    return  train_map, val_map, test_map

def get_dist_per(label_map, train_num, val_num):
    category_num = label_map.max()
    train_per_num = train_num
    val_per_num =val_num
    train_dist = np.zeros(category_num)
    val_dist = np.zeros(category_num)
    cate_i_dist = np.zeros(category_num)

    for i in range(1, category_num):
        cate_i_num = sum(sum(label_map==i))
        print('---------------第i类样本的数量',cate_i_num)
        train_dist[i-1]=int(min(np.floor(cate_i_num//2), train_per_num))
        val_dist[i-1]=int(min(np.floor(cate_i_num//2), val_per_num))
        cate_i_dist[i-1]=cate_i_num

    train_dist[-1] = train_num
    val_dist[-1] = val_num
    cate_i_dist[-1] = sum(sum(label_map==category_num))

    [m, n]=label_map.shape
    train_label_map = np.zeros((m,n))
    val_label_map = np.zeros((m,n))
    test_label_map = np.zeros((m,n))
    for i, [train_i_num, val_i_num, i_num] in enumerate(zip(train_dist, val_dist, cate_i_dist)):
        i_index = np.where(label_map==(i+1))
        shuffle_indices=np.random.permutation(int(i_num))
        train_indices=(i_index[0][shuffle_indices[:int(train_i_num)]], i_index[1][shuffle_indices[:int(train_i_num)]])
        val_indices=(i_index[0][shuffle_indices[int(train_i_num):int(train_i_num+val_i_num)]], i_index[1][shuffle_indices[int(train_i_num):int(train_i_num+val_i_num)]])
        test_indices=(i_index[0][shuffle_indices[int(train_i_num+val_i_num):]], i_index[1][shuffle_indices[int(train_i_num+val_i_num):]])

        train_label_map[train_indices]=i+1
        val_label_map[val_indices]=i+1
        test_label_map[test_indices]=i+1
    
    return train_label_map,val_label_map,test_label_map

def readdata(image_file, label_file, train_nsamples=200, validation_nsamples=100, windowsize=27,
             istraining=True, shuffle_number=None, batchnumber=10000, times=0, rand_seed=10):

    image, label = load_data(image_file, label_file,windowsize)
    print(image.shape,label.shape)
    shape = np.shape(image)
    halfsize = windowsize // 2
    number_class = np.max(label)
    
    not_zero_raw, not_zero_col = label.nonzero()    #非零位置的行列
    number_samples = len(not_zero_raw)
    print('总共非零的样本数',number_samples)
    dataset='houston'                 #-------indian----------------------------------------------------- 取样数据处
    
    # not_zero_raw, not_zero_col=change(not_zero_raw, not_zero_col,label)
    # height, width, category_num, train_map, val_map, test_map = 
    dist_dir = os.path.join(args.data_root, dataset+'_dist_per_'+str(windowsize)+'_train-{}_val-{}.h5'.
                            format(train_nsamples, validation_nsamples))
    train_label_map,val_label_map,test_label_map=h5_dist_loader(dist_dir)


    # print('行最大，列最大',max(not_zero_raw),max(not_zero_col))
    # print('行最小，列最小',min(not_zero_raw),min(not_zero_col))

    not_zero_raw1, not_zero_col1 = train_label_map.nonzero()    #非零位置的行列

    not_zero_raw2, not_zero_col2 = val_label_map.nonzero()    #非零位置的行列

    not_zero_raw, not_zero_col = test_label_map.nonzero()    #非零位置的行列

    t_samples=len(not_zero_col1)
    v_samples=len(not_zero_col2)
    test_nsamples = number_samples - t_samples - v_samples
    print(t_samples)
    print(v_samples)
    print('总共测试的样本数',test_nsamples)

    # if(test_nsamples==42776):
    # sio.savemat("map/not_zero_raw.mat", {"not_zero_raw":not_zero_raw})
    # sio.savemat("map/not_zero_col.mat", {"not_zero_col":not_zero_col})
         #取数据的位置记录
   
    index_r=np.zeros((1,1))
    index_c=np.zeros((1,1))#-----------------

    if istraining:
        np.random.seed(rand_seed)

        shuffle_number = np.arange(t_samples)
        np.random.shuffle(shuffle_number)

        train_image = np.zeros([t_samples, windowsize, windowsize, shape[2]], dtype=np.float32)
        validation_image = np.zeros([v_samples, windowsize, windowsize, shape[2]], dtype=np.float32)

        train_label = np.zeros([t_samples, number_class], dtype=np.uint8)
        validation_label = np.zeros([v_samples, number_class], dtype=np.uint8)

        for i in range(t_samples):
            train_image[i, :, :, :] = image[(not_zero_raw1[shuffle_number[i]] - halfsize):(not_zero_raw1[shuffle_number[i]] + halfsize+1 ),
                                            (not_zero_col1[shuffle_number[i]] - halfsize):(not_zero_col1[shuffle_number[i]] + halfsize+1 ), :]
            train_label[i, :] = one_hot_transform(label[not_zero_raw1[shuffle_number[i]],
                                                  not_zero_col1[shuffle_number[i]]], number_class)

        shuffle_number = np.arange(v_samples)
        np.random.shuffle(shuffle_number)
        for i in range(v_samples):
            validation_image[i, :, :, :] = image[(not_zero_raw2[shuffle_number[i]] - halfsize):(not_zero_raw2[shuffle_number[i]] + halfsize+1 ),
                                                 (not_zero_col2[shuffle_number[i]] - halfsize):(not_zero_col2[shuffle_number[i]] + halfsize+1 ), :]
            validation_label[i, :] = one_hot_transform(label[not_zero_raw2[shuffle_number[i]],
                                                       not_zero_col2[shuffle_number[i]]], number_class)

        train_image = np.transpose(train_image, axes=[0, 3, 1, 2])
        validation_image = np.transpose(validation_image, axes=[0, 3, 1, 2])
        train = DataSet(train_image, train_label)
        validation = DataSet(validation_image, validation_label)

        shuffle_number = np.arange(test_nsamples)
        np.random.shuffle(shuffle_number)
        return base.Datasets(train=train, validation=validation, test=None), shuffle_number

    else:
        # print('datapre---------------')
        print(test_nsamples)
        n_batch = test_nsamples // batchnumber

        if times > n_batch:

            return None

        if n_batch == times:

            batchnumber_test = test_nsamples - n_batch * batchnumber
            # if(test_nsamples==148152):   #----------------pavia-42776----------
            #     index_raw=np.zeros((batchnumber_test,1))
            #     index_col=np.zeros((batchnumber_test,1))  #-----------------------
            # index=0
            test_image = np.zeros([batchnumber_test, windowsize, windowsize, shape[2]], dtype=np.float32)
            test_label = np.zeros([batchnumber_test, number_class], dtype=np.uint8)

            for i in range(batchnumber_test):
                test_image[i, :, :, :] = image[(not_zero_raw[shuffle_number[batchnumber*times+i]] - halfsize):(not_zero_raw[shuffle_number[batchnumber*times+i]] + halfsize+1 ),
                                               (not_zero_col[shuffle_number[batchnumber*times+i]] - halfsize):(not_zero_col[shuffle_number[batchnumber*times+i]] + halfsize+1 ), :]
                test_label[i, :] = one_hot_transform(label[not_zero_raw[shuffle_number[batchnumber*times+i]],
                                                     not_zero_col[shuffle_number[batchnumber*times+i]]], number_class)
                # #---------------
                # if(test_nsamples==148152):
                #     index_raw[i]=not_zero_raw[shuffle_number[batchnumber*times+i]] 
                #     #行-列
                #     index_col[i]=not_zero_col[shuffle_number[batchnumber*times+i]]
            
            test_image = np.transpose(test_image, axes=[0, 3, 1, 2])
            test = DataSet(test_image, test_label)
            
            # if(test_nsamples==148152):
            #     index_r=np.append(index_r,index_raw,axis=0)
            #     index_c=np.append(index_c,index_col,axis=0)

            #     sio.savemat("index/paviaU_map/"+str(times)+"index_r.mat", {"index_r":index_r})
            #     sio.savemat("index/paviaU_map/"+str(times)+"index_c.mat", {"index_c":index_c})

            return base.Datasets(train=None, validation=None, test=test)

        if times < n_batch:

            test_image = np.zeros([batchnumber, windowsize, windowsize, shape[2]], dtype=np.float32)
            test_label = np.zeros([batchnumber, number_class], dtype=np.uint8)
            # if(test_nsamples==148152):   #--------------------------
            #     index_raw=np.zeros((batchnumber,1))
            #     index_col=np.zeros((batchnumber,1))  

            for i in range(batchnumber):
                test_image[i, :, :, :] = image[(not_zero_raw[shuffle_number[batchnumber*times+i]] - halfsize):(not_zero_raw[shuffle_number[batchnumber*times+i]] + halfsize+1 ),
                                               (not_zero_col[shuffle_number[batchnumber*times+i]] - halfsize):(not_zero_col[shuffle_number[batchnumber*times+i]] + halfsize+1 ), :]
                test_label[i, :] = one_hot_transform(label[not_zero_raw[shuffle_number[batchnumber*times+i]],
                                                     not_zero_col[shuffle_number[batchnumber*times+i]]], number_class)

                # if(test_nsamples==148152):
                #     #- #---------------
                #     index_raw[i]=not_zero_raw[shuffle_number[batchnumber*times+i]] 
                #     #行-列
                #     index_col[i]=not_zero_col[shuffle_number[batchnumber*times+i]]
               
            test_image = np.transpose(test_image, axes=[0, 3, 1, 2])
            test = DataSet(test_image, test_label)
            # if(test_nsamples==148152):    #15029-Houston
            #     index_r=np.append(index_r,index_raw,axis=0)#--------------------------------------
            #     index_c=np.append(index_c,index_col,axis=0)
            #     sio.savemat("index/paviaU_map/"+str(times)+"index_r.mat", {"index_r":index_r})
            #     sio.savemat("index/paviaU_map/"+str(times)+"index_c.mat", {"index_c":index_c})#----------------------------
            return base.Datasets(train=None, validation=None, test=test)



