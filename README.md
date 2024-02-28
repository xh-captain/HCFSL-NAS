# HCFSL-NAS
Few-shot learning (FSL) has achieved promising performance in hyperspectral image classification (HSIC) with few labeled samples by designing a proper embedding feature extractor. However, the performance of embedding feature extractors relies on the design of efficient deep convolutional neural network architectures, which heavily depends on the expertise knowledge. Particularly, FSL requires extracting discriminative features effectively across different domains, which makes the construction even more challenging. In this paper, we propose a novel neural architecture search-based FSL model for HSI classification, called HCFSL-NAS. Three novel strategies are proposed in this work. First, a neural architecture search-based embedding feature extractor is developed to the FSL in HSIC, whose search space includes a group of proposed multi-scale convolutions with channel attention. Second, a multi-source learning framework is employed to aggregate abundant heterogeneous and homogeneous source data, which enables the powerful generalization of network to the HSIC with only few labeled samples. Finally, the pointwise-based cross-entropy loss and the pairwise-based adaptive sparse loss are jointly optimized to maximize inter-class distance and minimize the distance within a class simultaneously. Experimental results on four publicly hyperspectral data sets demonstrate that HCFSL-NAS outperforms both the exiting FSL methods and supervised learning methods for HSI classification with only  few labeled samples.

## Requirements
CUDA = 10.2
Python =  3.6.12
Pytorch =  1.0.1 
sklearn = 0.0
numpy = 1.19.2
tensorflow = 1.13.1  
## dataset
Take HCFSL-NAS method on the IP dataset as an example: 
You can also download the hyperspectral datasets from the following link.
Link: https://pan.baidu.com/s/1atfmqJJh134dxAeAF8KJaQ 
Extract code: 3vcy
An example dataset folder has the following structure:
```
datasets
├── IP
│   ├── indian_pines.mat
│   ├── indian_pines_corrected.mat
│   ├── indian_pines_gt.mat
└── Chikusei_imdb_128.pickle
└──miniImagenet


## Usage:
Take HFSL method on the PU dataset as an example: 
1. Download the required data set and move to folder`./datasets`.
2. To run the file: HSI_search.py
3. To run the file: HCFSL_NAS_IP.py. 

  
