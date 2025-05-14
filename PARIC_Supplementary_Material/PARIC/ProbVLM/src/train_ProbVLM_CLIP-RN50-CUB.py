#!/usr/bin/env python
# coding: utf-8


import sys
sys.path.append('../LAVIS')
import lavis

import os

from os.path import join as ospj
from os.path import expanduser
from munch import Munch as mch
import numpy as np

from ds import prepare_coco_dataloaders, load_mnist_data_loader, prepare_flickr_dataloaders, prepare_cub_dataloaders, prepare_flo_dataloaders, prepare_cub_dataloaders_extra

from utils import *
from networks import *
from train_probVLM import *

import matplotlib.pyplot as plt
import pickle
import os



def load_data_loaders(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None


dataset = 'waterbird_1.0_forest2water2' # coco or flickr
data_dir = ospj('/../ProbVLM/Datasets/', dataset) # e.g. ospj(expanduser('~'), 'Documents', 'jm', 'data', dataset)
dataloader_config = mch({
    'batch_size': 64,
    'random_erasing_prob': 0.,
    'traindata_shuffle': True
})
filename = '../ProbVLM/Datasets/CUB/data_loaders_waterbirds_12.12.pkl'
loaders = load_data_loaders(filename)
cub_train_loader, cub_valid_loader, cub_test_loader = loaders['train'], loaders['val'], loaders['test']


# clip_net = load_model('cuda')
CLIP_Net = load_model_p(device='cuda', model_path=None)
ProbVLM_Net = BayesCap_for_CLIP_p(
    inp_dim=1024,
    out_dim=1024,
    hid_dim=512,
    num_layers=3,
    p_drop=0.05,
)


train_ProbVLM(
    CLIP_Net,
    ProbVLM_Net,
    cub_train_loader,
    cub_valid_loader,
    Cri = TempCombLoss(),
    device='cuda',
    dtype=torch.cuda.FloatTensor,
    init_lr=8e-5,
    num_epochs=200,
    eval_every=5,
    ckpt_path='../ckpt/ProbVLM_waterbirds_200epochs_12.12',
    T1=1e0,
    T2=1e-4
)

