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

from ds import prepare_foodredmeat_dataloaders, prepare_foodmeat_dataloaders
#load_mnist_data_loader, prepare_flickr_dataloaders, prepare_cub_dataloaders, prepare_flo_dataloaders

from utils import *
from networks import *
from train_probVLM_FOOD import *

import matplotlib.pyplot as plt

import pickle
import os



def load_data_loaders(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None

# Usage
dataset = 'food-101'  # coco or flickr
data_dir = ospj('../ProbVLM/Datasets/', dataset)
dataloader_config = mch({
    'batch_size': 64,
    'random_erasing_prob': 0.,
    'traindata_shuffle': True
})
#filename = '../ProbVLM/Datasets/coco/data_loaders_coco_person_extra_26.11.pkl'
#loaders = load_data_loaders(filename)


food_train_loader, food_valid_loader, food_test_loader = load_data_loader(dataset=dataset, data_dir='../GALS/data', dataloader_config=None)


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
    food_train_loader,
    food_valid_loader,
    Cri = TempCombLoss(),
    device='cuda',
    dtype=torch.cuda.FloatTensor,
    init_lr=8e-5,
    num_epochs=200,
    eval_every=5,
    ckpt_path='../ckpt/ProbVLM_FOOD_Meat_17.02',
    T1=1e0,
    T2=1e-4
)

