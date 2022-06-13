from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json
import time
import copy
import linecache
import gzip
import pickle
import itertools
from sklearn.preprocessing import MinMaxScaler
import datetime
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import Dataset
from torch.nn import functional as F

import sys, argparse, os
import numpy as np
import random
import sys, argparse, os

import torch.nn.functional as F
import torch.utils.data as torchdata
from PIL import Image
import os
import random

import sys
sys.path.insert(0, '../')
sys.path.append("/scratch2/ml_flood/mlflood/")
from pathlib import Path
from conf import PATH_DATA
from conf import rain_const, waterdepth_diff_const
import h5py
import pandas as pd

import torch
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from utils import new_log
from models import CNNrolling
from models.utae import UTAE
from dataset_utae import dataloader_args_utae_test, load_test_dataset
from dataset_utae import MyCatchment
from evaluation import predict_event 

from training import *

path_exp = "/scratch2/ml_flood/data/checkpoints/709/cluster/utae_L1/experiment_0/"

str_args = ["--model=simple_net",
            "--save-dir=./save_dir/", ### !! this folder must already exist !!
            "--save-model=last",
            "--mode=test",
            "--workers=4",
#             "--normalize=10.651",  # Seeing the histogram of the data, I do not think this normalization make sense.
            "--data=multi",
            "--epochs=3",
            "--logstep-train=10",
            "--batch-size=1",
            "--optimizer=adam",
            "--lr=0.0001",
            "--loss=l1_2",
            "--fix_indexes=False"
           ]

args = parser.parse_args(str_args)

# choose torch device
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using: ', device)

def dataloader_args(t_dataset, catchment_num = "toy", batch_size = 8):
    
    dataloaders_test = torchdata.DataLoader(t_dataset, 
                                            batch_size=batch_size,
                                            shuffle=False,
                                            drop_last=True)
    
    return dataloaders_test

### Catchment settings
catchment_num = "709"
catchment_kwargs = {}
catchment_kwargs["tau"] = 0.5
catchment_kwargs["timestep"]= 5      # for timestep >1 use CNN rolling or Unet
catchment_kwargs["sample_type"]="single"
catchment_kwargs["dim_patch"]=256
catchment_kwargs["fix_indexes"]=True
catchment_kwargs["border_size"] = 0
catchment_kwargs["normalize_output"] = False
catchment_kwargs["use_diff_dem"] = False

dataloaders = {}
dataset = load_test_dataset(catchment_num=catchment_num, **catchment_kwargs)
dataloaders["test"] = dataloader_args_utae_test(dataset, catchment_num = catchment_num)
dataset_test = dataloaders["test"]


model = UTAE(args)

file_path1 = path_exp + "model.pth.tar"
model.load_state_dict(torch.load(file_path1))
model.cuda()

event_num = 0
start_ts = None
timestep = 5

pred_cnn, gt_cnn, mask_cnn = predict_event(model, dataset, event_num, start_ts, ar = False)

print("Done")


