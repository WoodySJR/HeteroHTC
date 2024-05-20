#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch, pandas as pd, numpy as np, math
from torch import nn
from torch.utils.data import DataLoader
from Dataset import MyDataset
import warnings
warnings.filterwarnings("ignore")
from train import train
from HGT import HGT
from test import test


# In[2]:


label_tokens = torch.tensor(torch.load("/home/tiny-stupid-papa/Public/HGT_codes_and_data/other_baselines/data_BGC/bgc_label_tokens.pt"))
mask = torch.load("/home/tiny-stupid-papa/Public/HGT_codes_and_data/other_baselines/data_BGC/bgc_att_mask.pt")
label_tokens = label_tokens[0:130,:]
mask = mask[0:130,0:130]

# In[3]:


device = "cuda:0"
torch.cuda.manual_seed(2002)
torch.manual_seed(2002)

model = HGT(hidden_dim=128, num_heads=4, num_labels=130, num_1=7, num_2=46, num_3=77,
           att_mask=mask, label_tokens=label_tokens, num_hgt_layers=2)

model = model.to(device)


# In[4]:


train_data = MyDataset("/home/tiny-stupid-papa/Public/HGT_codes_and_data/other_baselines/data_BGC/bgc_train_x.pt", "/home/tiny-stupid-papa/Public/HGT_codes_and_data/other_baselines/data_BGC/bgc_train_y.pt", device=device)
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)

val_data = MyDataset("/home/tiny-stupid-papa/Public/HGT_codes_and_data/other_baselines/data_BGC/bgc_val_x.pt", "/home/tiny-stupid-papa/Public/HGT_codes_and_data/other_baselines/data_BGC/bgc_val_y.pt", device=device)
val_dataloader = DataLoader(val_data, batch_size=32, shuffle=False)

test_data = MyDataset("/home/tiny-stupid-papa/Public/HGT_codes_and_data/other_baselines/data_BGC/bgc_test_x.pt", "/home/tiny-stupid-papa/Public/HGT_codes_and_data/other_baselines/data_BGC/bgc_test_y.pt", device=device)
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False)


train(model, train_dataloader, val_dataloader, 200, 0.0001, 15, "/home/tiny-stupid-papa/Public/HGT_codes_and_data/other_baselines/result_BGC/HGT", 0.5)

test(test_dataloader, "/home/tiny-stupid-papa/Public/HGT_codes_and_data/other_baselines/result_BGC/HGT/",0.5)


# In[ ]:




