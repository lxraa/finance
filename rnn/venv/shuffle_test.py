import sys
import torch
import random
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset


class DealDataset(Dataset):
    def __init__(self):
        xy = np.loadtxt(open('./iris.csv', 'rb'), delimiter=',', dtype=np.float32)
        # data = pd.read_csv("iris.csv",header=None)
        # xy = data.values
        self.x_data = torch.from_numpy(xy[:, 0:-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])
        self.len = xy.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dealDataset = DealDataset()

train_loader2 = DataLoader(dataset=dealDataset,
                           batch_size=3,
                           shuffle=False)
# print(dealDataset.x_data)
for i, data in enumerate(train_loader2):
    inputs, labels = data

    # inputs, labels = Variable(inputs), Variable(labels)
    a = torch.tensor([1,1,1,1,1])
    print(a.shape)
    b = torch.tensor(a.view(-1,a.shape[0]))
    print(b.shape)
    # print("epoch：", epoch, "的第" , i, "个inputs", inputs.data.size(), "labels", labels.data.size())