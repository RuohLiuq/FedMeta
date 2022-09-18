#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import numpy as np
import xlrd                                                        #导入xlrd模块
import torch
from torch.utils.data import Dataset
import pandas as pd
from torch.utils.data import DataLoader
import time
import math
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os


class GRU_NILM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, seq_len,num_layers=2):
        super(GRU_NILM, self).__init__()
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.rnn = torch.nn.GRU(input_size = self.input_size,
                                hidden_size = self.hidden_size,
                                num_layers = self.num_layers)
        self.nn = torch.nn.Linear(in_features = self.batch_size * self.seq_len * 4,
                                  out_features= self.batch_size * 4)
        self.relu = torch.nn.LeakyReLU()

    def forward(self, input):
        hidden = torch.zeros(self.num_layers,
                             self.batch_size,
                             self.hidden_size)
        hidden = hidden.float().cuda()
        #print(hidden.shape,"hidden")
        out, hidden = self.rnn(input, hidden)
        out = out.view(self.seq_len * self.batch_size * 4)
        out = self.nn(out)
        out = self.relu(out)
        return out.view(-1, 4)

class GRU_NILM_2(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, seq_len,num_layers=2):
        super(GRU_NILM_2, self).__init__()
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.rnn = torch.nn.GRU(input_size = self.input_size,
                                hidden_size = self.hidden_size,
                                num_layers = self.num_layers)
        self.nn = torch.nn.Linear(in_features = self.batch_size * self.seq_len,
                                  out_features= self.batch_size)
        self.relu = torch.nn.LeakyReLU()

    def forward(self, input):
        hidden = torch.zeros(self.num_layers,
                             self.batch_size,
                             self.hidden_size)
        hidden = hidden.float().cuda()
        #print(hidden.shape,"hidden")
        out, hidden = self.rnn(input, hidden)
        out = out.view(self.seq_len * self.batch_size)
        out = self.nn(out)
        out = self.relu(out)
        return out.view(-1, 1)