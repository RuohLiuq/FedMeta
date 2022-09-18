import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch import nn, autograd
from models.Nets import GRU_NILM, GRU_NILM_2
from torch import optim
import copy
import os

class TASK(object):
    def __init__(self,args, idx, dataset):
        self.args = args
        self.data_task = DataLoader(
            dataset=dataset,
            batch_size=args.local_bs,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
        self.loss_func = nn.MSELoss()
        self.save_path = os.path.join('models', 'modelpre'+str(idx)+'.pt')

    def train(self, net):
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        for i in range(0,10):
            for step, (x, y) in enumerate(self.data_task):
                net.train()
                x = x.permute(1, 0, 2)
                x = x.cuda()
                y = y.cuda()
                output = net(x)
                loss = self.loss_func(output, y)
                net.zero_grad()
                loss.backward()
                optimizer.step()

        torch.save(net.state_dict(), self.save_path)


    def test(self,net):
        loss_list = []
        net = copy.deepcopy(net)
        net.load_state_dict(torch.load(self.save_path), strict=False)
        for i in range(0,10):
            for step, (x, y) in enumerate(self.data_task):
                net.train()
                x = x.permute(1, 0, 2)
                x = x.cuda()
                y = y.cuda()
                output = net(x)
                loss = self.loss_func(output, y)
                loss_list.append(loss)

        loss_avg = sum(loss_list) / len(loss_list)
        return loss_avg