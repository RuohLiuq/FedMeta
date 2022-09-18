#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import torch
import pandas as pd
import os

from utils.sampling import nilm_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import GRU_NILM
from models.Fed import FedAvg
from models.task import TASK

def load(path):

    # load csv
    file_name = path
    single_csv = pd.read_csv(file_name,
                             header=0,
                             names=['aggregate','AIR','CAR','DR','OV'],
                             usecols=[0,1,2,3,4],
                             na_filter=False,
                             parse_dates=True,
                             infer_datetime_format=True,
                             memory_map=True
                             )
    return single_csv

def data_preprocess(data, start,  N_data):
    x_ = data['aggregate'].values.tolist()
    # print(x_.shape)
    y_air = data['AIR'].values.tolist()
    y_car = data['CAR'].values.tolist()
    y_dryer = data['DR'].values.tolist()
    y_oven = data['OV'].values.tolist()
    # print(x_[0:100])
    x_seq = []
    gt = []
    for i in range(start, N_data):
        x_seq_ = x_[i:i + args.seq_len]
        x_seq.append(x_seq_)
        y_ = [y_air[i + 60], y_car[i + 60], y_dryer[i + 60], y_oven[i + 60]]
        gt.append(y_)
    x_seq = np.asarray(x_seq).astype(np.float32).reshape(-1, args.seq_len, 1)
    gt = np.asarray(gt).astype(np.float32).reshape(-1, 4)
    print("x", x_seq.shape)
    print("gt", gt[0].shape, gt[0])
    x = torch.from_numpy(x_seq)
    y = torch.from_numpy(gt)
    return x ,y
def create_dataset(path, idx):
    get_data_train = load(path)
    train_x, train_y = data_preprocess(get_data_train, idx*1200, (idx+1)*1200)
    data_load = TensorDataset(train_x, train_y)
    return data_load

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # create task class
    num_spt_task = args.num_spt_task
    num_qur_task = args.num_qur_task
    spt_task_list = []
    qur_task_list = []
    # load dataset and split users
    data_path_fed = "../dataset/pecanstreet/preprocess/feddata.csv"
    get_fed_data = load(data_path_fed)
    train_x, train_y = data_preprocess(get_fed_data,0, 48000)
    spt_dataset_list = []
    data_path_spt1 = "../dataset/pecanstreet/preprocess/task1.csv"
    data_path_spt2 = "../dataset/pecanstreet/preprocess/task2.csv"
    spt_task = []
    for idx in range(num_spt_task*2):
        data_1 = create_dataset(data_path_spt1,idx)
        task1 = TASK(args = args, idx=idx, dataset = data_1)
        spt_task.append(task1)
        data_2 = create_dataset(data_path_spt2,idx)
        spt_dataset_list.append(data_2)
        task2 = TASK(args = args, idx=idx*2, dataset = data_2)
        spt_task.append(task2)

    if args.dataset == 'nilm':
        print('dataset is nilm')
        dataset_train = TensorDataset(train_x, train_y)
        dict_users= nilm_iid(train_x,train_y,args.num_users)
        #test_dict_users = nilm_iid(test_x,test_y,args.MAML_epochs)
        print('here')
    else:
        exit('Error: unrecognized dataset')
    #data_size = dataset_train[0][0].shape

    # build model
    if args.model == 'GRU' and args.dataset == 'nilm':
        print('model is GRU')
        net_glob = GRU_NILM(input_size=1, hidden_size=4, batch_size=args.local_bs, seq_len = args.seq_len, num_layers=2).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()
    optimizer_glob = torch.optim.Adam(net_glob.parameters(), lr=0.0005)
    criterion = torch.nn.MSELoss()
    # copy weights
    print('here')
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    loss_1 = []
    loss_2 = []
    loss_3 = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    for iter_maml in range(args.MAML_epochs):
        loss_fl = []
        for iter in range(args.FL_epochs):
            loss_locals = []
            if not args.all_clients:
                w_locals = []
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            for idx in idxs_users:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                if args.all_clients:
                    w_locals[idx] = copy.deepcopy(w)
                else:
                    w_locals.append(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))
            # update global weights with Fed
            w_glob = FedAvg(w_locals)

            # copy weight to net_glob
            net_glob.load_state_dict(w_glob)

            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            #loss_fl.append(loss_test)
            print('Round {:3d}, Average loss {:.5f}'.format(iter, loss_avg))
            loss_train.append(loss_avg)

        # maml local update
        epoch_task_list = np.random.choice(range(4*num_spt_task), num_spt_task, replace=False)
        for idx in range(num_spt_task):
            local = spt_task[epoch_task_list[idx]]
            local.train(net=copy.deepcopy(net_glob).to(args.device))
            net_glob.load_state_dict(w_glob)

        for idx in range(num_spt_task):
            local = spt_task[epoch_task_list[idx]]
            loss = local.test(net=copy.deepcopy(net_glob).to(args.device))
            loss_fl.append(loss)
        # calculate loss and global update
        loss_test = sum(loss_fl)/len(loss_fl)
        net_glob.zero_grad()
        loss_test.backward()
        optimizer_glob.step()

        print('Round_MAML {:3d}, task loss {:.5f}'.format(iter_maml, loss_test))
        #print('Round_MAML {:3d}, task2 loss {:.5f}'.format(iter_maml, loss_m2))

    # save model
    model_save_path = os.path.join('F:/dataset/pecanstreet/preprocess/code/', 'modelfed.pt')
    torch.save(net_glob.state_dict(), model_save_path)

    # plot loss curve


