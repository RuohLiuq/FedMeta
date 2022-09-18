#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # fedmeta arguments
    parser.add_argument('--epochs', type=int, default=200, help="rounds of training")
    parser.add_argument('--MAML_epochs', type=int, default=10, help="rounds of training")
    parser.add_argument('--FL_epochs', type=int, default=20, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=20, help="number of users: K")
    parser.add_argument('--num_tasks', type=int, default=4, help="number of tasks: K")
    parser.add_argument('--frac', type=float, default=0.5, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=32, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.0005, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    parser.add_argument('--num_spt_task', type=int, default=3, help="number of support tasks: K")
    parser.add_argument('--num_qur_task', type=int, default=3, help="number of query tasks: K")

    # model arguments
    parser.add_argument('--model', type=str, default='GRU', help='model name')
    parser.add_argument('--seq_len', type=int, default=120, help='sequence length')

    # other arguments
    parser.add_argument('--dataset', type=str, default='nilm', help="name of dataset")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')


    args = parser.parse_args()
    return args
