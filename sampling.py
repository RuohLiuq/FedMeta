#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
def nilm_iid(x_FL,y_FL,num_users):
    num_items_FL = int(len(x_FL)/num_users)
    print('num_items_FL', num_items_FL)
    dict_users_FL, all_idxs = {}, [i for i in range(len(x_FL))]
    for i in range(num_users):
        dict_users_FL[i] = set(np.random.choice(all_idxs, num_items_FL, replace=False))
        all_idxs = list(set(all_idxs) - dict_users_FL[i])
    return dict_users_FL



