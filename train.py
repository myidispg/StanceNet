# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 18:33:28 2019

@author: myidispg
"""

import pickle
import os
import torch

import numpy as np

from models.full_model import OpenPoseModel

import utilities.constants as constants
import utilities.helper as helper

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Read the pickle files into dictionaries.
pickle_in = open(os.path.join(constants.dataset_dir, 'keypoints_train_new.pickle'), 'rb')
keypoints_train = pickle.load(pickle_in)

pickle_in = open(os.path.join(constants.dataset_dir, 'keypoints_val_new.pickle'), 'rb')
keypoints_val = pickle.load(pickle_in)

pickle_in.close()

model = OpenPoseModel(constants.num_joints, constants.num_limbs).to(device)

criterion_conf = torch.nn.MSELoss()
criterion_paf = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters())

count = 1
for images, conf_maps, pafs in helper.gen_data(keypoints_val, batch_size=2, val=True):
    
    # Convert all to PyTorch Tensors and move to the training device
    images = torch.from_numpy(images).view(2, 3, 224, 224).float().to(device)
    conf_maps = torch.from_numpy(conf_maps).float().to(device).view(2, constants.num_joints, 56, 56)
    pafs = torch.from_numpy(pafs).float().to(device).view(2, 2, constants.num_limbs, 56, 56)
    outputs = model(images)
    loss_conf_total = 0
    loss_paf_total = 0
    for i in range(1, 4): # There are 3 stages
        # Sums losses for all 3 stages.
        conf_out = outputs[i]['conf']
        print(f'conf_out: {conf_out.shape}')
        paf_out = outputs[i]['paf']
        print(f'paf_out: {paf_out.shape}')
        loss_conf_total += criterion_conf(conf_out, conf_maps)
        loss_paf_total += criterion_paf(paf_out, pafs)
    loss = loss_conf_total + loss_paf_total
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'count: {count+1}, loss: {loss.item()}')
#    print(f'images: {type(images)}, conf: {type(conf_maps)}, pafs: {type(pafs)}')
#    print(f'images: {images.shape}, conf: {conf_maps.shape}, pafs: {pafs.shape}')
#    print(f'outputs: {type(outputs)}')
    if count == 9:
        break
    count += 1