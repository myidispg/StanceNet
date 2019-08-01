# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 18:48:23 2019

@author: myidispg
"""

import torch

import numpy as np
import cv2
import os


from models.full_model import OpenPoseModel

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

model = OpenPoseModel(15, 17).to(device).eval()

checkpoint = torch.load(os.path.join('trained_models', 'stancenet_4_epochs.pth'))
model.load_state_dict(checkpoint['model_state'])

img = cv2.imread(os.path.join('Coco_Dataset', 'val2017', '000000000785.jpg'))
img = cv2.resize(img, (400, 400))
img = torch.from_numpy(img).view(1, img.shape[0], img.shape[1], img.shape[2]).permute(0, 3, 1, 2)

img = img.to(device).float()

outputs = model(img)

paf = outputs[5]['paf'].cpu().detach().numpy()
conf = outputs[5]['conf'].cpu().detach().numpy()

conf = conf.transpose(2, 3, 1, 0)
conf = conf.reshape(conf.shape[0], conf.shape[1], -1)

paf = paf.transpose(2, 3, 1, 0)
paf = paf.reshape(paf.shape[0], paf.shape[1], 2, -1)

# Visualize Confidence map
conf_map = np.zeros((100, 100))
for i in range(conf.shape[2]):
    conf_map += conf[:, :, i]

conf_map = cv2.resize(conf_map, (400, 400))

conf_map = (conf_map > 0.7).astype(np.float32)

cv2.imshow('conf map', conf_map)
cv2.waitKey()
cv2.destroyAllWindows()

# Visualize Parts Affinity Fields
paf_map = np.zeros((100, 100))
for i in range(paf.shape[3]):
    paf_map += paf[:, :, 0, i] + paf[:, :, 1, i]

paf_map = cv2.resize(paf_map, (400, 400))

paf_map = (paf_map > 1.05).astype(np.float32)

cv2.imshow('paf map', paf_map)
cv2.waitKey()
cv2.destroyAllWindows()