# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 18:48:23 2019

@author: myidispg
"""

import torch

import numpy as np
import cv2
import os


#from models.full_model import OpenPoseModel
from models.paf_model_v2 import StanceNet
from models.bodypose_model import bodypose_model
from utilities.constants import MEAN, STD

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

#model = OpenPoseModel(15, 17).to(device).eval()
model = StanceNet(19, 38,).eval()
model.load_state_dict(torch.load('trained_model.pth'))
model = model.to(device)

img = cv2.imread(os.path.join('Coco_Dataset', 'val2017', '000000000785.jpg'))
#img = cv2.imread(os.path.join('Coco_Dataset', 'train2017', '000000000036.jpg'))
img = ((img/255)-MEAN)/STD
img = cv2.resize(img, (400, 400))
cv2.imshow('image', img)
cv2.waitKey()
cv2.destroyAllWindows()

img = torch.from_numpy(img).view(1, img.shape[0], img.shape[1], img.shape[2]).permute(0, 3, 1, 2)

#img = img.float()
img = img.to(device).float()

paf, conf = model(img)

paf = paf.cpu().detach().numpy()
conf = conf.cpu().detach().numpy()

# Remove the extra dimension of batch size
conf = np.squeeze(conf.transpose(2, 3, 1, 0))
paf = np.squeeze(paf.transpose(2, 3, 1, 0))

for i in range(conf.shape[2]):
    conf_map = cv2.resize(conf[:, :, i], (400, 400))
    print(i)
    cv2.imshow('conf', conf_map)
    cv2.waitKey()
    cv2.destroyAllWindows()

# Visualize Confidence map
conf_map = np.zeros((conf.shape[0], conf.shape[1]))
for i in range(conf.shape[2]):
    conf_map += conf[:, :, i]

conf_map = cv2.resize(conf_map, (400, 400))

conf_map = (conf_map > 1.3).astype(np.float32)

cv2.imshow('conf map', conf_map)
cv2.waitKey()
cv2.destroyAllWindows()

# Visualize Parts Affinity Fields

for i in range(19):
    paf_map = np.zeros((paf.shape[0], paf.shape[1]))
    for j in range(2):
        paf_map += paf[:, :, j, i] + paf[:, :, j, i]
    
    paf_map = cv2.resize(paf_map, (400, 400))
    
    cv2.imshow('paf map', paf_map)
    cv2.waitKey()
    cv2.destroyAllWindows()

paf_map = np.zeros((paf.shape[0], paf.shape[1]))
for i in range(paf.shape[3]):
    paf_map += paf[:, :, 0, i] + paf[:, :, 1, i]

paf_map = cv2.resize(paf_map, (400, 400))

paf_map = (paf_map > 0.3).astype(np.float32)

cv2.imshow('paf map', paf_map*255)
cv2.waitKey()
cv2.destroyAllWindows()


import matplotlib.pyplot as plt

losses = checkpoint['losses']
losses = np.asarray(losses)
plt.plot(range(losses.shape[0]), losses)
plt.show()

losses = [8334, 6148, 6241, 6658, 6073, 5948, 6669, 6380, 6299, 6538, 6187, 6219, 
          6779, 6176, 6524, 5560, 6555, 6405, 6586, 6329, 6507, 6667, 6426, 6840,
          6251, 6313, 6345, 6371, 6590, 6144, 6723, 6154, 6334, 6235, 5719, 6223]

plt.plot(range(len(losses)), losses)
plt.show()

losses2 = [5487, 6390, 6361, 6632, 6677, 6123, 6118, 6368, 6028, 6042, 6229, 6481,
           6032, 6298, 6789]

plt.plot(range(len(losses2)), losses2)
plt.show()

losses3 = losses + losses2
plt.plot(range(len(losses3)), losses3)
plt.show()

losses = [1956.4697583007812, 2009.9187451171874, 2044.7890478515626, 1880.589375,
          1912.268826904297, 2052.7990075683592, 1733.8141821289062, 1911.1695349121094, 
          1893.9861364746093, 1749.7577038574218, 1868.565889892578, 1801.087548828125,
          1821.4928649902345, 2163.157142333984, 1929.000440673828, 2046.2779663085937,
          1972.6919104003907, 2054.4262060546876, 1933.7513372802734, 1961.0204321289064,
          1852.0578210449219, 1940.6544921875, 1789.6563854980468, 1900.6374017333985,
          1997.0811608886718, 2043.9666784667968, 2052.3782849121094, 1994.7729248046876, 
          1934.7258770751953]
plt.plot(range(len(losses)), losses)
plt.show()


