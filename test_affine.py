# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 19:20:39 2019

@author: myidispg
"""

import cv2
import numpy as np
import os

img = cv2.imread(os.path.join('Coco_Dataset', 'new_val2017', '000000000001.jpg'))

scale = 3

transformation_matrix = [[scale, 0, 0], [0, scale, 0]]
rows, cols, channels = img.shape

transformation_matrix = np.asarray(transformation_matrix, dtype=np.float32)
dest = cv2.warpAffine(img, transformation_matrix, (int(cols*scale), int(rows*scale)))

cv2.imshow('dest', dest)
cv2.waitKey()
cv2.destroyAllWindows()

def affine_transform(img, scale):
    transformation_matrix = [[scale, 0, 0], [0, scale, 0]]
    rows, cols, channels = img.shape
    
    transformation_matrix = np.asarray(transformation_matrix, dtype=np.float32)
    return cv2.warpAffine(img, transformation_matrix, (int(cols*scale), int(rows*scale)))

import pickle

from utilities.constants import dataset_dir
from utilities.helper import generate_confidence_maps, do_affine_transform, generate_paf
pickle_in = open(os.path.join(dataset_dir, 'keypoints_val_new.pickle'), 'rb')
keypoints_val = pickle.load(pickle_in)

pickle_in.close()

conf_map = generate_confidence_maps(keypoints_val, range(3, 4), val=True, affine_transform=False)
dest = affine_transform(conf_map[0], 4)

image = np.zeros((224, 224))
for i in range(17):
    image += conf_map[0, :, :, i]

cv2.imshow('image', image)
cv2.waitKey()
cv2.destroyAllWindows()

paf = generate_paf(keypoints_val, range(3, 4), val=True, affine_transform=False)
dest = affine_transform(paf[0, :, :, :, 1], 0.25)

image = np.zeros((224, 224))
for i in range(15):
    image += paf[0, :, :, 0, i] + paf[0, :, :, 1, i]

cv2.imshow('image', image)
cv2.waitKey()
cv2.destroyAllWindows()
