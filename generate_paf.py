# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 11:06:33 2019

@author: myidispg
"""

import numpy as np
import cv2

import pickle
import os

from constants import dataset_dir, num_joints, im_height, im_width, skeleton_limb_indices
from helper import get_image_name, draw_skeleton

# Read the pickle files into dictionaries.
pickle_in = open(os.path.join(dataset_dir, 'keypoints_train_new.pickle'), 'rb')
keypoints_train = pickle.load(pickle_in)

pickle_in = open(os.path.join(dataset_dir, 'keypoints_val_new.pickle'), 'rb')
keypoints_val = pickle.load(pickle_in)

pickle_in.close()