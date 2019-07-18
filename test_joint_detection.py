# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 18:02:16 2019

@author: myidispg
"""

import pickle

import os
import cv2

import numpy as np

from utilities.constants import dataset_dir
from utilities.helper import generate_confidence_maps

from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure

# Read the pickle files into dictionaries.
pickle_in = open(os.path.join(dataset_dir, 'keypoints_train_new.pickle'), 'rb')
keypoints_train = pickle.load(pickle_in)

pickle_in = open(os.path.join(dataset_dir, 'keypoints_val_new.pickle'), 'rb')
keypoints_val = pickle.load(pickle_in)

pickle_in.close()

conf_map = generate_confidence_maps(keypoints_val, range(5), True, sigma=7)

# Take confidence map for joint one and reshape to 2-d
joint_one = conf_map[4, :, :, 1].reshape(224, 224)
# Threshold the map to eliminate low probability locations.
joint_one = np.where(joint_one >= 0.7, joint_one, 0)
# Apply a 2x1 convolution(kind of). This replaces a 2x1 window with the max of that window.
peaks = maximum_filter(joint_one.astype(np.float64), footprint=generate_binary_structure(2, 1))
# Replace all the 0's with 0.1 (it is well below threshold).
peaks = np.where(peaks == 0, 0.1, peaks)
# Now equate the peaks with joint_one.
# This works because the maxima in joint_one will be at a single place and
# equating with peaks will result in a single peak with all others as 0. 
peaks = np.where(peaks == joint_one, peaks, 0)

