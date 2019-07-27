# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 08:20:43 2019

@author: myidispg
"""

import os
import pickle

from visualization.visualization_functions import draw_skeleton
from utilities.constants import dataset_dir, im_width, im_height, skeleton_limb_indices


pickle_in = open(os.path.join(dataset_dir, 'keypoints_val_new.pickle'), 'rb')
keypoints_val = pickle.load(pickle_in)

pickle_in = open(os.path.join(dataset_dir, 'keypoints_train_new.pickle'), 'rb')
keypoints_train = pickle.load(pickle_in)

pickle_in.close()

draw_skeleton(1, keypoints_train[1], skeleton_limb_indices,
              val=False, wait_time=None)
