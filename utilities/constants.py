# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 14:41:36 2019

@author: myidispg
"""

import os

im_height = 368
im_width = 368
im_height_small = 92
im_width_small = 92

transform_scale = 4

num_joints = 17
num_limbs = 15

# Used in creating confidence maps.
sigma = 1500

dataset_dir = os.path.join(os.getcwd(), 'Coco_Dataset')
model_path = os.path.join(os.getcwd(), 'trained_models')

# The joint pairs to create skeletons.
skeleton_limb_indices = [(3,5), (3,2), (2, 4), (7,6), (7,9), (9,11), (6,8),
                         (8,10), (7,13), (6,12), (13,15), (12,14), (15,17),
                         (14, 16), (13, 12)]
