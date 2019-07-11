# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 14:41:36 2019

@author: myidispg
"""

import os

im_height = 240
im_width = 240

num_joints = 17

# Used in creating confidence maps.
sigma = 1500

dataset_dir = os.path.join('C:\Machine Learning Projects\OpenPose', 'Coco_Dataset')

# The joint pairs to create skeletons.
skeleton_limb_indices = [(3,5), (3,2), (2, 4), (7,6), (7,9), (9,11), (6,8),
                         (8,10), (7,13), (6,12), (13,15), (12,14), (15,17),
                         (14, 16), (13, 12)]
