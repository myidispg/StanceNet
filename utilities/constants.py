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

img_size = 400 # All the images will be resized to this size. PAFs, mask etc. will be adjusted accordingly.

# IMAGE NET CONSTANTS
MEAN = [0.485, 0.456, 0.406],
STD = [0.229, 0.224, 0.225]

transform_scale = 8

num_joints = 18
num_limbs = 19

# Used in creating confidence maps.
sigma = 1500

dataset_dir = os.path.join(os.getcwd(), 'Coco_Dataset')
model_path = os.path.join(os.getcwd(), 'trained_models')

# Since I had to use a pre-trained model for inference as training was taking 
# a very long time on my machine, I had to map the keypoints order in coco data
# to the joint indexes used in the pre-trained model.
# in the pre-trained model's skeleton, joint 1 is the neck joint.
# First is coco index, second is required index
joint_map_coco = [(0, 0), (17, 1), (6, 2), (8, 3), (10, 4), (5, 5), (7, 6), 
                   (9, 7), (12, 8), (14, 9), (16, 10), (11, 11), (13, 12),
                   (15, 13), (2, 14), (1, 15), (4, 16), (3, 17)]

#
#
#
#joint_map_coco = [(0, 0), (1, 15), (2, 14), (3, 17), (4, 16), (5, 5), (6, 2), 
#                  (7, 6), (8, 3), (9, 7), (10, 4), (11, 11), (12, 8), (13, 12), 
#                  (14, 9), (15, 13), (16, 10), (17, 1)] # 17 is the added neck.
                    

# The joint pairs to create skeletons.
#skeleton_limb_indices = [(3,5), (3,2), (2, 4), (7,6), (7,9), (9,11), (6,8),
#                         (8,10), (7,13), (6,12), (13,15), (12,14), (15,17),
#                         (14, 16), (13, 12)]

skeleton_limb_indices = [(1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), 
                         (1, 2), (2, 3), (3, 4), (2, 16),  (1, 5), (5, 6),
                         (6, 7), (5, 17), (1, 0), (0, 14), (0, 15), (14, 16),
                         (15, 17)]

