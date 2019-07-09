# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 14:41:02 2019

@author: myidi
"""

import os
import json

from helper import draw_skeleton, group_keypoints
from constants import dataset_dir, skeleton_limb_indices

with open(os.path.join(dataset_dir,
                       'annotations', 'person_keypoints_val2017.json'), 'r') as JSON:
    val_dict = json.load(JSON)

with open(os.path.join(dataset_dir,
                       'annotations', 'person_keypoints_train2017.json'), 'r') as JSON:
    train_dict = json.load(JSON)

print(f'The length of train annotations is: {len(train_dict["annotations"])}')
print(f'The length of validation annotations is: {len(val_dict["annotations"])}')

"""
The training data is going to be converted into a single dictionary
"""

keypoints_val = dict()

for annotation in val_dict['annotations']:
    if annotation['num_keypoints'] != 0:
        if annotation['image_id'] in keypoints_val.keys():
            keypoints_val[annotation['image_id']].append(
                    group_keypoints(annotation['keypoints']))
        else:
            keypoints_val[annotation['image_id']] = [group_keypoints(
                    annotation['keypoints'])]
            

keypoints_train = dict()

for annotation in val_dict['annotations']:
    if annotation['num_keypoints'] != 0:
        if annotation['image_id'] in keypoints_train.keys():
            keypoints_train[annotation['image_id']].append(
                    group_keypoints(annotation['keypoints']))
        else:
            keypoints_train[annotation['image_id']] = [group_keypoints(
                    annotation['keypoints'])]

count = (0, 0)

for key in keypoints_val.keys():
    if count[1] < len(keypoints_val[key]):
        count = (key, len(keypoints_val[key]))

draw_skeleton(489842, keypoints_val[489842], skeleton_limb_indices, val=True)
