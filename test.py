# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 18:37:10 2019

@author: myidispg
"""

import numpy as np
import random

def generate_random_array():
    array = np.zeros((2, 92, 92, 17))
    for w in range(array.shape[0]):
        for x in range(array.shape[1]):
            for y in range(array.shape[2]):
                for z in range(array.shape[3]):
                    array[w, x, y, z] = random.random()
    return array

d1 = np.random.rand(2, 92, 92, 17)
d2 = np.random.rand(2, 92, 92, 17)

d1 = generate_random_array()
d2 = generate_random_array()

dist = np.sqrt(np.sum((d2 -  d1)**2))
test = np.abs(1/(1 - np.exp(-dist))-1) * 100


#--------Create mask------
import json
import os

with open(os.path.join(os.path.join(os.getcwd(), 'Coco_Dataset'),
                       'annotations', 'person_keypoints_val2017.json'), 'r') as JSON:
    val_dict = json.load(JSON)
with open(os.path.join(os.path.join(os.getcwd(), 'Coco_Dataset'),
                       'annotations', 'person_keypoints_train2017.json'), 'r') as JSON:
    train_dict = json.load(JSON)
    
segment_val = dict()

for annotation in val_dict['annotations']:
    print(annotation)
    segment_val[annotation['image_id']] = annotation['segmentation']

for idx in segment_val.keys():
    for person in segment_val[idx]:
        print(person)
    break
