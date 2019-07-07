# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 17:31:55 2019

@author: myidi
"""

import numpy as np

from scipy import io
from PIL import Image

import os

dataset_dir = 'C:\Machine Learning Projects\OpenPose\MPII Dataset'

mat = io.loadmat(os.path.join(dataset_dir, 'mpii_human_pose_v1_u12_1.mat'))

release = mat['RELEASE']

release['annolist'][0][0][0].shape

print( type(release), release.shape)

object1 = release[0,0]
print(object1._fieldnames)
print(object1.__dict__['annolist'])

import json

with open('Coco Dataset/annotations/person_keypoints_val2017.json', 'r') as JSON:
    json_dict = json.load(JSON)

