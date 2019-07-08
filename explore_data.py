# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 17:31:55 2019

@author: myidi
"""

import numpy as np

from scipy import io
from PIL import Image

import os
import json

dataset_dir = os.path.join('C:\Machine Learning Projects\OpenPose', 'Coco_Dataset')

#mat = io.loadmat(os.path.join(dataset_dir, 'mpii_human_pose_v1_u12_1.mat'))
#
#release = mat['RELEASE']
#
#release['annolist'][0][0][0].shape
#
#print( type(release), release.shape)
#
#object1 = release[0,0]
#print(object1._fieldnames)
#print(object1.__dict__['annolist'])

with open(os.path.join(dataset_dir,
                       'annotations', 'person_keypoints_val2017.json'), 'r') as JSON:
    json_dict = json.load(JSON)

print(json_dict['annotations'][0]['keypoints'])

image_path = os.path.join(dataset_dir, 
                                'val2017', 
                                f"000000{json_dict['annotations'][0]['image_id']}.jpg", )

import cv2
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
# Draw BBOX
cv2.rectangle(image, (x1, y1), (x2, y2), (255,0,0), 2)
# Draw KeyPoints
cv2.circle(image, (142, 309), 3, (0, 255, 0), 1)
cv2.circle(image, (177, 320), 3, (0, 255, 0), 1)
cv2.circle(image, (191, 398), 3, (0, 255, 0), 1)
cv2.circle(image, (237, 317), 3, (0, 255, 0), 1)
cv2.circle(image, (233, 426), 3, (0, 255, 0), 1)
cv2.circle(image, (306, 233), 3, (0, 255, 0), 1)
cv2.circle(image, (92, 452), 3, (0, 255, 0), 1)
cv2.circle(image, (123, 468), 3, (0, 255, 0), 1)
cv2.circle(image, (251, 469), 3, (0, 255, 0), 1)
cv2.circle(image, (162, 551), 3, (0, 255, 0), 1)

cv2.imshow('image', image)
cv2.waitKey(10000)
cv2.destroyAllWindows()
