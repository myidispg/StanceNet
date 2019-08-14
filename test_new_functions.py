# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 17:46:44 2019

@author: myidispg
"""

from pycocotools.coco import COCO
import os
import numpy as np

from data_process.process_functions import add_neck_joint, generate_confidence_maps
from data_process.process_functions import adjust_keypoints,  generate_paf

import cv2

coco = COCO(os.path.join(os.path.join(os.getcwd(), 'Coco_Dataset'),
                       'annotations', 'person_keypoints_val2017.json'))
person_ids = coco.getCatIds(catNms=['person'])
img_indices = sorted(coco.getImgIds(catIds=person_ids))
annotations_indices = coco.getAnnIds(img_indices[2])
annotations = coco.loadAnns(annotations_indices) 

keypoints = []

for annotation in annotations:
    if annotation['num_keypoints'] != 0:
        keypoints.append(annotation['keypoints'])

img = cv2.imread(os.path.join('Coco_Dataset', 'val2017', '000000000872.jpg')).transpose(1, 0, 2)
orig_shape = img.shape
img = cv2.resize(img, (368, 368))

keypoints = adjust_keypoints(keypoints, orig_shape)

for list_ in keypoints:
    count = 0
    for i in range(0, len(list_), 3):
        x_index = list_[i]
        y_index = list_[i+1]
        cv2.circle(img, (y_index, x_index), 3, (255,0,0))
        cv2.putText(img, str(count), (y_index + 4, x_index + 4), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), lineType=cv2.LINE_AA)
        count += 1
    # Add neck
neck = add_neck_joint(keypoints)
for keypoint in neck:
    cv2.circle(img, (keypoint[1], keypoint[0]), 3, (0, 255, ))

cv2.imshow('image', img)
cv2.waitKey()
cv2.destroyAllWindows()

conf_maps = generate_confidence_maps(keypoints, 400)

for i in range(conf_maps.shape[2]):
    map_ = conf_maps[:, :, i]
    map_ = cv2.resize(map_, (400, 400))
    cv2.imshow('map',  map_* 255)
    cv2.waitKey()
    cv2.destroyAllWindows()