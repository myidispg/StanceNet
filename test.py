# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 18:37:10 2019

@author: myidispg
"""

import numpy as np
from pycocotools.coco import COCO
import os
from data_process.process_functions import generate_confidence_maps, generate_paf

import cv2

img = cv2.imread(os.path.join(os.getcwd(), 'Coco_Dataset', 'val2017', '000000000785.jpg')).transpose(1, 0, 2)

coco = COCO(os.path.join(os.path.join(os.getcwd(), 'Coco_Dataset'),
                       'annotations', 'person_keypoints_val2017.json'))
# Get category id for person
person_ids = coco.getCatIds(catNms=['person'])
# Get img_id for all images with people.
person_indices = sorted(coco.getImgIds(catIds=person_ids))
#annids = coco.getAnnIds()
# get ann_indices of all annotations for the image
annids = sorted(coco.getAnnIds(person_indices[1]))
# Load all the annotations for a person
anns = coco.loadAnns(annids)
mask = np.zeros(img.shape[:2], np.uint8)
keypoints = []
for ann in anns:
    if ann['num_keypoints'] != 0:
        keypoints.append(ann['keypoints'])
#    print(coco.annToMask(ann).transpose().shape())
    mask = mask | coco.annToMask(ann).transpose()
shape = img.shape
conf_map = generate_confidence_maps(keypoints, shape)
paf = generate_paf(keypoints, shape)

map_ = np.zeros((conf_map.shape[:2]))
for i in range(15):
    map_ += paf[:, :, 0, i] + paf[:, :, 1, i]
    
cv2.imshow('mask', map_*255)
cv2.waitKey()
cv2.destroyAllWindows()

test = list(range(51))
for i in range(0, len(test), 3):
    print(i)