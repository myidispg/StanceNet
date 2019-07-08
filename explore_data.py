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
    val_dict = json.load(JSON)


total_val_data = len(val_dict['annotations'])

print(val_dict['annotations'][3]['keypoints'])

def display_im_keypoints(index):
    """
    Takes in the index of the image from the validation annotations,
    returns the keypoints from that image that are labeled and shows image
    with keypoints.
    """
    
    image_id = val_dict['annotations'][index]['image_id']
    
    image_path = os.path.join(dataset_dir, 
                                'val2017', 
                                f"000000000{image_id}.jpg")
    
    keypoints = val_dict['annotations'][index]['keypoints']
    
    filtered_keypoints = list()
    
    for i in range(0, len(keypoints), 3):
        if keypoints[i+2] != 0:
            filtered_keypoints.append((keypoints[i], keypoints[i+1]))
            
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    if img is None:
        print('None Image')
    
    count = 1
    for keypoint in filtered_keypoints:
        cv2.circle(img, keypoint, 0, (0, 255, 255), 8)
        cv2.putText(img, str(count), keypoint, cv2.FONT_HERSHEY_PLAIN, 1,
                    (0, 255, 255), 1, cv2.LINE_AA)        
        count += 1
        
    for i in range(0, len(filtered_keypoints)-1, 2):
        cv2.line(img, filtered_keypoints[i], filtered_keypoints[i+1], (0, 255, 255), 2)
        
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return image_id, filtered_keypoints
    
img_id, keypoints = display_im_keypoints(4875)

index = None
for i in range(len(val_dict['annotations'])):
    if val_dict['annotations'][i]['image_id'] == 785:
        index = i
        break
    
print(val_dict['annotations'][index])

image_path = os.path.join(dataset_dir, 
                                'val2017', 
                                f"000000{val_dict['annotations'][10287]['image_id']}.jpg", )

import cv2
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

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
