# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 14:41:02 2019

@author: myidi
"""

import os
import json

import cv2

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

def group_keypoints(keypoints):
    """
    Given a list of 51 keypoints, groups the keypoints into a list of 17 tuples.
    Each tuple is (x, y, v) where 
    v=0: not labeled, v=1: labeled but not visible, v=2: labeled and visible.
    """
    arranged_keypoints = list()
    for i in range(0, len(keypoints), 3):
        arranged_keypoints.append((keypoints[i], keypoints[i+1], keypoints[i+2]))
    
    return arranged_keypoints

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

def get_image_name(image_id):
    """
    Given an image id, adds zeros before it so that the image name length is 
    of 12 digits as required by the database.
    Input:
        image_id: the image id without zeros.
    Output:
        image_name: image name with zeros added and the .jpg extension
    """
    num_zeros = 12 - len(str(image_id))
    zeros = '0' * num_zeros
    image_name = zeros + str(image_id) + '.jpg'
    
    return image_name

def draw_skeleton(image_id, all_keypoints, skeleton_limb_indices, val=False):
    """
    Given the image_id and keypoints of the image, draws skeleton accordingly.
    Inputs:
        image_id: The id of the image as per the dataset.
        all_keypoints: A 2d list with keypoints of size: (num_people, 51)
                   Here, 51 is for the keyppoints in a single person.
        skeleton_limb_indices: The joint indices to make connections from 
                               keypoints to make a skeleton. Defined in constants.
        val: True is using validation data. False by default.

    Outputs: None. Only shows the image.                       
    """
    
    image_name = get_image_name(image_id)
    
    if val:
        image_path = os.path.join(dataset_dir, 
                                    'val2017', 
                                    image_name)
    else:
        image_path = os.path.join(dataset_dir,
                                  'train2017',
                                  image_name)
    
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    # For each keypoint in keypoints, draw the person.
    for person_keypoints in all_keypoints:
        
        # Display joints
        for keypoint in person_keypoints:
            if keypoint[2] != 0:
                cv2.circle(img, (keypoint[0], keypoint[1]), 0, (0, 255, 255), 6)
        
        # Draw limbs for the visible joints
        # Note: 1 is subtracted because indices start from 0.
        for joint_index in skeleton_limb_indices:
            if person_keypoints[joint_index[0]-1][2] != 0:
                if person_keypoints[joint_index[1]-1][2] != 0:
                    x1 = person_keypoints[joint_index[0]-1][0]
                    y1 = person_keypoints[joint_index[0]-1][1]
                    x2 = person_keypoints[joint_index[1]-1][0]
                    y2 = person_keypoints[joint_index[1]-1][1]
                    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 4)

                
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
draw_skeleton(489842, keypoints_val[489842], skeleton_limb_indices, val=True)
