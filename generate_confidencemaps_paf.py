# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 17:23:03 2019

@author: myidi
"""
import numpy as np

import pickle
import os

import cv2
im = cv2.imread('Coco_Dataset/val2017/000000000785.jpg', cv2.IMREAD_COLOR)

from constants import dataset_dir, num_joints, im_height, im_width
from helper import get_image_name

# Read the pickle files into dictionaries.
pickle_in = open(os.path.join(dataset_dir, 'keypoints_train.pickle'), 'rb')
keypoints_train = pickle.load(pickle_in)

pickle_in = open(os.path.join(dataset_dir, 'keypoints_val.pickle'), 'rb')
keypoints_val = pickle.load(pickle_in)

pickle_in.close()


def generate_confidence_maps(all_keypoints, im_width, im_height, num_joints, val=False):
    """
    Generate confidence maps given all_keypoints dictionary.
    The generated confidence_maps are of shape: 
        (num_images, im_width, im_height, num_joints)
    Input:
        all_keypoints: Keypoints for all the image in the dataset. It is a 
        dictionary that contains image_id as keys and keypoints for each person.
        im_height: height of image in pixels
        im_width: width of image in pixels
        num_joints: total number of joints labelled.
    Output:
        conf_map: A numpy array of shape: (num_images, im_width, im_height, num_joints)
    """
    
    num_images = len(all_keypoints)
    
    conf_map = np.zeros((num_images, im_width, im_height, num_joints), np.int8)
    
    for image_id in all_keypoints.keys():
        print(image_id)
        img_name = get_image_name(image_id)
        print(os.path.join(dataset_dir, 'val2017', img_name))
        if val:
            img = cv2.imread(os.path.join(dataset_dir, 'val2017', img_name))
        else:
            img = cv2.imread(os.path.join(dataset_dir, 'train2017', img_name))
        
        print(img.shape)
        print(len(all_keypoints[image_id]))
        for person in range(len(all_keypoints[image_id])):
            for keypoint in all_keypoints[image_id][person]:
                print(img[keypoint[0], keypoint[1]])
        
        break
    
    
    return conf_map

print(generate_confidence_maps(keypoints_val, im_width, im_height, num_joints, True).shape)
