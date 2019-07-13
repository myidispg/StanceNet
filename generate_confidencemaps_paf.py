# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 17:23:03 2019

@author: myidi
"""
import numpy as np

import pickle
import os

import cv2
#im = cv2.imread('Coco_Dataset/new_val2017/000000000000.jpg', cv2.IMREAD_COLOR)

from constants import dataset_dir, num_joints, im_height, im_width, skeleton_limb_indices
from helper import get_image_name, draw_skeleton

# Read the pickle files into dictionaries.
pickle_in = open(os.path.join(dataset_dir, 'keypoints_train_new.pickle'), 'rb')
keypoints_train = pickle.load(pickle_in)

pickle_in = open(os.path.join(dataset_dir, 'keypoints_val_new.pickle'), 'rb')
keypoints_val = pickle.load(pickle_in)

pickle_in.close()


def generate_confidence_maps(all_keypoints, indices, val=False, sigma=1500):
    """
    Generate confidence maps given all_keypoints dictionary.
    The generated confidence_maps are of shape: 
        (num_images, im_width, im_height, num_joints)
    Input:
        all_keypoints: Keypoints for all the image in the dataset. It is a 
        dictionary that contains image_id as keys and keypoints for each person.
        indices: a list of indices for which the conf_maps are to be generated.
        val: True if used for validation set, else false
        sigma: used to generate the confidence score. Higher values lead higher score.
    Output:
        conf_map: A numpy array of shape: (num_images, im_width, im_height, num_joints)
    """
    import math
    
    num_images = len(indices)
    
    conf_map = np.zeros((num_images, im_width, im_height, num_joints), np.float16)
    
    # For image in all images
    for image_id in indices:
        
        img_name = get_image_name(image_id)
        if val:
            img = cv2.imread(os.path.join(dataset_dir, 'new_val2017', img_name))
        else:
            img = cv2.imread(os.path.join(dataset_dir, 'new_train2017', img_name))
        
        # For a person in the image
        for person in range(len(all_keypoints[image_id])):
            # For all keypoints in the image.
            for part_num in range(len(all_keypoints[image_id][person])):
#            for keypoint in all_keypoints[image_id][person]:
                # Get the pixel values at a given keypoint across all 3 channels.
                # Note that our labels have images (im_width, im_height),
                # OpenCV has (im_height, im_width)
                x_index = all_keypoints[image_id][person][part_num][0]
                y_index = all_keypoints[image_id][person][part_num][1]
                pix_1, pix_2, pix_3 = list(img[y_index, x_index, :])
                
#                print(f'pix1: {pix_1}, pix2: {pix_2}, pix3: {pix_3}')
                
                norm = -((0-pix_1)**2) - ((0-pix_2)**2) - ((0-pix_3)**2)
                
#                print(math.exp((norm) / (sigma) ** 2))
                
                confidence_score = math.exp((norm) / (sigma) ** 2)
                
#                print(f'Confidence score: {confidence_score}')
                
                conf_map[image_id % num_images, x_index, y_index] = confidence_score
#        break
        
    
    return conf_map

#print(generate_confidence_maps(keypoints_val, im_width, im_height, num_joints, True).shape)

val_conf_maps = generate_confidence_maps(keypoints_val, range(64), val=True)

# Visualize a confidence map.
index = 1
img = np.ceil(val_conf_maps[index, :, :, 0]).astype(np.uint8)
img = np.where(img > 0, 255, img)
img = np.transpose(img)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

draw_skeleton(16, keypoints_val[16], skeleton_limb_indices, val=True)

