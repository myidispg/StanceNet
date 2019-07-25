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

from utilities.constants import dataset_dir, num_joints, im_height, im_width, skeleton_limb_indices
from utilities.helper import get_image_name
from visualization.visualization_functions import draw_skeleton

# Read the pickle files into dictionaries.
pickle_in = open(os.path.join(dataset_dir, 'keypoints_train_new.pickle'), 'rb')
keypoints_train = pickle.load(pickle_in)

pickle_in = open(os.path.join(dataset_dir, 'keypoints_val_new.pickle'), 'rb')
keypoints_val = pickle.load(pickle_in)

pickle_in.close()


def generate_confidence_maps(all_keypoints, indices, val=False, sigma=7):
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
        
        heatmap_image = np.zeros((im_width, im_height, num_joints))
        
        # For a person in the image
        for person in range(len(all_keypoints[image_id])):
            # For all keypoints in the image.
            for part_num in range(len(all_keypoints[image_id][person])):
                # Get the pixel values at a given keypoint across all 3 channels.
                # Note that our labels have images (im_width, im_height),
                # OpenCV has (im_height, im_width)
                x_index = all_keypoints[image_id][person][part_num][1]
                y_index = all_keypoints[image_id][person][part_num][0]
                visibility = all_keypoints[image_id][person][part_num][2]
                
#                print(f'image_id: {image_id}')
#                print(f'person: {person}')
#                print(f'part_num: {part_num}')
#                print(f'x_index: {x_index}, y_index: {y_index}')
#                print(f'x_spread: {x_spread}\ty_spread: {y_spread}')
                
                # Generate heatmap only around the keypoint, leave others as 0
                if visibility != 0:
                    x_ind, y_ind = np.meshgrid(np.arange(im_width), np.arange(im_height))
                    numerator = (-(x_ind-x_index)**2) + (-(y_ind-y_index)**2)
                    heatmap_joint = np.exp(numerator/sigma)
                    heatmap_image[:, :, part_num] = np.maximum(heatmap_joint, heatmap_image[:, :, part_num])
                    
                    
                    conf_map[image_id % num_images, :, :, :] = heatmap_image                         
            
            # Find max of heatmaps for all persons.
            
            
#        break
        
    
    return conf_map

import time

time1 = time.time_ns() // 1000000 
val_conf_maps = generate_confidence_maps(keypoints_val, range(64), val=True)
time2 = time.time_ns() // 1000000 
print(f'The operation took: {time2 - time1} milliseconds')

# Visualize a confidence map.
heatmap_global = np.zeros((224, 224))

index = 1
for i in range(17):
    heatmap = (val_conf_maps[index, :, :, i] * 255)
    heatmap = np.transpose(heatmap).astype(np.uint8)
    heatmap_global += heatmap
    img = cv2.imread(os.path.join(dataset_dir, 'new_val2017', get_image_name(index)))
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img = cv2.addWeighted(img, 0.6, colored, 0.4, 0)
#    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#    img = cv2.resize(img, (700, 700))
#    cv2.imwrite(f'{index}_{i}.jpg', img)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#    break

heatmap_global = heatmap_global.astype(np.uint8)
img = cv2.imread(os.path.join(dataset_dir, 'new_val2017', get_image_name(index)))
colored = cv2.applyColorMap(heatmap_global, cv2.COLORMAP_JET)
img = cv2.addWeighted(img, 0.6, colored, 0.4, 0)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

draw_skeleton(index, keypoints_val[index], skeleton_limb_indices, val=True)

