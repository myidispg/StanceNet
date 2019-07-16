# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 11:06:33 2019

@author: myidispg
"""

import numpy as np
import cv2

import pickle
import os

from constants import dataset_dir, num_joints, im_height, im_width, skeleton_limb_indices
from helper import get_image_name, draw_skeleton, bressenham_line

# Read the pickle files into dictionaries.
pickle_in = open(os.path.join(dataset_dir, 'keypoints_train_new.pickle'), 'rb')
keypoints_train = pickle.load(pickle_in)

pickle_in = open(os.path.join(dataset_dir, 'keypoints_val_new.pickle'), 'rb')
keypoints_val = pickle.load(pickle_in)

pickle_in.close()

def generate_paf(all_keypoints, indices, skeleton_limb_indices, sigma=5, val=False):
    """
    Generate Part Affinity Fields given a batch of keypoints.
    Inputs:
        all_keypoints: Keypoints for all the images in the dataset. It is a 
        dictionary that contains image_id as keys and keypoints for each person.
        indices: The list of indices from the keypoints for which PAFs are to 
        be generated.
        skeleton_limb_indices: The indices to create limbs from joints.
        val(Default=False): True if validation set, False otherwise. 
        
    Outputs:
        paf: A parts affinity fields map of shape: 
            (batch_size, im_width, im_height, 2, num_joints)
    """
    
    import cv2
    import os
    import numpy as np
    from constants import dataset_dir, im_width, im_height, num_joints
    
    num_images = len(indices)
    
    paf = np.zeros((num_images, im_width, im_height, 2, len(skeleton_limb_indices)), np.float16)
    
    # For image in all_images
    for image_id in indices:
        print(f'Image: {image_id}')
        # For each person in the image
        for person in range(len(all_keypoints[image_id])):
            # For each limb in the skeleton
            for limb in range(len(skeleton_limb_indices)):
                
                limb_indices = skeleton_limb_indices[limb]
                
                joint_one_index = limb_indices[0] - 1
                joint_two_index = limb_indices[1] - 1
                
                joint_one_loc = np.asarray(all_keypoints[image_id][person][joint_one_index])
                joint_two_loc = np.asarray(all_keypoints[image_id][person][joint_two_index])
                
#                if image_id == 0:
#                    print(f'limb: {limb}')
#                    print(f'joint 1: {joint_one_index}, joint 2: {joint_two_index}')
#                    print(f'joint_one_loc: {joint_one_loc}\tjoint_two_loc: {joint_two_loc}')
#                
                # If there is 0 for visibility, skip the limb
                if all_keypoints[image_id][person][joint_one_index][2] != 0 and all_keypoints[image_id][person][joint_two_index][2] != 0:
                    joint_one_loc = np.asarray(all_keypoints[image_id][person][joint_one_index][:2])
                    joint_two_loc = np.asarray(all_keypoints[image_id][person][joint_two_index][:2])
#                    print(f'joint_one_loc: {joint_one_loc}\tjoint_two_loc: {joint_two_loc}')
                    
#                    norm = np.linalg.norm(joint_two_loc - joint_one_loc, ord=2)
                    norm = ((joint_one_loc[0] - joint_two_loc[0]) ** 2 + (joint_one_loc[1] - joint_two_loc[1]) ** 2) ** (1/2)
#                    
#                    if norm == 0:
#                        break
#                    else:
#                        vector = (joint_two_loc - joint_one_loc)/norm
                    vector = (joint_two_loc - joint_one_loc)/norm
#                    print(f'vector: {vector}')
#                    print(f'norm: {norm}')                    
                    # Found how to get perpendicular 2-d vector here: 
                    # https://gamedev.stackexchange.com/questions/70075/how-can-i-find-the-perpendicular-to-a-2d-vector
                    perpendicular_vec = [-vector[1], vector[0]]
#                    print(f'vector: {vector}')
#                    print(f'Perpendicular: {perpendicular_vector}')
                    x, y = np.meshgrid(np.arange(im_width), np.arange(im_height))
                    along_limb = vector[0] * (x-joint_one_loc[0]) + vector[1] * (y-joint_one_loc[1])
                    across_limb = np.abs(perpendicular_vec[0] * (x-joint_one_loc[0]) + perpendicular_vec[1] * (y - joint_one_loc[1])) 
                    
                    cond_1 = along_limb >= 0
                    cond_2 = along_limb <= norm
                    cond_3 = across_limb <= sigma
                    mask = cond_1 & cond_2 & cond_3
                    
                    paf[image_id % num_images, :, :, 0, limb] += np.transpose(mask) * vector[0]
                    paf[image_id % num_images, :, :, 1, limb] += np.transpose(mask) * vector[1]
                    
#                    for i in range(im_width):
#                        for j in range(im_height):
#                            distance_vec = [i-joint_one_loc[0],j-joint_one_loc[1]]
##                            print(f'dis_vec: {distance_vec}')
#                            if 0 <= np.dot(vector, distance_vec) <= norm:
#                                if abs(np.dot(perpendicular_vec, distance_vec)) <= sigma:
##                                    print(f'i: {i}, j: {j}, limb: {limb}')
#                                    paf[image_id % num_images, i, j, 0, limb] = vector[0]
#                                    paf[image_id % num_images, i, j, 1, limb] = vector[1]

#                    print(np.dot(vector, perpendicular_vector))
#        
#        break
    return paf

import time

time1 = time.time_ns() // 1000000 
val_paf = generate_paf(keypoints_val, range(10), skeleton_limb_indices, 2, False)
time2 = time.time_ns() // 1000000 
print(f'The operation took: {time2 - time1} milliseconds')

test = val_paf[0, :, :, 0, 5] +  val_paf[0, :, :, 1, 5]

# Visualize a paf for a joint
for img_index in range(10):
    for limb_index in range(len(skeleton_limb_indices)):
        img = val_paf[img_index, :, :, 0, limb_index] + val_paf[img_index, :, :, 0, limb_index]
        img = img.transpose()
    #    img = np.ceil(val_paf[img_index, :, :, 0, limb_index]).astype(np.uint8)
        img = np.where(img != 0, 255, img).astype(np.uint8)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    draw_skeleton(img_index, keypoints_val[img_index], skeleton_limb_indices, val=True)
    
    if img_index == 1:
        break
    
