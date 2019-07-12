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
from helper import get_image_name, draw_skeleton

# Read the pickle files into dictionaries.
pickle_in = open(os.path.join(dataset_dir, 'keypoints_train_new.pickle'), 'rb')
keypoints_train = pickle.load(pickle_in)

pickle_in = open(os.path.join(dataset_dir, 'keypoints_val_new.pickle'), 'rb')
keypoints_val = pickle.load(pickle_in)

pickle_in.close()

def generate_paf(all_keypoints, indices, skeleton_limb_indices, val=False):
    """
    Generate Part Affinity Fields given a batch of keypoints.
    Inputs:
        all_keypoints: Keypoints for all the images in the dataset. It is a 
        dictionary that contains image_id as keys and keypoints for each person.
        indices: The list of indices from the keypoints for which PAFs are to 
        be generated.
        skeleton_limb_indices: The indices to create limbs from joints.
        val(Default=False): True if validation set, False otherwise. 
    """
    
    import cv2
    import os
    import numpy as np
    import math
    from constants import dataset_dir, im_width, im_height, num_joints
    
    num_images = len(indices)
    
    paf = np.zeros((num_images, im_width, im_height, 2, num_joints), np.float16)
    
    # For image in all_images
    for image_id in indices:
        
        image_name = get_image_name(image_id)
        
        if val:
            image_path = os.path.join(dataset_dir, 'new_val2017', image_name)
        else:
            image_path = os.path.join(dataset_dir, 'new_train2017', image_name)
            
        print(image_path)
        
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # For each person in the image
        for person in range(len(all_keypoints[image_id])):
            # For each limb in the skeleton
            for limb_indices in skeleton_limb_indices:
#                print(limb_indices)
                joint_one_index = limb_indices[0] - 1
                joint_two_index = limb_indices[1] - 1
                joint_one_loc = np.asarray(all_keypoints[image_id][person][joint_one_index][:2])
                joint_two_loc = np.asarray(all_keypoints[image_id][person][joint_two_index][:2])
                
                print(joint_one_loc)
                print(joint_two_loc)
                print()
#                print(joint_one_loc -  joint_two_loc)
                
                norm = np.linalg.norm(joint_one_loc - joint_two_loc, ord=2)
#                print(norm)
                
                if norm == 0:
                    vector = joint_one_loc - joint_two_loc
                else:
                    vector = (joint_one_loc - joint_two_loc)/norm
                
        break
        
    return paf

val_paf = generate_paf(keypoints_val, range(10), skeleton_limb_indices, True)
print(val_paf.shape)
    
draw_skeleton(0, keypoints_val[0], skeleton_limb_indices, val=True)
