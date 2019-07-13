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
    from constants import dataset_dir, im_width, im_height, num_joints
    
    num_images = len(indices)
    
    paf = np.zeros((num_images, im_width, im_height, 2, len(skeleton_limb_indices)), np.float16)
    
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
            for i in range(len(skeleton_limb_indices)):
                
                limb_indices = skeleton_limb_indices[i]
                
                print(i)
                print(limb_indices)
                
                joint_one_index = limb_indices[0] - 1
                joint_two_index = limb_indices[1] - 1
                # If there is 0 for visibility, skip the limb
                if all_keypoints[image_id][person][joint_one_index][2] == 0 or all_keypoints[image_id][person][joint_two_index][2] == 0:
                    pass
                else:
                    joint_one_loc = np.asarray(all_keypoints[image_id][person][joint_one_index][:2])
                    joint_two_loc = np.asarray(all_keypoints[image_id][person][joint_two_index][:2])
                    
                    norm = np.linalg.norm(joint_two_loc - joint_one_loc, ord=2)
                    
                    if norm == 0:
                        vector = joint_two_loc - joint_one_loc
                    else:
                        vector = (joint_two_loc - joint_one_loc)/norm
                    
#                    print(f'joint one: {joint_one_loc}\tjoint two: {joint_two_loc}')
                    
                    line_points = bressenham_line(joint_one_loc, joint_two_loc)
                    print(f'joint one: {joint_one_loc}\tjoint two: {joint_two_loc}')
                    if i == 3:

                        print(line_points)
                    for point in line_points:
                        paf[image_id % num_images, point[1], point[0], 0, i] = vector[0]
                        paf[image_id % num_images, point[1], point[0], 1, i] = vector[1]
                    
#                    print(vector)
                
        break
        
    return paf

val_paf = generate_paf(keypoints_val, range(4, 10), skeleton_limb_indices, True)
print(val_paf.shape)
print(val_paf[0, :, : , 0, 2])

something = val_paf[img_index, :, :, 0, 3]
# Visualize a paf for a joint
img_index = 4
for i in range(len(skeleton_limb_indices)):
    limb_index = i
    img = val_paf[img_index, :, :, 0, limb_index]
#    img = np.ceil(val_paf[img_index, :, :, 0, limb_index]).astype(np.uint8)
    img = np.where(img != 0, 255, img).astype(np.uint8)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

draw_skeleton(img_index, keypoints_val[img_index], skeleton_limb_indices, val=True)
