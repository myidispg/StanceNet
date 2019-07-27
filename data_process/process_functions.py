# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 09:42:17 2019

@author: myidispg
"""

import cv2
import numpy as np

from utilities.constants import dataset_dir, im_width_small, im_height_small
from utilities.constants import im_height, im_width, num_joints, skeleton_limb_indices

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

def do_affine_transform(img, scale=0.25):
    """
    Takes an image and perform scaling affine transformation on it.
    Inputs: 
        img: the image on which to apply the transform
        scale: The factor by which to scale. Default 0.25 as required by model.
    Outputs:
        tranformed image is returned.
    """
    transformation_matrix = [[scale, 0, 0], [0, scale, 0]]
    rows, cols = img.shape[0], img.shape[1]
    
#    print(f'rows: {rows}, cols: {cols}')
    
    transformation_matrix = np.asarray(transformation_matrix, dtype=np.float32)
    return cv2.warpAffine(np.float32(img), transformation_matrix, (int(cols*scale), int(rows*scale)))

def generate_confidence_maps(all_keypoints, img_id, affine_transform=True, sigma=7):
    """
    Generate confidence maps for a batch of indices.
    The generated confidence_maps are of shape: 
        (batch_size, im_width, im_height, num_joints)
    Input:
        all_keypoints: Keypoints for all the images in the dataset. It is a 
        dictionary that contains image_id as keys and keypoints for each person.
        img_id: The img id for which the data is accessed.
        affine_transform: A bool to whether apply affine transform or not
        sigma: used to control the spread of the peak.
    Output:
        conf_map: A numpy array of shape: (batch_size, im_width, im_height, num_joints)
    """
    
    if affine_transform:
        conf_map = np.zeros((im_width_small, im_height_small, num_joints), np.float32)
    else:
        conf_map = np.zeros((im_width, im_height, num_joints), np.float32)
    
    # For a person in the image
    for person in range(len(all_keypoints[img_id])):
        # For all keypoints of the person
        for part_num in range(len(all_keypoints[img_id][person])):
            # For keypoint in all_keypoints[image_id][person]:
            # Get the pixel values at a given keypoint across all 3 channels.
            # Note that our labels have images (im_width, im_height),
            # OpenCV has (im_height, im_width)
            x_index = all_keypoints[img_id][person][part_num][0]
            y_index = all_keypoints[img_id][person][part_num][1]
            visibility = all_keypoints[img_id][person][part_num][2]
            
            if visibility != 0:
                x_ind, y_ind = np.meshgrid(np.arange(im_width), np.arange(im_height))
                numerator = (-(x_ind-x_index)**2) + (-(y_ind-y_index)**2)
                heatmap_joint = np.exp(numerator/sigma).transpose()
                if affine_transform:
                    heatmap_joint = do_affine_transform(heatmap_joint)
#                print(conf_map.shape)
                conf_map[:, :, part_num] = np.maximum(heatmap_joint, conf_map[:, :, part_num])
#                if affine_transform:
#                    conf_map = do_affine_transform(conf_map)  
    return conf_map
    

def generate_paf(all_keypoints, img_id, sigma=5, affine_transform=True):
    """
    Generate Part Affinity Fields given a batch of keypoints.
    Inputs:
        all_keypoints: Keypoints for all the images in the dataset. It is a 
        dictionary that contains image_id as keys and keypoints for each person.
        img_id: The image id for which the data is accessed.
        sigma: The width of a limb in pixels.
        affine_transform: A bool to check whether to apply affine_transform or not
    Outputs:
        paf: A parts affinity fields map of shape: 
            (batch_size, im_width, im_height, 2, num_joints)
    """
    
    
    if affine_transform:
        paf = np.zeros((im_width_small, im_height_small, 2,
                        len(skeleton_limb_indices)), np.float32)
    else:
        paf = np.zeros((im_width, im_height, 2, len(skeleton_limb_indices)),
                       np.float32)
    
    # For each person in the image
    for person in range(len(all_keypoints[img_id])):
        # For each limb in the skeleton
        for limb in range(len(skeleton_limb_indices)):
            
            limb_indices = skeleton_limb_indices[limb]
            
            joint_one_index = limb_indices[0] - 1
            joint_two_index = limb_indices[1] - 1
            
            # If there is 0 for visibility, skip the limb
            if all_keypoints[img_id][person][joint_one_index][2] != 0 and all_keypoints[img_id][person][joint_two_index][2] != 0:
                joint_one_loc = np.asarray(all_keypoints[img_id][person][joint_one_index][:2])
                joint_two_loc = np.asarray(all_keypoints[img_id][person][joint_two_index][:2])
                
                part_line_segment = joint_two_loc - joint_one_loc
                norm = np.linalg.norm(part_line_segment)

                norm = norm + 1e-8 if norm == 0 else norm # To make sure it is not equal to zero.

                vector = (part_line_segment)/norm

                # https://gamedev.stackexchange.com/questions/70075/how-can-i-find-the-perpendicular-to-a-2d-vector
                perpendicular_vec = [-vector[1], vector[0]]
                
                x, y = np.meshgrid(np.arange(im_width), np.arange(im_height))
                # Formula according to the paper
                along_limb = vector[0] * (x-joint_one_loc[0]) + vector[1] * (y-joint_one_loc[1])
                across_limb = np.abs(perpendicular_vec[0] * (x-joint_one_loc[0]) + perpendicular_vec[1] * (y - joint_one_loc[1])) 
                
                # Check the given conditions
                cond_1 = along_limb >= 0
                cond_2 = along_limb <= norm
                cond_3 = across_limb <= sigma
                mask = (cond_1 & cond_2 & cond_3).astype(np.float32)
                
                # put the values
                if affine_transform:
                    mask = do_affine_transform(mask)
                    paf[:, :, 0, limb] += mask * vector[0]
                    paf[:, :, 1, limb] += mask * vector[1]
                else:
                    paf[:, :, 0, limb] += mask * vector[0]
                    paf[:, :, 1, limb] += mask * vector[1]
    return paf
                
                