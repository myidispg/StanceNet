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

def generate_confidence_maps(all_keypoints, indices, val=False, affine_transform=True, sigma=7):
    """
    Generate confidence maps for a batch of indices.
    The generated confidence_maps are of shape: 
        (batch_size, im_width, im_height, num_joints)
    Input:
        all_keypoints: Keypoints for all the images in the dataset. It is a 
        dictionary that contains image_id as keys and keypoints for each person.
        indices: a list of indices for which the conf_maps are to be generated.
        val: True if used for validation set, else false
        affine_transform: A bool to whether apply affine transform or not
        sigma: used to control the spread of the peak.
    Output:
        conf_map: A numpy array of shape: (batch_size, im_width, im_height, num_joints)
    """
    
    num_images = len(indices)
    
    if affine_transform:
        conf_map = np.zeros((num_images, im_width_small, im_height_small, num_joints), np.float16)
        conf_mask = np.zeros((num_images, im_width_small, im_height_small, num_joints), np.float16)
    else:
        conf_map = np.zeros((num_images, im_width, im_height, num_joints), np.float16)
        conf_mask = np.zeros((num_images, im_width, im_height, num_joints), np.float16)
    
    # For image in all images
    for image_id in indices:
        
        heatmap_image = np.zeros((im_width, im_height, num_joints), np.float16)
        mask = np.zeros((im_width, im_height, num_joints), np.float16)
        
        # For a person in the image
        for person in range(len(all_keypoints[image_id])):
            # For all keypoints for the person.
            for part_num in range(len(all_keypoints[image_id][person])):
                # For keypoint in all_keypoints[image_id][person]:
                # Get the pixel values at a given keypoint across all 3 channels.
                # Note that our labels have images (im_width, im_height),
                # OpenCV has (im_height, im_width)
                x_index = all_keypoints[image_id][person][part_num][0]
                y_index = all_keypoints[image_id][person][part_num][1]
                visibility = all_keypoints[image_id][person][part_num][2]
                
#                print(f'part_num: {part_num}: {x_index}, {y_index}, {visibility}')
                
                # Generate heatmap only around the keypoint, leave others as 0
                if visibility != 0:
                    x_ind, y_ind = np.meshgrid(np.arange(im_width), np.arange(im_height))
                    numerator = (-(x_ind-x_index)**2) + (-(y_ind-y_index)**2)
                    heatmap_joint = np.exp(numerator/sigma).transpose()
                    heatmap_image[:, :, part_num] = np.maximum(heatmap_joint, heatmap_image[:, :, part_num])
                    
                    # Generate mask
                    mask[x_index, y_index, part_num] = 1

                    if affine_transform:
                        conf_map[image_id % num_images, :, :, :] = do_affine_transform(heatmap_image)       
#                        conf_mask[image_id % num_images, :, :, part_num] = do_affine_transform(mask)
                    else:
                        conf_map[image_id % num_images, :, :, :] = heatmap_image
#                        conf_mask[image_id % num_images, :, :, part_num] = mask
        if affine_transform:
            conf_mask[image_id % num_images] = do_affine_transform(mask)
        else:
            conf_mask[image_id % num_images] = mask
                        
    
    return conf_map #, conf_mask

def generate_paf(all_keypoints, indices, sigma=5, val=False, affine_transform=True):
    """
    Generate Part Affinity Fields given a batch of keypoints.
    Inputs:
        all_keypoints: Keypoints for all the images in the dataset. It is a 
        dictionary that contains image_id as keys and keypoints for each person.
        indices: The list of indices from the keypoints for which PAFs are to 
        be generated.
        sigma: The width of a limb in pixels.
        val(Default=False): True if validation set, False otherwise. 
        affine_transform: A bool to check whether to apply affine_transform or not
    Outputs:
        paf: A parts affinity fields map of shape: 
            (batch_size, im_width, im_height, 2, num_joints)
    """
    
    num_images = len(indices)
    
    if affine_transform:
        paf = np.zeros((num_images, im_width_small, im_height_small, 2, len(skeleton_limb_indices)), np.float16)
        paf_mask = paf = np.zeros((num_images, im_width_small, im_height_small, 2, len(skeleton_limb_indices)), np.float16)
    else:
        paf = np.zeros((num_images, im_width, im_height, 2, len(skeleton_limb_indices)), np.float16)
        paf_mask = paf = np.zeros((num_images, im_width, im_height, 2, len(skeleton_limb_indices)), np.float16)
    
    # For image in all_images
    for image_id in indices:
        # For each person in the image
        for person in range(len(all_keypoints[image_id])):
            # For each limb in the skeleton
            for limb in range(len(skeleton_limb_indices)):
                
                limb_indices = skeleton_limb_indices[limb]
                
                joint_one_index = limb_indices[0] - 1
                joint_two_index = limb_indices[1] - 1
                
                joint_one_loc = np.asarray(all_keypoints[image_id][person][joint_one_index])
                joint_two_loc = np.asarray(all_keypoints[image_id][person][joint_two_index])
                
                # If there is 0 for visibility, skip the limb
                if all_keypoints[image_id][person][joint_one_index][2] != 0 and all_keypoints[image_id][person][joint_two_index][2] != 0:
                    joint_one_loc = np.asarray(all_keypoints[image_id][person][joint_one_index][:2])
                    joint_two_loc = np.asarray(all_keypoints[image_id][person][joint_two_index][:2])

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
                        paf[image_id % num_images, :, :, 0, limb] += do_affine_transform(mask * vector[0])
                        paf[image_id % num_images, :, :, 1, limb] += do_affine_transform(mask * vector[1])
                        paf_mask[image_id % num_images, :, :, 0, limb] = do_affine_transform(mask)
                        paf_mask[image_id % num_images, :, :, 1, limb] = do_affine_transform(mask)
                    else:
                        paf[image_id % num_images, :, :, 0, limb] += mask * vector[0]
                        paf[image_id % num_images, :, :, 1, limb] += mask * vector[1]
                        paf_mask[image_id % num_images, :, :, 0, limb] = mask
                        paf_mask[image_id % num_images, :, :, 1, limb] = mask
#                    if affine_transform:
#                        paf[image_id % num_images, :, :, limb] += do_affine_transform(mask * vector[0]) + do_affine_transform(mask * vector[1])
#                    else:
#                        paf[image_id % num_images, :, :, limb] += (mask * vector[0]) + (mask * vector[1])
    return paf #, paf_mask