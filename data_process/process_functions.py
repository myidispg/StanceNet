# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 09:42:17 2019

@author: myidispg
"""

import cv2
import numpy as np

from utilities.constants import dataset_dir, img_size, transform_scale, MEAN, STD
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

def normalize(img):
#     img = img[:, :, ::-1]
    img = (img - MEAN) / STD
#     img = img.transpose(2, 0, 1)
    return img

def adjust_keypoints(keypoints, original_shape):
  """
  Adjust keypoints according to the new image size.
  """
  # For a sublist in keypoints
  for list_ in range(len(keypoints)):
    for i in range(0, len(keypoints[list_]), 3):
      keypoints[list_][i] = (keypoints[list_][i]/original_shape[0]) * img_size
      keypoints[list_][i] = (keypoints[list_][i]/original_shape[1]) * img_size
  return keypoints

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

def generate_confidence_maps(keypoints, img_shape, affine_transform=True, sigma=7):
    """
    Generate confidence maps for a batch of indices.
    The generated confidence_maps are of shape: 
        (batch_size, im_width, im_height, num_joints)
    Input:
        keypoints: The list of keypoints labeled in the image. If multiple people,
            there can be sub-lists too.
        img_shape: The shape of the image. Used to create confidence map array.
        affine_transform: A bool to whether apply affine transform or not
        sigma: used to control the spread of the peak.
    Output:
        conf_map: A numpy array of shape: (batch_size, im_width, im_height, num_joints)
    """
    if affine_transform:
        conf_map = np.zeros((img_size // 4,
                             img_size // 4, num_joints), np.float32)
    else:
        conf_map = np.zeros((img_size, img_size, num_joints), np.float32)
    
    # For sub list in keypoints:
    for list_ in keypoints:
        part_num = 0
        for i in range(0, len(list_), 3):
            x_index = list_[i]
            y_index = list_[i+1]
            visibility = list_[i+2]
            
            if visibility != 0:
                x_ind, y_ind = np.meshgrid(np.arange(img_size), 
                                          np.arange(img_size))
                numerator = (-(x_ind-x_index)**2) + (-(y_ind-y_index)**2)
                heatmap_joint = np.exp(numerator/sigma).transpose()
                if affine_transform:
                    heatmap_joint = cv2.resize(heatmap_joint, (img_size // 4, img_size // 4))
#                    heatmap_joint = do_affine_transform(heatmap_joint)
                conf_map[:, :, part_num] = np.maximum(heatmap_joint, conf_map[:, :, part_num])
            
            part_num += 1
                    
    return conf_map
    

def generate_paf(keypoints, img_shape, sigma=5, affine_transform=True):
    """
    Generate Part Affinity Fields given a batch of keypoints.
    Inputs:
         keypoints: The list of keypoints labeled in the image. If multiple people,
            there can be sub-lists too.
        img_shape: The shape of the image. Used to create confidence map array.
        sigma: The width of a limb in pixels.
        affine_transform: A bool to check whether to apply affine_transform or not
    Outputs:
        paf: A parts affinity fields map of shape: 
            (batch_size, im_width, im_height, 2, num_joints)
    """
    
    if affine_transform:
        paf = np.zeros((img_size // 4, img_size // 4, 2,
                        len(skeleton_limb_indices)), np.float32)
    else:
        paf = np.zeros((img_size, img_size, 2,
                        len(skeleton_limb_indices), np.float32))
    
    # For sub list in keypoints
    for list_ in keypoints:
        for limb in range(len(skeleton_limb_indices)):
            limb_indices = skeleton_limb_indices[limb]
            
            # Convert part num given in skeleton_limb_indices to index in keypoints
            joint_one_index = (limb_indices[0]-1) * 3
            joint_two_index = (limb_indices[1]-1) * 3
            
            
            # If there is 0 for visibility, skip the limb
            if list_[joint_one_index + 2] != 0 and list_[joint_two_index + 2] != 0:
                joint_one_loc = np.asarray((list_[joint_one_index], list_[joint_one_index+1]))
                joint_two_loc = np.asarray((list_[joint_two_index], list_[joint_two_index+1]))
                
                part_line_segment = joint_two_loc - joint_one_loc
                norm = np.linalg.norm(part_line_segment)
                
                norm = norm + 1e-8 if norm == 0 else norm # To make sure it is not equal to zero.
                
                vector = part_line_segment / norm
                # https://gamedev.stackexchange.com/questions/70075/how-can-i-find-the-perpendicular-to-a-2d-vector
                perpendicular_vec = [-vector[1], vector[0]]
                x, y = np.meshgrid(np.arange(img_size), np.arange(img_size))
                 # Formula according to the paper
                along_limb = vector[0] * (x-joint_one_loc[0]) + vector[1] * (y-joint_one_loc[1])
                across_limb = np.abs(perpendicular_vec[0] * (x-joint_one_loc[0]) + perpendicular_vec[1] * (y - joint_one_loc[1])) 
                
                # Check the given conditions
                cond_1 = along_limb >= 0
                cond_2 = along_limb <= norm
                cond_3 = across_limb <= sigma
                mask = (cond_1 & cond_2 & cond_3).astype(np.float32).transpose()
                
                
                # put the values
                if affine_transform:
                    mask = cv2.resize(mask, (img_size // 4, img_size // 4))
#                    mask = do_affine_transform(mask)
                    paf[:, :, 0, limb] += mask * vector[0]
                    paf[:, :, 1, limb] += mask * vector[1]
                else:
                    paf[:, :, 0, limb] += mask * vector[0]
                    paf[:, :, 1, limb] += mask * vector[1]
    return paf
                
                
                