# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 18:48:23 2019

@author: myidispg
"""

import torch

import numpy as np
import cv2
import os

from scipy.ndimage.filters import gaussian_filter, maximum_filter
from scipy.ndimage.morphology import generate_binary_structure

#from models.full_model import OpenPoseModel
from models.paf_model_v2 import StanceNet
from utilities.constants import MEAN, STD, threshold

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

#model = OpenPoseModel(15, 17).to(device).eval()
model = StanceNet(18, 38,).eval()
model.load_state_dict(torch.load('trained_model.pth'))
model = model.to(device)

#img = cv2.imread(os.path.join('Coco_Dataset', 'val2017', '000000008532.jpg'))
img = cv2.imread('market.jpg')/255
#img = cv2.imread(os.path.join('Coco_Dataset', 'train2017', '000000000036.jpg'))
#img = ((img/255)-MEAN)/STD
img = cv2.resize(img, (400, 400))
cv2.imshow('image', img)
cv2.waitKey()
cv2.destroyAllWindows()

img = torch.from_numpy(img).view(1, img.shape[0], img.shape[1], img.shape[2]).permute(0, 3, 1, 2)

#img = img.float()
img = img.to(device).float()

paf, conf = model(img)

paf = paf.cpu().detach().numpy()
conf = conf.cpu().detach().numpy()

# Remove the extra dimension of batch size
conf = np.squeeze(conf.transpose(2, 3, 1, 0))
paf = np.squeeze(paf.transpose(2, 3, 1, 0))

paf = paf.reshape(paf.shape[0], paf.shape[1], -1, 2)

#for i in range(conf.shape[2]):
#    conf_map = cv2.resize(conf[:, :, i], (400, 400))
#    print(i)
#    cv2.imshow('conf', conf_map)
#    cv2.waitKey()
#    cv2.destroyAllWindows()
#
## Visualize Confidence map
#conf_map = np.zeros((conf.shape[0], conf.shape[1]))
#for i in range(conf.shape[2]):
#    conf_map += conf[:, :, i]
#
#conf_map = cv2.resize(conf_map, (400, 400))
#
#conf_map = (conf_map > 1.3).astype(np.float32)
#
#cv2.imshow('conf map', conf_map)
#cv2.waitKey()
#cv2.destroyAllWindows()

# Visualize Parts Affinity Fields

#for i in range(19):
#    paf_map = np.zeros((paf.shape[0], paf.shape[1]))
#    for j in range(2):
#        paf_map += paf[:, :, i, j] + paf[:, :, i, j]
#    
#    paf_map = cv2.resize(paf_map, (400, 400))
#    
#    cv2.imshow('paf map', paf_map)
#    cv2.waitKey()
#    cv2.destroyAllWindows()
#
#paf_map = np.zeros((paf.shape[0], paf.shape[1]))
#for i in range(paf.shape[3]):
#    paf_map += paf[:, :, 0, i] + paf[:, :, 1, i]
#
#paf_map = cv2.resize(paf_map, (400, 400))
#
#paf_map = (paf_map > 0.3).astype(np.float32)
#
#cv2.imshow('paf map', paf_map*255)
#cv2.waitKey()
#cv2.destroyAllWindows()
#
#test = paf[:, :, 12, 0]
#cv2.imshow('paf', test*255)
#cv2.waitKey()
#cv2.destroyAllWindows()

#---------This section is to find the peaks in the confidence maps
def find_resized_coordinates(coords, resizeFactor):
    """
    Computes resized coordinated to upscale the coordinates to the size of images.
    Inputs:
        coords: A tuple of x and y
        resizeFactor: the factor by which to resize. 
        It is a tuple for different axises. (x_scale, y_scale)
    Outputs:
        returns a tuple of the resized coordinates. (X, Y)
    """
    x, y = coords[0] + 0.5, coords[1] + 0.5
    x, y = x * resizeFactor[0], y * resizeFactor[1]
#    x, y = coords[0] * resizeFactor[0], coords[1] * resizeFactor[1]
    
    return (x - 0.5, y - 0.5)
#    return (x, y)
    
def find_joint_peaks(heatmap, upsampleScale, threshold):
    """
    Given a heatmap, find the peaks of the detected joints with 
    confidence score > threshold.
    Inputs:
        heatmap: The heatmaps for all the joints. It is of shape: h, w, num_joints.
        upsampleScale: The scale by which to increase the coordinates. 
            The heatmap is of low size and the image is of larger size.
            Again a tuple for both the axises.
        threshold: The minimum score for each joint
    Output:
        A list of the detected joints. There is a sublist for each joint 
        type (18 in total) and each sublist contains a tuple with 4 values:
        (x, y, confidence score, unique_id). The number of tuples is equal
        to the number of joints detected of that type. For example, if there
        are 3 nose detected (joint type 0), then there will be 3 tuples in the 
        nose sublist            
    """
    
    win_size = 2 # the window size used to smoothen the peaks.
    
    joints_list = list()
    counter = 0 # This will serve as a unique id for all the detected joints.
    for i in range(heatmap.shape[2]-1):
        sub_list = list()
        joint_map = heatmap[:, :, i] # the heatmap for a particular joint type.
        # Threshold the joint_map
        joint_map = np.where(joint_map > threshold, joint_map, 0)
        # Apply a convolution kind of operation and replace each pixel in a 
        # 5x5 window with the maximum of the pixels in that window.
        filter_ = maximum_filter(joint_map, footprint=np.ones((3, 3)))
        # Now, we compare the original joint_map with the convolved filter_
        # This will give us all the locations where the peaks remained at same location.
        mask = filter_ == joint_map
        # Apply the mask so that only peaks remain. All other are 0.
        joint_map *= mask
        # Get the x and y of all the peaks
        x_indices, y_indices = np.where(joint_map != 0)
        # Now, we can create tuples and append to the sublist
        for j in range(len(x_indices)):
            x_index, y_index = x_indices[j], y_indices[j]
            confidence = joint_map[x_index, y_index]
            
            # Refine the center of the heatmap.
            # Taken from here:
            #https://github.com/NiteshBharadwaj/part-affinity/blob/0f49c6804f75b2153e0fa3c9ae07d73c200d4717/src/evaluation/post.py#L115
#            x_min, y_min = np.maximum(0, x_index - win_size), np.maximum(0, y_index - win_size)
#            x_max, y_max = np.maximum(conf.shape[0] - 1, x_index + win_size), np.maximum(conf.shape[1] - 1, y_index + win_size)
#            
#            # Take a small path around the peak and only upsample that region.
#            patch = joint_map[y_min:y_max + 1, x_min:x_max + 1]
#            map_upsamp = cv2.resize(
#                    patch, None, fx=upsampleScale[0], fy=upsampleScale[1], interpolation=cv2.INTER_CUBIC)
#            
#            location_of_max = np.unravel_index(
#                    map_upsamp.argmax(), map_upsamp.shape)
#            print(f'location of max: {location_of_max}')
#            print(f'xindex, yindex: {(x_index, y_index)}')
#            
#            location_of_patch_center = find_resized_coordinates((x_index-x_min, y_index-y_min), upsampleScale)
#            print(f'loc patch center: {location_of_patch_center}')
#            
#            refined_center = (location_of_max[0] - location_of_patch_center[0], 
#                              location_of_max[1] - location_of_patch_center[1])
#            print(refined_center)
#            confidence = map_upsamp[location_of_max]
#            print(f'confidence: {confidence}')
#            
#            x_index, y_index = find_resized_coordinates((x_index, y_index), upsampleScale)
#            x_index, y_index = x_index + refined_center[0], y_index + refined_center[1]
#            sub_list.append((x_index, y_index, confidence, counter))
#           Compute the resize factor for both the axises.
#            x_resize, y_resize = orig_shape[0] / conf.shape[0], orig_shape[1] / conf.shape[1]
            x_index, y_index = find_resized_coordinates((x_index, y_index), 
                                                        upsampleScale)
#            x_index, y_index = x_index, y_index
            sub_list.append((x_index, y_index, confidence, counter))
            counter += 1
            
        joints_list.append(sub_list)

    return joints_list    

joints_list = find_joint_peaks(conf, (8, 8), threshold)
    
#test = np.where(conf[:, :, 0] > 0.09, conf[:, :, 0], 0)
#
#filter_ = maximum_filter(test, footprint=np.ones((3, 3)))
#mask = filter_ == test
#test *= mask
## 14, 14 and 9, 31
#x_index, y_index = np.where(test != 0)
#
# Given the found joints_list, draw on the images
img = np.squeeze(img.cpu().detach().numpy())
img = img.transpose(1, 2, 0)
img = img.copy()
#orig_img = cv2.resize(orig_img, (400, 400))
for joint_type in joints_list:
    for tuple_ in joint_type:
        x_index = int(tuple_[0])
        y_index = int(tuple_[1])
        print(x_index, y_index)
        cv2.circle(img, (y_index, x_index), 3, (255, 0, 0))
        
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
