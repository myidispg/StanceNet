# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 17:59:53 2019

@author: myidispg
"""

import pickle

import os
import cv2

import numpy as np

#img = cv2.imread(os.path.join(dataset_dir,'new_val2017/000000000000.jpg'))

from utilities.constants import dataset_dir
from utilities.helper import generate_confidence_maps, get_image_name, generate_paf

# Read the pickle files into dictionaries.
pickle_in = open(os.path.join(dataset_dir, 'keypoints_train_new.pickle'), 'rb')
keypoints_train = pickle.load(pickle_in)

pickle_in = open(os.path.join(dataset_dir, 'keypoints_val_new.pickle'), 'rb')
keypoints_val = pickle.load(pickle_in)

pickle_in.close()

def gen_data(all_keypoints, batch_size = 64, im_width = 224, im_height = 224, val=False, affine_transform=True):
    
    # Necessary imports
    from utilities.constants import dataset_dir
    from utilities.helper import generate_confidence_maps, get_image_name, generate_paf
    
    batch_count = len(all_keypoints.keys()) // batch_size
    
    count = 0
    
    # Loop over all keypoints in batches
    for batch in range(1, batch_count * batch_size + 1, batch_size):
        
        count += 1
        
        images = np.zeros((batch_size, im_width, im_height, 3), dtype=np.uint8)
        
        # Loop over all individual indices in a batch
        for image_id in range(batch, batch + batch_size):
            img_name = get_image_name(image_id-1)
            
            if val:
                img = cv2.imread(os.path.join(dataset_dir,'new_val2017',img_name))
            else:
                img = cv2.imread(os.path.join(dataset_dir,'new_train2017',img_name))
            
            images[image_id % batch] = img
        
        conf_maps, conf_mask = generate_confidence_maps(all_keypoints, range(batch-1, batch+batch_size-1), affine_transform=affine_transform)
        pafs, pafs_mask = generate_paf(all_keypoints, range(batch-1, batch+batch_size-1), affine_transform=affine_transform)
    
        yield count, images, conf_maps, conf_mask, pafs, pafs_mask
    
    # Handle cases where the total size is not a multiple of batch_size
    
    if len(all_keypoints.keys()) % batch_size != 0:
        
        start_index = batch_size * batch_count
        final_index = list(all_keypoints.keys())[-1]
        
#        print(final_index + 1 - start_index)
        
        images = np.zeros((final_index + 1 - start_index, im_width, im_height, 3), dtype=np.uint8)
        
        for image_id in range(start_index, final_index + 1):
            img_name = get_image_name(image_id)
            
            if val:
                img = cv2.imread(os.path.join(dataset_dir, 'new_val2017', img_name))
            else:
                img = cv2.imread(os.path.join(dataset_dir, 'new_train2017', img_name))
            
            images[image_id % batch_size] = img
            
        conf_maps, conf_mask = generate_confidence_maps(all_keypoints, range(start_index, final_index + 1), affine_transform=affine_transform)
        pafs, pafs_mask = generate_paf(all_keypoints, range(start_index, final_index + 1), affine_transform=affine_transform)
        
        yield count + 1, images, conf_maps, conf_mask, pafs, pafs_mask
        
for batch, images, conf_maps, conf_mask, pafs_mask,  pafs in gen_data(keypoints_val, 8, 224, 224, True):
    print(f'batch: {batch}\timages: {images.shape}\tconf_map: {conf_maps.shape}\tpafs: {pafs.shape}')
    break

total = 0
for i in range(17):
    total += np.sum(conf_maps[3, :, :, i])
    
import cv2
single_image = np.zeros((56, 56)) 
for i in range(17):
    single_image += conf_mask[1, :, :, i]
    
cv2.imshow('single', single_image)
cv2.waitKey()
cv2.destroyAllWindows()