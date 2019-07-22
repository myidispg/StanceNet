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

def gen_data(all_keypoints, batch_size = 64, im_width = 224, im_height = 224, val=False):
    
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
        
        conf_maps = generate_confidence_maps(all_keypoints, range(batch-1, batch+batch_size-1))
        pafs = generate_paf(all_keypoints, range(batch-1, batch+batch_size-1))
    
        yield count, images, conf_maps, pafs
    
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
            
        conf_maps = generate_confidence_maps(all_keypoints, range(start_index, final_index + 1))
        pafs = generate_paf(all_keypoints, range(start_index, final_index + 1))
        
        yield count + 1, images, conf_maps, pafs
        
for batch, images, conf_maps, pafs in gen_data(keypoints_val, 64, 224, 224, True):
    print(f'batch: {batch}\timages: {images.shape}\tconf_map: {conf_maps.shape}\tpafs: {pafs.shape}')
    break