# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 17:59:53 2019

@author: myidispg
"""

import pickle

import os
import cv2

import numpy as np

img = cv2.imread(os.path.join(dataset_dir,'new_val2017/000000000000.jpg'))

from constants import dataset_dir
from helper import generate_confidence_maps, get_image_name

# Read the pickle files into dictionaries.
pickle_in = open(os.path.join(dataset_dir, 'keypoints_train_new.pickle'), 'rb')
keypoints_train = pickle.load(pickle_in)

pickle_in = open(os.path.join(dataset_dir, 'keypoints_val_new.pickle'), 'rb')
keypoints_val = pickle.load(pickle_in)

pickle_in.close()

def gen_data(all_keypoints, batch_size = 64, im_width = 224, im_height = 224, val=False):
    
    index = 0
    
    # Loop over all keypoints in batches
    for batch in range(index, len(all_keypoints.keys()), batch_size):
        
        batch += 1
        
        images = np.zeros((batch_size, im_width, im_height, 3), dtype=np.uint8)
        
        # Loop over all individual indices in a batch
        for image_id in range(batch, batch + batch_size):
            
            img_name = get_image_name(image_id)
            
            if val:
                img = cv2.imread(os.path.join(dataset_dir,'new_val2017',img_name))
            else:
                img = cv2.imread(os.path.join(dataset_dir,'new_train2017',img_name))
            
            images[image_id % batch] = img
            
    
        yield images
        
        
for i in gen_data(keypoints_val, 64, 240, 240):
    print(i.shape)