# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 19:05:53 2019

@author: myidi
"""

import os
import pickle
import cv2

from constants import dataset_dir, im_width, im_height, skeleton_limb_indices
from helper import get_image_name, draw_skeleton

pickle_in = open(os.path.join(dataset_dir, 'keypoints_val.pickle'), 'rb')
keypoints_val = pickle.load(pickle_in)

pickle_in = open(os.path.join(dataset_dir, 'keypoints_train.pickle'), 'rb')
keypoints_train = pickle.load(pickle_in)

pickle_in.close()

# Resize the images to 240x240 and maintain the keypoint information.
def resize_image_keypoint(all_keypoints, im_width, im_height, val=False):
    
    new_keypoints = dict()

    for img_id in all_keypoints.keys():
        new_keypoints[img_id] = []
        img_name = get_image_name(img_id)
        if val:
            img = cv2.imread(os.path.join(dataset_dir, 'val2017', img_name))
        else:
            img = cv2.imread(os.path.join(dataset_dir, 'train2017', img_name))
        
        old_height, old_width, _ = img.shape
        
        all_person_keypoints = []
        
        # Iterate over the all the people in the image
        for person in range(len(all_keypoints[img_id])):
            person_keypoints = []
           # Iterate over all keypoints
            for keypoint in all_keypoints[img_id][person]:
                width_ratio = keypoint[0] / old_width
                height_ratio = keypoint[1] / old_height
                
                new_x = int(width_ratio * im_width)
                new_y = int(height_ratio * im_height)
                
                new_keypoint = (new_x, new_y, keypoint[2])
                
                person_keypoints.append(new_keypoint)
            all_person_keypoints.append(person_keypoints)
        new_keypoints[img_id] = all_person_keypoints
        img = cv2.resize(img, (im_height, im_width))
        if val:
            print(os.path.join(dataset_dir, 'new_val2017', img_name))
            cv2.imwrite(os.path.join(dataset_dir, 'new_val2017', img_name), img)
        else:
            print(os.path.join(dataset_dir, 'new_train2017', img_name))
            cv2.imwrite(os.path.join(dataset_dir, 'new_train2017', img_name), img)
        
    
    return new_keypoints
    
new_keypoints_val = resize_image_keypoint(keypoints_val, im_width, im_height, True)
new_keypoints_train = resize_image_keypoint(keypoints_train, im_width, im_height)

#draw_skeleton(5, new_keypoints_train[5], skeleton_limb_indices, val=False)

# Save the keypoints to a pickle file
import pickle 
pickle_out = open('Coco_Dataset/keypoints_train_new.pickle', 'wb')
pickle.dump(new_keypoints_train, pickle_out)
pickle_out.close()

pickle_out = open('Coco_Dataset/keypoints_val_new.pickle', 'wb')
pickle.dump(new_keypoints_val, pickle_out)
pickle_out.close()
