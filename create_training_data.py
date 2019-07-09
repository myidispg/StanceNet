# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 14:41:02 2019

@author: myidispg
"""

import os
import json

from helper import draw_skeleton, group_keypoints, get_image_name
from constants import dataset_dir, skeleton_limb_indices

with open(os.path.join(dataset_dir,
                       'annotations', 'person_keypoints_val2017.json'), 'r') as JSON:
    val_dict = json.load(JSON)

with open(os.path.join(dataset_dir,
                       'annotations', 'person_keypoints_train2017.json'), 'r') as JSON:
    train_dict = json.load(JSON)

print(f'The length of train annotations is: {len(train_dict["annotations"])}')
print(f'The length of validation annotations is: {len(val_dict["annotations"])}')

"""
The training data is going to be converted into a single dictionary.
The dictionary will have the image id as the key and a list of keypoints 
for each labelled person in the image.
The list of keypoints contains 17 tuples with each tuple of format: (x, y, v)
"""

keypoints_val = dict()

for annotation in val_dict['annotations']:
    if annotation['num_keypoints'] != 0:
        if annotation['image_id'] in keypoints_val.keys():
            keypoints_val[annotation['image_id']].append(
                    group_keypoints(annotation['keypoints']))
        else:
            keypoints_val[annotation['image_id']] = [group_keypoints(
                    annotation['keypoints'])]
            

keypoints_train = dict()

for annotation in train_dict['annotations']:
    if annotation['num_keypoints'] != 0:
        if annotation['image_id'] in keypoints_train.keys():
            keypoints_train[annotation['image_id']].append(
                    group_keypoints(annotation['keypoints']))
        else:
            keypoints_train[annotation['image_id']] = [group_keypoints(
                    annotation['keypoints'])]

draw_skeleton(209468, keypoints_train[209468], skeleton_limb_indices, val=False)

# Now, I am going to remove all the images from the test and validation directory
# that are not labelled with people in the images.

def get_image_id_from_filename(filename):
    """
    Get the image_id from a filename with .jpg extension
    """    
    return int(filename.split('.')[0])
   
# Loop over all the images in val folder and remove image if not in keypoints_val 
val_images = os.listdir(os.path.join(dataset_dir, 'val2017'))

list_ = list()
for image in val_images:
    image_id = get_image_id_from_filename(image)
    if image_id not in keypoints_val.keys():
        list_.append(image)
        image_path = os.path.join(dataset_dir, 'val2017', image)
        if os.path.exists(image_path):
            os.remove(image_path)
        
print(f'Removed {len(list_)} images from validation folder.')

# Loop over all the images in val folder and remove image if not in keypoints_val 
train_images = os.listdir(os.path.join(dataset_dir, 'train2017'))

list_ = list()
for image in train_images:
    image_id = get_image_id_from_filename(image)
    if image_id not in keypoints_train.keys():
        list_.append(image)
        image_path = os.path.join(dataset_dir, 'train2017', image)
        if os.path.exists(image_path):
            os.remove(image_path)
        
print(f'Removed {len(list_)} images from train_folder.')



