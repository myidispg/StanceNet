# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 14:41:02 2019

@author: myidispg
"""

import os
import json

from utilities.helper import get_image_name, get_image_id_from_filename
from utilities.constants import dataset_dir, skeleton_limb_indices
from data_process.process_functions import group_keypoints
from visualization.visualization_functions import draw_skeleton


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

#draw_skeleton(69213, keypoints_val[69213], skeleton_limb_indices,
#              val=True, wait_time=None)

# Now, I am going to remove all the images from the test and validation directory
# that are not labelled with people in the images.
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

# As a final check, see if all the remaining images have keypoints for them.
val_images = os.listdir(os.path.join(dataset_dir, 'val2017'))

for image in val_images:
    image_id = get_image_id_from_filename(image)
    if image_id in keypoints_val.keys():
        pass
    else:
        print(f'Problem with {image_id}')
    
print('There seems to be no issues with the validation set and labels.')

train_images = os.listdir(os.path.join(dataset_dir, 'train2017'))
for image in train_images:
    image_id = get_image_id_from_filename(image)
    if image_id in keypoints_train.keys():
        pass
    else:
        print(f'Problem with {image_id}')
    
print('There seems to be no issues with the train set and labels.')

## Finally, change image names so that the image ids start with 0.
#new_keypoints_val = dict()
#ids_val_list = list(keypoints_val.keys())
#ids_val_list.sort()
#
#count = 0
#for img_id in ids_val_list:
#    new_keypoints_val[count] = keypoints_val[img_id]
#    os.rename(os.path.join(dataset_dir, 'val2017', get_image_name(img_id)),
#              os.path.join(dataset_dir, 'val2017', get_image_name(count)))
#    count += 1
#    
#print(f'Renamed {count} files.')
#
#new_keypoints_train = dict()
#ids_train_list = list(keypoints_train.keys())
#ids_train_list.sort()
#
#count = 0
#for img_id in ids_train_list:
#    new_keypoints_train[count] = keypoints_train[img_id]
#    os.rename(os.path.join(dataset_dir, 'train2017', get_image_name(img_id)),
#              os.path.join(dataset_dir, 'train2017', get_image_name(count)))
#    count += 1
#    
#print(f'Renamed {count} train files.')
#    
#del image, annotation, count, dataset_dir, ids_train_list, ids_val_list, image_id, img_id, keypoints_train, keypoints_val, list_, skeleton_limb_indices, train_dict, train_images, val_dict, val_images

# Save the keypoints to a pickle file
import pickle 
pickle_out = open('Coco_Dataset/keypoints_train.pickle', 'wb')
pickle.dump(keypoints_train, pickle_out)
pickle_out.close()

pickle_out = open('Coco_Dataset/keypoints_val.pickle', 'wb')
pickle.dump(keypoints_val, pickle_out)
pickle_out.close()


## Save the keypoints to a pickle file
#import pickle 
#pickle_out = open('Coco_Dataset/keypoints_train.pickle', 'wb')
#pickle.dump(new_keypoints_train, pickle_out)
#pickle_out.close()
#
#pickle_out = open('Coco_Dataset/keypoints_val.pickle', 'wb')
#pickle.dump(new_keypoints_val, pickle_out)
#pickle_out.close()
