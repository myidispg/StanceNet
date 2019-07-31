# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 14:28:39 2019

@author: myidispg
"""

from torch.utils.data import Dataset

import numpy as np
import os
import cv2

import utilities.constants as constants
from data_process.process_functions import generate_confidence_maps
from data_process.process_functions import generate_paf, do_affine_transform
from utilities.helper import get_image_name

MEAN = [0.485, 0.456, 0.406],
STD = [0.229, 0.224, 0.225]

img_size= 400

def normalize(img):
#     img = img[:, :, ::-1]
    img = (img - MEAN) / STD
#     img = img.transpose(2, 0, 1)
    return img

def adjust_keypoints(keypoints, original_shape):
    # For a sublist in keypoints
    for list_ in range(len(keypoints)):
      for i in range(0, len(keypoints[list_]), 3):
        keypoints[list_][i] = (keypoints[list_][i]/original_shape[0]) * img_size
        keypoints[list_][i] = (keypoints[list_][i]/original_shape[1]) * img_size
    return keypoints

class StanceNetDataset(Dataset):
    """
    Custom Dataset to get objects in batches while training. 
    Must be used along with Dataloaders.
    """
    
    def __init__(self, coco, img_dir):
        """
        Args:
            coco: A COCO object for the annotations. Used for mask generation.
            img_dir: The path to the images directory. Used to differentiate
                between train and validation images.
        """
        self.img_dir = img_dir
        self.coco = coco
        
        # We have a COCO data object for a person. Get the category id for a person.
        person_ids = self.coco.getCatIds(catNms=['person'])
        # Get the ids of all images with a person in it.
        self.img_indices = sorted(self.coco.getImgIds(catIds=person_ids))
    
    def __len__(self):
        # The length will be equal to count of those images that have a person.
        return len(self.img_indices)
#        return len(self.keypoints.keys())
    
    def __getitem__(self, idx):
        """
        Given the id of an image, return the image and the corresponding 
        confidence maps and the parts affinity fields.
        """
        # Get a specific image id from the list of image ids.
        img_index = self.img_indices[idx]
        
        # Load the image
        img_name = os.path.join(self.img_dir, get_image_name(img_index))
        print(img_name)
        img = cv2.imread(img_name).transpose(1, 0, 2)
        original_shape = img.shape[:2]
        # Resize image to 400x400 dimensions.
        img = cv2.resize(normalize(img/255), (img_size, img_size))
#        print(f'{img_name}\n{img.shape}')
        # Get the annotation id of the annotaions about the image.
        annotations_indices = self.coco.getAnnIds(img_index)
        # Load the annotations from the annotaion ids.
        annotations = self.coco.loadAnns(annotations_indices) 
        keypoints = []
        mask = np.zeros((img.shape[:2]), np.uint8)
        for annotation in annotations:
            if annotation['num_keypoints'] != 0:
                keypoints.append(annotation['keypoints'])
            mask = mask | cv2.resize(self.coco.annToMask(annotation), (img_size, img_size))
            
        # Adjust keypoints according to resized images.
        keypoints = adjust_keypoints(keypoints, original_shape)
        
        conf_maps = generate_confidence_maps(keypoints, img.shape[:2])
        paf = generate_paf(keypoints, img.shape[:2])
        
        return img, conf_maps, paf, do_affine_transform(mask)
        
        