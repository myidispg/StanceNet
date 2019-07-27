# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 14:28:39 2019

@author: myidispg
"""

from torch.utils.data import Dataset

import os
import cv2

import utilities.constants as constants
from utilities.helper import get_image_name, generate_confidence_maps, generate_paf

class StanceNetDataset(Dataset):
    """
    Custom Dataset to get objects in batches while training. 
    Must be used along with Dataloaders.
    """
    
    def __init__(self, keypoints, img_dir):
        """
        Args:
            keypoints: All keypoints dictionary. Used to generate confidence 
                maps and PAFs
            img_dir: The path to the images directory. Used to differentiate
                between train and validation images.
        """
        self.keypoints = keypoints
        self.img_dir = img_dir
    
    def __len__(self):
        return len(self.keypoints.keys())
    
    def __getitem__(self, idx):
        """
        Given the id of an image, return the image and the corresponding 
        confidence maps and the parts affinity fields.
        """
        img_name = os.path.join(self.img_dir, get_image_name(idx))
        print(img_name)
        img = cv2.imread(img_name)
        
        conf_maps = generate_confidence_maps(self.keypoints, idx)
        paf = generate_paf(self.keypoints, idx)
        
        return img, conf_maps, paf
        
        