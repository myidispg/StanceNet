# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 18:51:51 2019

@author: myidispg
"""

import argparse

import torch
import numpy as np
import cv2

import os

from utilities.constants import threshold

from pose_detect import PoseDetect
from models.paf_model_v2 import StanceNet

parser = argparse.ArgumentParser()
parser.add_argument('image_path', type=str, help='The path to the image file.')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

image_path = args.image_path
image_path = image_path.replace('\\', '/')

if os.path.exists(image_path):
    pass
else:
    print('No such path exists. Please check')
    exit()
    
detect = PoseDetect('trained_models/trained_model.pth')

# now, break the path into components
path_components = image_path.split('/')
image_name = path_components[-1].split('.')[0]
extension = path_components[-1].split('.')[1]

try:
    os.mkdir('processed_images')
except FileExistsError:
    pass

output_path = os.path.join(os.getcwd(), 'processed_images', f'{image_name}_keypoints.{extension}')
print(f'The processed image file will be saved in: {output_path}')

# Read the original image
orig_img = cv2.imread(image_path)

# Perform pose detection on the given image
orig_img = detect.detect_poses(orig_img, use_gpu=True)

cv2.imwrite(output_path, orig_img)