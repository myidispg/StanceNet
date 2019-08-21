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

from utilities.detect_poses import get_connected_joints, find_joint_peaks
from utilities.constants import threshold

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
    
print('Loading the pre-trained model')
model = StanceNet(18, 38).eval()
model.load_state_dict(torch.load('trained_models/trained_model.pth'))
model = model.to(device)
print(f'Loading the model complete.')

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

# Perform joint detection and limb detection, drawing and saving
orig_img = cv2.imread(image_path)
orig_img_shape = orig_img.shape
img = orig_img.copy()/255
img = cv2.resize(img, (400, 400))
# Convert to torch tensor
img = torch.from_numpy(img).view(1, img.shape[0], img.shape[1], img.shape[2]).permute(0, 3, 1, 2)

img = img.to(device).float()
# Forward pass through the network
paf, conf = model(img)
# Convert back to numpy array
paf = paf.cpu().detach().numpy()
conf = conf.cpu().detach().numpy()

# Remove the extra dimension of batch size
conf = np.squeeze(conf.transpose(2, 3, 1, 0))
paf = np.squeeze(paf.transpose(2, 3, 1, 0))

joints_list = find_joint_peaks(conf, orig_img_shape, threshold)

for joint_type in joints_list:
    for tuple_ in joint_type:
        x_index = tuple_[0]
        y_index = tuple_[1]
        cv2.circle(orig_img, (x_index, y_index), 3, (255, 0, 0))
        
# We need the PAF's upsampled to the to original image resolution.
paf_upsampled = cv2.resize(paf, (orig_img_shape[1], orig_img_shape[0]))

connected_limbs = get_connected_joints(paf_upsampled, joints_list)

# Visualize the limbs
for limb_type in connected_limbs:   
    for limb in limb_type:
        src, dest = limb[3], limb[4]
        cv2.line(orig_img, src, dest, (0, 255, 0), 2)
        
cv2.imwrite(output_path, orig_img)