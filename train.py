# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 18:33:28 2019

@author: myidispg
"""

import os
import torch

import numpy as np

from pycocotools.coco import COCO

from models.full_model import OpenPoseModel

import utilities.constants as constants
import utilities.helper as helper
from training_utilities.train_utils import train_epoch, train
from training_utilities.stancenet_dataset import StanceNetDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Loading training COCO Annotations used for mask generation. Might take time.')
#coco_train = COCO(os.path.join(os.path.join(os.getcwd(), 'Coco_Dataset'),
#                       'annotations', 'person_keypoints_train2017.json'))
coco_valid = COCO(os.path.join(os.path.join(os.getcwd(), 'Coco_Dataset'),
                       'annotations', 'person_keypoints_val2017.json'))

#ann_ids = coco_train.getAnnIds()
#anns = coco_train.loadAnns(ann_ids)
print('Annotation load complete.')

#Collate function to manage variable input sizes
def collate_fn(batch):
    # There is a list of batches.
    # Then each batch is a tuple.
    output = []
    for data in batch:
        batch_data = list()
        for item in data:
            print(type(torch.from_numpy(item)))
            batch_data.append(item)
        output.append(tuple(batch_data))
    return output
        
#train_data = StanceNetDataset(coco_train, os.path.join(constants.dataset_dir, 'train2017'))
valid_data = StanceNetDataset(coco_valid, 
                              os.path.join(constants.dataset_dir, 'val2017'))
#
#train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=1,
#                                               shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=1,
                                               shuffle=True)

#for img, conf, paf, mask in train_dataloader:
#    break
#
#conf = conf.numpy()
#paf = paf.numpy()
#mask = mask.numpy().astype(np.uint8)

status = train(valid_dataloader, device, num_epochs=5, val_every=False,
               print_every=50, resume=False)
if status == None:
    print('There was some issue in the training process. Please check.')


for batch, (img, conf_map, paf, mask) in enumerate(valid_dataloader):
    img = ((img.numpy() * constants.STD) + constants.MEAN) * 255
    cv2.imshow('image', img)
    cv2.waitKey()
    cv2.destroyAllWindows() 
    cv2.imshow('image', conf_map.numpy())
    cv2.waitKey()
    cv2.destroyAllWindows()
    cv2.imshow('image', paf.numpy())
    cv2.waitKey()
    cv2.destroyAllWindows()
    cv2.imshow('image', mask.numpy())
    cv2.waitKey()
    cv2.destroyAllWindows()
#    print(type(data[0][0]))
    break

import cv2

cv2.imshow('image', img)
cv2.waitKey()
cv2.destroyAllWindows()

# Visualize predicted conf_map in grayscale.
def process_conf_map(conf_map):
    processed_map = np.zeros((conf_map.shape[1], conf_map.shape[2]))
    for i in range(17):
        processed_map += conf_map[0, :, :, i]
    return processed_map

processed_map = process_conf_map(conf)

cv2.imshow('Confidence map', processed_map*255)
cv2.waitKey()
cv2.destroyAllWindows()

# Visualize predicted paf in grayscale
def process_paf(paf):
    processed = np.zeros((paf.shape[1:3]))
    for i in range(15):
        processed += paf[0, :, :, 0, i] + paf[0, :, :, 1, i]
    return processed

processed_paf = process_paf(paf)

cv2.imshow('Parts Affinity Fields', processed_paf)
cv2.waitKey()
cv2.destroyAllWindows()

cv2.imshow('mask', mask*255)
cv2.waitKey()
cv2.destroyAllWindows()

import matplotlib.pyplot as plt
plt.plot(list(range(len(losses))), losses)
plt.xlabel('Epcohs')
plt.ylabel('Loss')

# Test an image
import cv2

img = cv2.imread(os.path.join(constants.dataset_dir, 'new_val2017', '000000000001.jpg'))
cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()

img = img.reshape(1, -1, 224, 224)
outputs = model(torch.from_numpy(img).float().to(device))
conf = outputs[3]['conf'].cpu().detach().numpy().reshape(56, 56, -1)

def process_output_conf_map(image, scale_factor=4):
    """
    Returns the heatmap generated by model output is a visualizable form.
    Inputs:
        image: The heatmap generated by the model. Must be of shape:
            (batch, num_joints, im_width, im_height)
        scale_factor: The factor by which to enlarge the heatmap. Default=4
    """
    
    from utilities.helper import do_affine_transform
    
    print(image.shape)
    
    conf = np.zeros((image.shape[2], image.shape[3]))
    image = image.reshape(image.shape[2], image.shape[3], 17)
    for i in range(17):
        conf += image[:, :, i]
        
    conf = do_affine_transform(image, scale_factor)
    return conf

def visualize_output_conf_map(conf_map):
    """
    Visualizes a conf map using OpenCV.
    Inputs:
        conf_map: The conf_map to be visualized. Needs to be of the shape:
            (1, num_joints, width, height)
    """
    conf_map = process_output_conf_map(conf_map)
    cv2.imshow('COnfidence Map',conf_map)
    cv2.waitKey()
    cv2.destroyAllWindows()

visualize_output_conf_map(conf)
conf = process_output_conf_map(outputs[3]['conf'].cpu().detach().numpy())

disp = np.zeros((56, 56))
for i in range(17):
    disp += conf[:, :, i]

disp = helper.do_affine_transform(disp, 10)
    
cv2.imshow('disp', conf)
cv2.waitKey()
cv2.destroyAllWindows()