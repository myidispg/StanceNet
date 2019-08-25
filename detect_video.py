# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 17:26:46 2019

@author: myidispg
"""
import argparse

import torch
import numpy as np
import cv2

import os

from models.paf_model_v2 import StanceNet
from pose_detect import PoseDetect

parser = argparse.ArgumentParser()
parser.add_argument('video_path', type=str, help='The path to the video file.')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

video_path = args.video_path
# If the path has backslashes like in windows, replace with forward slashes
video_path = video_path.replace('\\', '/')

if os.path.exists(video_path):
    pass
else:
    print('No such path or file exists. Please check.')
    exit()
    
detect = PoseDetect('trained_models/trained_model.pth')

# now, break the path into components
path_components = video_path.split('/')
video_name = path_components[-1].split('.')[0]
extension = path_components[-1].split('.')[1]

try:
    os.mkdir('processed_videos')
except FileExistsError:
    pass

output_path = os.path.join(os.getcwd(), 'processed_videos', f'{video_name}_keypoints.{extension}')
print(f'The processed video file will be saved in: {output_path}')

# Now that we have the video name, start the detection process.
vid = cv2.VideoCapture(video_path)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'H264')
out = cv2.VideoWriter(output_path, fourcc, 29.90,
                      (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                       int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))))

total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

print(f'Video file loaded and working on detecting joints.')
frames_processed = 0
while(vid.isOpened()):
    print(f'Processed {frames_processed} frames out of {total_frames}\n', end='\r')
    ret, orig_img = vid.read()
    if ret == False:
        break
    orig_img = detect.detect_poses(orig_img, use_gpu=True)
            
    out.write(orig_img)
    frames_processed += 1

vid.release()
out.release()
print(f'The video has been processed and saved at: {output_path}')
