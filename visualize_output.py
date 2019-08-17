# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 18:48:23 2019

@author: myidispg
"""

import torch

import numpy as np
import cv2

from scipy.ndimage.filters import gaussian_filter, maximum_filter
from scipy.ndimage.morphology import generate_binary_structure

#from models.full_model import OpenPoseModel
from models.paf_model_v2 import StanceNet
from utilities.constants import threshold, BODY_PARTS

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

#model = OpenPoseModel(15, 17).to(device).eval()
model = StanceNet(18, 38,).eval()
model.load_state_dict(torch.load('trained_model.pth'))
model = model.to(device)

#img = cv2.imread(os.path.join('Coco_Dataset', 'val2017', '000000008532.jpg'))
orig_img = cv2.imread('many_people.jpeg')
orig_img_shape = orig_img.shape
img = orig_img.copy()/255
img = cv2.resize(img, (400, 400))
cv2.imshow('image', orig_img)
cv2.waitKey()
cv2.destroyAllWindows()

img = torch.from_numpy(img).view(1, img.shape[0], img.shape[1], img.shape[2]).permute(0, 3, 1, 2)

#img = img.float()
img = img.to(device).float()

paf, conf = model(img)

paf = paf.cpu().detach().numpy()
conf = conf.cpu().detach().numpy()

# Remove the extra dimension of batch size
conf = np.squeeze(conf.transpose(2, 3, 1, 0))
paf = np.squeeze(paf.transpose(2, 3, 1, 0))

paf = paf.reshape(paf.shape[0], paf.shape[1], 2, -1)

#for i in range(conf.shape[2]):
#    conf_map = cv2.resize(conf[:, :, i], (400, 400))
#    print(i)
#    cv2.imshow('conf', conf_map)
#    cv2.waitKey()
#    cv2.destroyAllWindows()
#
## Visualize Confidence map
#conf_map = np.zeros((conf.shape[0], conf.shape[1]))
#for i in range(conf.shape[2]):
#    conf_map += conf[:, :, i]
#
#conf_map = cv2.resize(conf_map, (400, 400))
#
#conf_map = (conf_map > 1.3).astype(np.float32)
#
#cv2.imshow('conf map', conf_map)
#cv2.waitKey()
#cv2.destroyAllWindows()

# Visualize Parts Affinity Fields

#for i in range(19):
#    paf_map = np.zeros((paf.shape[0], paf.shape[1]))
#    for j in range(2):
#        paf_map += paf[:, :, i, j] + paf[:, :, i, j]
#    
#    paf_map = cv2.resize(paf_map, (400, 400))
#    
#    cv2.imshow('paf map', paf_map)
#    cv2.waitKey()
#    cv2.destroyAllWindows()
#
#paf_map = np.zeros((paf.shape[0], paf.shape[1]))
#for i in range(paf.shape[3]):
#    paf_map += paf[:, :, 0, i] + paf[:, :, 1, i]
#
#paf_map = cv2.resize(paf_map, (400, 400))
#
#paf_map = (paf_map > 0.3).astype(np.float32)
#
#cv2.imshow('paf map', paf_map*255)
#cv2.waitKey()
#cv2.destroyAllWindows()
#
#test = paf[:, :, 12, 0]
#cv2.imshow('paf', test*255)
#cv2.waitKey()
#cv2.destroyAllWindows()

#---------This section is to find the peaks in the confidence maps
def find_joint_peaks(heatmap, orig_img_shape, threshold):
    """
    Given a heatmap, find the peaks of the detected joints with 
    confidence score > threshold.
    Inputs:
        heatmap: The heatmaps for all the joints. It is of shape: h, w, num_joints.
        orig_img_shape: The shape of the original image: (x, y, num_channels)
        threshold: The minimum score for each joint
    Output:
        A list of the detected joints. There is a sublist for each joint 
        type (18 in total) and each sublist contains a tuple with 4 values:
        (x, y, confidence score, unique_id). The number of tuples is equal
        to the number of joints detected of that type. For example, if there
        are 3 nose detected (joint type 0), then there will be 3 tuples in the 
        nose sublist            
    """
    joints_list = list()
    counter = 0 # This will serve as unique id for all joints
    heatmap = cv2.resize(heatmap, (orig_img_shape[1], orig_img_shape[0]),
                                    interpolation=cv2.INTER_CUBIC)
    for i in range(heatmap.shape[2]-1):
        sub_list = list()
        joint_map = heatmap[:, :, i] # the heatmap for the particular joint type
        # Smoothen the heatmap
        joint_map = gaussian_filter(joint_map, sigma=3)
        map_left = np.zeros(joint_map.shape)
        map_left[1:, :] = joint_map[:-1, :]
        map_right = np.zeros(joint_map.shape)
        map_right[:-1, :] = joint_map[1:, :]
        map_up = np.zeros(joint_map.shape)
        map_up[:, 1:] = joint_map[:, :-1]
        map_down = np.zeros(joint_map.shape)
        map_down[:, :-1] = joint_map[:, 1:]
        
        peaks_binary = np.logical_and.reduce((joint_map >= map_left,
                                              joint_map >= map_right,
                                              joint_map >= map_up,
                                              joint_map >= map_down,
                                              joint_map > threshold))
        x_index, y_index = np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]
#        print(joint_map.shape)
#        print(type(confidence))
        for x, y in zip(x_index, y_index):
            confidence = joint_map[y, x]
            sub_list.append((x, y, confidence, counter))
            counter += 1
        
        joints_list.append(sub_list)
    return joints_list
                
joints_list = find_joint_peaks(conf, orig_img_shape, threshold)
        

for joint_type in joints_list:
    for tuple_ in joint_type:
        x_index = tuple_[0]
        y_index = tuple_[1]
        cv2.circle(orig_img, (x_index, y_index), 3, (255, 0, 0))
        
cv2.imshow('img', orig_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# -------Now, we can use the PAFs to connect joints.------
for i in range(paf.shape[3]):
    map_ = paf[:, :, 0, i] + paf[:, :, 1, i]
    cv2.imshow('paf', cv2.resize(map_*255, (400, 400)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_connected_joints(upsampled_paf, joints_list, num_inter_pts = 10):
    """
    For every type of limb (eg: forearm, shin, etc.), look for every potential
    pair of joints (eg: every wrist-elbow combination) and evaluate the PAFs to
    determine which pairs are indeed body limbs.
    Inputs:
        upsampled_paf: PAFs upsampled to the original image size.
        joints_list: The ist of joints made by the find_joints_peaks()
        num_inter_pts: The number of points to consider to integrate the PAFs
            and give score to connection candidate
    """
    
    connected_limbs = []
    
    limb_intermed_coords = np.empty((4, num_inter_pts), dtype=np.int)
    for limb_type in range(len(BODY_PARTS)):
        # List of all joints of source
        joints_src = joints_list[BODY_PARTS[limb_type][0]]
        # List of all joints of destination
        joints_dest = joints_list[BODY_PARTS[limb_type][1]]
        if len(joints_src) == 0 or len(joints_dest) == 0:
            # No limbs of this type found. For example if no left knee or no left waist
            # found, then there is no limb connection possible.
            connected_limbs.append([])
        else: 
            connection_candidates = []
            # Specify the paf index that contains the x-coord of the paf for
            # this limb
            limb_intermed_coords[2, :] = 2*limb_type
            # And the y-coord paf index
            limb_intermed_coords[3, :] = 2*limb_type + 1
            
            for i, joint_src in enumerate(joints_src):
                for j, joint_dest in enumerate(joints_dest):
                    # Subtract the position of both joints to obtain the
                    # direction of the potential limb
                    x_src, y_src = joint_src[0], joint_src[1]
                    x_dest, y_dest = joint_dest[0], joint_dest[1]
                    limb_dir = np.asarray([x_dest - x_src, y_dest - y_src])
                    # Compute the norm of the potential limb
                    limb_dist = np.sqrt(np.sum(limb_dir ** 2)) + 1e-8
                    limb_dir = limb_dir / limb_dist  # Normalize limb_dir to be a unit vector
                    
                    # Get intermediate points between the source and dest
                    # For x coordinate
                    limb_intermed_coords[1, :] = np.round(np.linspace(
                            joint_src[0], joint_dest[0], num=num_inter_pts))
                    # For y coordinate
                    limb_intermed_coords[0, :] = np.round(np.linspace(
                            joint_src[1], joint_dest[1], num=num_inter_pts))
#                    print(f'index 2: {limb_intermed_coords[2]}')
#                    print(f'index 3: {limb_intermed_coords[3]}')
#                    print(limb_intermed_coords)
                    # Get all the intermediate points
                    intermed_paf = upsampled_paf[limb_intermed_coords[0,: ],
                                                 limb_intermed_coords[1,:],
                                                 limb_intermed_coords[2:4, :]
                                                 ].transpose()
                    score_intermed_points = intermed_paf.dot(limb_dir)
                    score_penalizing_long_dist = score_intermed_points.mean() + min(0.5 * upsampled_paf.shape[0] / limb_dist - 1, 0)
                    
                    # Append to connection candidates
                    connection_candidates.append([joint_src[3], joint_dest[3], score_penalizing_long_dist])
                    
            # Sort the connections based on their score_penalizing_long_distance
            # Key is used to specify which element to consider while sorting.
            connection_candidates = sorted(connection_candidates, key=lambda x: x[2],
                                           reverse=True) # Reverse to use descending order.
            
            # Create an empty array to store connections.
            # We will add more using numpy.vstack
            connections = np.empty((0, 3)) 
            # See how many connections are possible. 
            # There can be only 1 connection if only 1 neck is found but 5 noses are found.
            max_connections = min(len(joints_dest), len(joints_src))
            # Go over all potential candidates that are already sorted.
            for potentail_connections in connection_candidates:
                src_id, dest_id = potentail_connections[:2]
                # Check that neither src nor destination have already been connected.
                if src_id not in connections and dest_id not in connections:
                    connections = np.vstack([connections, potentail_connections])
                if len(connections) > max_connections:
                    break
            connected_limbs.append(connections)
        
    return connected_limbs
    
# We need the PAF's upsampled to the to original image resolution.
paf_upsampled = cv2.resize(paf, (orig_img_shape[1], orig_img_shape[0]))

connected_limbs = get_connected_joints(paf_upsampled, joints_list)

num_intermed_points = 10

connected_limbs = []

limb_intermed_coords = np.empty((4, num_intermed_points), dtype=np.int)

# Added by me
limb_type = 6

# List of all joints of source
joints_src = joints_list[BODY_PARTS[limb_type][0]]
# List of all joints of destination
joints_dest = joints_list[BODY_PARTS[limb_type][1]]
if len(joints_src) == 0 or len(joints_dest) == 0:
    # No limbs of this type found. For example if no left knee or no left waist
    # found, then there is no limb connection possible.
    connected_limbs.append[0]
else: 
    connection_candidates = []
    # Specify the paf index that contains the x-coord of the paf for
    # this limb
    limb_intermed_coords[2, :] = 2*limb_type
    # And the y-coord paf index
    limb_intermed_coords[3, :] = 2*limb_type + 1
    
    for i, joint_src in enumerate(joints_src):
        for j, joint_dest in enumerate(joints_dest):
            # Subtract the position of both joints to obtain the
            # direction of the potential limb
            x_src, y_src = joint_src[0], joint_src[1]
            x_dest, y_dest = joint_dest[0], joint_dest[1]
#            confidence
#            joint_src = np.asarray([x_src, y_src])
#            joint_dest = np.asarray([x_dest, y_dest])
            limb_dir = np.asarray([x_dest - x_src, y_dest - y_src])
            # Compute the norm of the potential limb
            limb_dist = np.sqrt(np.sum(limb_dir ** 2)) + 1e-8
            limb_dir = limb_dir / limb_dist  # Normalize limb_dir to be a unit vector
            
            # Get intermediate points between the source and dest
            # For x coordinate
            limb_intermed_coords[0, :] = np.round(np.linspace(
                    joint_src[0], joint_dest[0], num=num_intermed_points))
            # For y coordinate
            limb_intermed_coords[1, :] = np.round(np.linspace(
                    joint_src[1], joint_dest[1], num=num_intermed_points))
            # Get all the intermediate points
            intermed_paf = paf_copy[limb_intermed_coords[0,: ],
                                    limb_intermed_coords[1,:],
                                    limb_intermed_coords[2:4, :]].transpose()
            score_intermed_points = intermed_paf.dot(limb_dir)
            score_penalizing_long_dist = score_intermed_points.mean() + min(0.5 * paf_copy.shape[0] / limb_dist - 1, 0)
            
            # Append to connection candidates
            connection_candidates.append([joint_src[3], joint_dest[3], score_penalizing_long_dist])
            
    # Sort the connections based on their score_penalizing_long_distance
    # Key is used to specify which element to consider while sorting.
    connection_candidates = sorted(connection_candidates, key=lambda x: x[2],
                                   reverse=True) # Reverse to use descending order.
    
    # Create an empty array to store connections.
    # We will add more using numpy.vstack
    connections = np.empty((0, 3)) 
    # See how many connections are possible. 
    # There can be only 1 connection if only 1 neck is found but 5 noses are found.
    max_connections = min(len(joints_dest), len(joints_src))
    # Go over all potential candidates that are already sorted.
    for potentail_connections in connection_candidates:
        src_id, dest_id = potentail_connections[:2]
        # Check that neither src nor destination have already been connected.
        if src_id not in connections and dest_id not in connections:
            connections = np.vstack([connections, potentail_connections])
        if len(connections) > max_connections:
            break
    connected_limbs.append(connections)
