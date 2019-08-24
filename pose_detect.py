# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 12:08:04 2019

@author: myidispg
"""

import torch
import numpy as np
import os
import cv2

from scipy.ndimage.filters import gaussian_filter, maximum_filter
from scipy.ndimage.morphology import generate_binary_structure

from models.paf_model_v2 import StanceNet
from utilities.constants import threshold, BODY_PARTS, num_joints

class PoseDetect():
    
    def __init__(self, img, model_path, use_gpu=True):
        self.img = img # Img is numpy array
        self.orig_img = img # Keep a copy
        self.orig_img_shape = img.shape
        print('Loading pre-trained model now.')
        self.model = StanceNet(18, 38).eval()        
        self.model.load_state_dict(torch.load(model_path))
        print('Model loaded.')
        
        # Turn into PyTorch tensor, add batch size dimension and permute to required shape
        self.img = torch.from_numpy(self.img).view(1,
                                   self.img.shape[0],
                                   self.img.shape[1],
                                   self.img.shape[2]).permute(0, 3, 1, 2)
        
        if use_gpu:
            if torch.cuda.is_available():
                device = torch.device('cuda')
                self.model = self.model.to(device)
                self.img = self.img.to(device)
            else:
                print('No GPU available. Proceeding on CPU')
        
                
    def find_joint_peaks(heatmap, orig_img_shape):
        """
        Given a heatmap, find the peaks of the detected joints with 
        confidence score > threshold.
        Inputs:
            heatmap: The heatmaps for all the joints. It is of shape: h, w, num_joints.
            orig_img_shape: The shape of the original image: (x, y, num_channels)
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
            for x, y in zip(x_index, y_index):
                confidence = joint_map[y, x]
                sub_list.append((x, y, confidence, counter))
                counter += 1
            
            joints_list.append(sub_list)
        return joints_list
    
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
                # No limbs of this type found. For example if no left knee or
                # no left waist found, then there is no limb connection possible.
                connected_limbs.append([])
            else: 
                connection_candidates = [] # list for all candiates of this type.
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
                        limb_intermed_coords[0, :] = np.round(np.linspace(
                                x_src, x_dest, num=num_inter_pts))
                        # For y coordinate
                        limb_intermed_coords[1, :] = np.round(np.linspace(
                                y_src, y_dest, num=num_inter_pts))
                        # Get all the intermediate points
                        intermed_paf = upsampled_paf[limb_intermed_coords[1,: ],
                                                     limb_intermed_coords[0,:],
                                                     limb_intermed_coords[2:4, :]
                                                     ].transpose()
                        score_intermed_points = intermed_paf.dot(limb_dir)
                        score_penalizing_long_dist = score_intermed_points.mean() + min(0.5 * upsampled_paf.shape[0] / limb_dist - 1, 0)
                        
    #                   Criterion 1: At least 80% of the intermediate points have
    #                    a score higher than 0.05
                        criterion1 = (np.count_nonzero(
                            score_intermed_points > 0.02) > 0.5 * num_inter_pts)
    ##                     Criterion 2: Mean score, penalized for large limb
    ##                     distances (larger than half the image height), is
    ##                     positive
                        criterion2 = (score_penalizing_long_dist > 0)
                        if criterion1 and criterion2:
                            # Last value is the combined paf(+limb_dist) + heatmap
                            # scores of both joints
                            connection_candidates.append([joint_src[3], joint_dest[3],
                                                          score_penalizing_long_dist,
                                                          (x_src, y_src),
                                                          (x_dest, y_dest)])
    
    #                    
    #                    connection_candidates.append([joint_src[3], joint_dest[3],
    #                                                      score_penalizing_long_dist,
    #                                                      (x_src, y_src),
    #                                                      (x_dest, y_dest)])
                        
    #            Sort the connections based on their score_penalizing_long_distance
    #            Key is used to specify which element to consider while sorting.
                connection_candidates = sorted(connection_candidates, key=lambda x: x[2],
                                               reverse=True) # Reverse to use descending order.
                
                used_idx1 = [] # A list to keep track of the used src joints
                used_idx2 = [] # A list to keep track of the used dest joints
                
                connection = [] # A list to store all the final connection of limb type.
                
                for potential_connection in connection_candidates:
                    if potential_connection[0] in used_idx1 or potential_connection[1] in used_idx2:
                        continue
                    
                    connection.append(potential_connection)
                    used_idx1.append(potential_connection[0])
                    used_idx2.append( potential_connection[1])
                
                connected_limbs.append(connection)
        return connected_limbs
    
    def find_people(connected_limbs, joints_list):
        """
        Associate limbs belonging to the same person together.
        Inputs:
            connected_limbs: The limbs outputs of the get_connected_joints()
            joints_list: An unraveled list of all the joints.
        Outputs:
            people: 2d np.array of size num_people x (NUM_JOINTS+2). For each person found:
                # First NUM_JOINTS columns contain the index (in joints_list) of the
                joints associated with that person (or -1 if their i-th joint wasn't found)
                # 2nd last column: Overall score of the joints+limbs that belong
                to this person.
                # Last column: Total count of joints found for this person
            
        """
        
        people = list()
        
        for limb_type in range(len(BODY_PARTS)): # Detected limb of a type
            joint_src_type, joint_dest_type = BODY_PARTS[limb_type]
    #        print(f'src: {joint_src_type}, dest: {joint_dest_type}')
            
            for limbs_of_type in connected_limbs[limb_type]: # All limbs detected of a type
                person_assoc_idx = list()
                for person, person_limbs in enumerate(people): # Find all people who can be associated.
                    if limbs_of_type[0] == person_limbs[joint_src_type] or limbs_of_type[1] == person_limbs[joint_dest_type]:
                        person_assoc_idx.append(person)
                        
                # If one of the joints has been associated to a person, and either
                # the other joint is also associated with the same person or not
                # associated to anyone yet:
                if len(person_assoc_idx) == 1:
                    person_limbs = people[person_assoc_idx[0]]
                    # If the other joint is not associated not yet
                    if person_limbs[joint_dest_type] != limbs_of_type[1]:
                        # Associate with current person
                        person_limbs[joint_dest_type] = limbs_of_type[1]
                        # Increase the number of limbs associated to this person
                        person_limbs[-1] += 1
                        # And update the total score (+= heatmap score of joint_dst
                        # + score of connecting joint_src with joint_dst)
                        person_limbs[-2] += joints_list[limbs_of_type[1], 2] + limbs_of_type[2]
                elif len(person_assoc_idx) == 2: # if found 2 and disjoint, merge them
                    person1_limbs = people[person_assoc_idx[0]]
                    person2_limbs = people[person_assoc_idx[1]]
                    membership = ((person1_limbs >= 0) & (person2_limbs >= 0))[:-2]
                    if not membership.any(): # If both people have no same joints connected, merge them into a single person
                        # Update which joints are connected
                        person1_limbs[:-2] += (person2_limbs[:-2] + 1)
                        # Update the overall score and total count of joints
                        # connected by summing their counters
                        person1_limbs[-2:] += person2_limbs[-2:]
                        # Add the score of the current joint connection to the 
                        # overall score
                        person1_limbs[-2] += limbs_of_type[2]
                        people.pop(person_assoc_idx[1])
                    else: # Same case as len(person_assoc_idx)==1 above
                        person1_limbs[joint_dest_type] = limbs_of_type[1]
                        person1_limbs[-1] += 1
                        person1_limbs[-2] += joints_list[limbs_of_type[1], 2] + limbs_of_type[2]
                else: # No person has claimed any of these joints, create a new person
                    # Initialize person info to all -1 (no joint associations)
                    row = -1 * np.ones(num_joints + 2)
                    # Store the joint info of new connection
                    row[joint_src_type] = limbs_of_type[0]
                    row[joint_dest_type] = limbs_of_type[1]
                    # Total count of conencted joints for this person: 2
                    row[-1] = 2
                    # Compute overall score: score joint_src + score joint_dst + score connection
                    # {joint_src,joint_dst}
                    row[-2] = sum(joints_list[limbs_of_type[:2], 2]
                                  ) + limbs_of_type[2]
                    people.append(row)
                
                    
        # Delete people who have very few parts connected
        people_to_delete = []
        for person_id, person_info in enumerate(people):
            if person_info[-1] < 3 or person_info[-2] / person_info[-1] < 0.2:
                people_to_delete.append(person_id)
        
        # Traverse the list in reverse order so we delete indices starting from the
        # last one (otherwise, removing item for example 0 would modify the indices of
        # the remaining people to be deleted!)
        for index in people_to_delete[::-1]:
            people.pop(index)
        
        # Appending items to a np.array can be very costly (allocating new memory, copying over the array, then adding new row)
        # Instead, we treat the set of people as a list (fast to append items) and
        # only convert to np.array at the end
        return np.array(people)
    
    def plot_poses(self, people, joint_list_unraveled):
        
        

            