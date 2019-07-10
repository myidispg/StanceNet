# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 15:46:42 2019

@author: myidispg
"""

"""
This file stores all the helper functions for the project.
"""

def get_image_name(image_id):
    """
    Given an image id, adds zeros before it so that the image name length is 
    of 12 digits as required by the database.
    Input:
        image_id: the image id without zeros.
    Output:
        image_name: image name with zeros added and the .jpg extension
    """
    num_zeros = 12 - len(str(image_id))
    zeros = '0' * num_zeros
    image_name = zeros + str(image_id) + '.jpg'
    
    return image_name

def get_image_id_from_filename(filename):
    """
    Get the image_id from a filename with .jpg extension
    """    
    return int(filename.split('.')[0])

def group_keypoints(keypoints):
    """
    Given a list of 51 keypoints, groups the keypoints into a list of 17 tuples.
    Each tuple is (x, y, v) where 
    v=0: not labeled, v=1: labeled but not visible, v=2: labeled and visible.
    """
    arranged_keypoints = list()
    for i in range(0, len(keypoints), 3):
        arranged_keypoints.append((keypoints[i], keypoints[i+1], keypoints[i+2]))
    
    return arranged_keypoints

def draw_skeleton(image_id, all_keypoints, skeleton_limb_indices,
                  wait_time = 0, val=False):
    """
    Given the image_id and keypoints of the image, draws skeleton accordingly.
    Inputs:
        image_id: The id of the image as per the dataset.
        all_keypoints: A 2d list with keypoints of size: (num_people, 51)
                   Here, 51 is for the keyppoints in a single person.
        skeleton_limb_indices: The joint indices to make connections from 
                               keypoints to make a skeleton. Defined in constants.
        val: True is using validation data. False by default.

    Outputs: None. Only shows the image.                       
    """
    
    import cv2
    import os
    
    from constants import dataset_dir
    
    image_name = get_image_name(image_id)
    
    if val:
        image_path = os.path.join(dataset_dir, 
                                    'val2017', 
                                    image_name)
    else:
        image_path = os.path.join(dataset_dir,
                                  'train2017',
                                  image_name)
    
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    # For each keypoint in keypoints, draw the person.
    for person_keypoints in all_keypoints:
        
        # Display joints
        for keypoint in person_keypoints:
            if keypoint[2] != 0:
                cv2.circle(img, (keypoint[0], keypoint[1]), 0, (0, 255, 255), 6)
        
        # Draw limbs for the visible joints
        # Note: 1 is subtracted because indices start from 0.
        for joint_index in skeleton_limb_indices:
            if person_keypoints[joint_index[0]-1][2] != 0:
                if person_keypoints[joint_index[1]-1][2] != 0:
                    x1 = person_keypoints[joint_index[0]-1][0]
                    y1 = person_keypoints[joint_index[0]-1][1]
                    x2 = person_keypoints[joint_index[1]-1][0]
                    y2 = person_keypoints[joint_index[1]-1][1]
                    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

                
    cv2.imshow('image', img)
    cv2.waitKey(wait_time)
    cv2.destroyAllWindows()