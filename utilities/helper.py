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
                                    'new_val2017', 
                                    image_name)
    else:
        image_path = os.path.join(dataset_dir,
                                  'new_train2017',
                                  image_name)
    
    print(image_path)
    
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
    

def generate_confidence_maps(all_keypoints, indices, val=False, sigma=7):
    """
    Generate confidence maps for a batch of indices.
    The generated confidence_maps are of shape: 
        (batch_size, im_width, im_height, num_joints)
    Input:
        all_keypoints: Keypoints for all the images in the dataset. It is a 
        dictionary that contains image_id as keys and keypoints for each person.
        indices: a list of indices for which the conf_maps are to be generated.
        val: True if used for validation set, else false
        sigma: used to generate the confidence score. Higher values lead higher score.
    Output:
        conf_map: A numpy array of shape: (batch_size, im_width, im_height, num_joints)
    """
    
    # Necessary imports for this function.
    import math
    import cv2
    import os
    import numpy as np
    from constants import dataset_dir, im_width, im_height, num_joints
    
    num_images = len(indices)
    
    conf_map = np.zeros((num_images, im_width, im_height, num_joints), np.float16)
    
    # For image in all images
    for image_id in indices:
        
        # For a person in the image
        for person in range(len(all_keypoints[image_id])):
            # For all keypoints for the person.
            for part_num in range(len(all_keypoints[image_id][person])):
                # For keypoint in all_keypoints[image_id][person]:
                # Get the pixel values at a given keypoint across all 3 channels.
                # Note that our labels have images (im_width, im_height),
                # OpenCV has (im_height, im_width)
                x_index = all_keypoints[image_id][person][part_num][0]
                y_index = all_keypoints[image_id][person][part_num][1]
                visibility = all_keypoints[image_id][person][part_num][2]
                
                # Generate heatmap only around the keypoint, leave others as 0
                if visibility != 0:
                    for i in range(x_index - (sigma//2), x_index + (sigma//2)):
                        if i >= im_width:
                            break
                        for j in range(y_index - (sigma // 2), y_index + (sigma // 2)):
                            if j >= im_height:
                                break
                            l2_norm_squared = ((i - x_index) ** 2) + ((j-y_index) ** 2)
                            pixel_value = math.exp((-l2_norm_squared) / (sigma ** 2))
                            
                            conf_map[image_id % num_images, i, j, part_num] = pixel_value

    
    return conf_map


def sign(x):
    
    if x < 0:
        return -1
    elif x == 0:
        return 0
    else: 
        return 1

def bressenham_line(start, end):
    
    points_bet = list()
    
    x1, y1 = start[0], start[1]
    x2, y2 = end[0], end[1]
    
    x = x1
    y = y1
    
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    
    s1 = sign(x2-x1)
    s2 = sign(y2 - y1)
    
    if dy > dx:
        dx, dy = dy, dx
        interchange = 1
    else:
        interchange = 0
    
    e = 2 * dy - dx
    a = 2 * dy
    b = 2 * dy - 2 * dx
    
    
    points_bet.append((x, y))
    for i in range(dx):
        if e < 0:
            if interchange == 1:
                y += s2
            else:
                x += s1
            e += a
        else:
            y += s2
            x += s1
            e += b
        points_bet.append((x, y))
    
    return points_bet
    

def generate_paf(all_keypoints, indices, sigma=5, val=False):
    """
    Generate Part Affinity Fields given a batch of keypoints.
    Inputs:
        all_keypoints: Keypoints for all the images in the dataset. It is a 
        dictionary that contains image_id as keys and keypoints for each person.
        indices: The list of indices from the keypoints for which PAFs are to 
        be generated.
        sigma: The width of a limb in pixels.
        val(Default=False): True if validation set, False otherwise. 
        
    Outputs:
        paf: A parts affinity fields map of shape: 
            (batch_size, im_width, im_height, 2, num_joints)
    """
    
    import cv2
    import os
    import numpy as np
    from constants import dataset_dir, im_width, im_height, num_joints, skeleton_limb_indices
    
    num_images = len(indices)
    
    paf = np.zeros((num_images, im_width, im_height, 2, len(skeleton_limb_indices)), np.float16)
    
    # For image in all_images
    for image_id in indices:
        # For each person in the image
        for person in range(len(all_keypoints[image_id])):
            # For each limb in the skeleton
            for limb in range(len(skeleton_limb_indices)):
                
                limb_indices = skeleton_limb_indices[limb]
                
                joint_one_index = limb_indices[0] - 1
                joint_two_index = limb_indices[1] - 1
                
                joint_one_loc = np.asarray(all_keypoints[image_id][person][joint_one_index])
                joint_two_loc = np.asarray(all_keypoints[image_id][person][joint_two_index])
                
                # If there is 0 for visibility, skip the limb
                if all_keypoints[image_id][person][joint_one_index][2] != 0 and all_keypoints[image_id][person][joint_two_index][2] != 0:
                    joint_one_loc = np.asarray(all_keypoints[image_id][person][joint_one_index][:2])
                    joint_two_loc = np.asarray(all_keypoints[image_id][person][joint_two_index][:2])

                    norm = ((joint_one_loc[0] - joint_two_loc[0]) ** 2 + (joint_one_loc[1] - joint_two_loc[1]) ** 2) ** (1/2)

                    vector = (joint_two_loc - joint_one_loc)/norm

                    # https://gamedev.stackexchange.com/questions/70075/how-can-i-find-the-perpendicular-to-a-2d-vector
                    perpendicular_vec = [-vector[1], vector[0]]
                    
                    x, y = np.meshgrid(np.arange(im_width), np.arange(im_height))
                    # Formula according to the paper
                    along_limb = vector[0] * (x-joint_one_loc[0]) + vector[1] * (y-joint_one_loc[1])
                    across_limb = np.abs(perpendicular_vec[0] * (x-joint_one_loc[0]) + perpendicular_vec[1] * (y - joint_one_loc[1])) 
                    
                    # Check the given conditions
                    cond_1 = along_limb >= 0
                    cond_2 = along_limb <= norm
                    cond_3 = across_limb <= sigma
                    mask = cond_1 & cond_2 & cond_3
                    
                    # put the values
                    paf[image_id % num_images, :, :, 0, limb] += np.transpose(mask) * vector[0]
                    paf[image_id % num_images, :, :, 1, limb] += np.transpose(mask) * vector[1]
    return paf


def gen_data(all_keypoints, batch_size = 64, im_width = 224, im_height = 224, val=False):
    
    # Necessary imports
    import cv2
    import os
    import numpy as np
    
    from constants import dataset_dir
    from helper import generate_confidence_maps, get_image_name, generate_paf
    
    batch_count = len(all_keypoints.keys()) // batch_size
    
    count = 0
    
    # Loop over all keypoints in batches
    for batch in range(1, batch_count * batch_size + 1, batch_size):
        
        count += 1
        
        images = np.zeros((batch_size, im_width, im_height, 3), dtype=np.uint8)
        
        # Loop over all individual indices in a batch
        for image_id in range(batch, batch + batch_size):
            img_name = get_image_name(image_id-1)
            
            if val:
                img = cv2.imread(os.path.join(dataset_dir,'new_val2017',img_name))
            else:
                img = cv2.imread(os.path.join(dataset_dir,'new_train2017',img_name))
            
            images[image_id % batch] = img
        
        conf_maps = generate_confidence_maps(all_keypoints, range(batch-1, batch+batch_size-1))
        pafs = generate_paf(all_keypoints, range(batch-1, batch+batch_size-1))
    
        yield count, images, conf_maps, pafs
    
    # Handle cases where the total size is not a multiple of batch_size
    
    if len(all_keypoints.keys()) % batch_size != 0:
        
        start_index = batch_size * batch_count
        final_index = list(all_keypoints.keys())[-1]
        
#        print(final_index + 1 - start_index)
        
        images = np.zeros((final_index + 1 - start_index, im_width, im_height, 3), dtype=np.uint8)
        
        for image_id in range(start_index, final_index + 1):
            img_name = get_image_name(image_id)
            
            if val:
                img = cv2.imread(os.path.join(dataset_dir, 'new_val2017', img_name))
            else:
                img = cv2.imread(os.path.join(dataset_dir, 'new_train2017', img_name))
            
            images[image_id % batch_size] = img
            
        conf_maps = generate_confidence_maps(all_keypoints, range(start_index, final_index + 1))
        pafs = generate_paf(all_keypoints, range(start_index, final_index + 1))
        
        yield count + 1, images, conf_maps, pafs