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
    
    from utilities.constants import dataset_dir
    
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
        sigma: used to control the spread of the peak.
    Output:
        conf_map: A numpy array of shape: (batch_size, im_width, im_height, num_joints)
    """
    
    # Necessary imports for this function.
    import cv2
    import os
    import numpy as np
    from utilities.constants import dataset_dir, im_width_small, im_height_small
    from utilities.constants import im_height, im_width, num_joints
    
    num_images = len(indices)
    
    conf_map = np.zeros((num_images, im_width_small, im_height_small, num_joints), np.float16)
    
    # For image in all images
    for image_id in indices:
        
        heatmap_image = np.zeros((im_width, im_height, num_joints))
        
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
                    x_ind, y_ind = np.meshgrid(np.arange(im_width), np.arange(im_height))
                    numerator = (-(x_ind-x_index)**2) + (-(y_ind-y_index)**2)
                    heatmap_joint = np.exp(numerator/sigma)
                    heatmap_image[:, :, part_num] = np.maximum(heatmap_joint, heatmap_image[:, :, part_num])
                    heatmap_image[:, :, part_num] = cv2.resize(heatmap_image[:,:, part_num], (im_width_small, im_height_small))
                    
                    
                    conf_map[image_id % num_images, :, :, :] = heatmap_image        
    
    return conf_map



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
    from utilities.constants import dataset_dir, im_width, im_height, num_joints, skeleton_limb_indices
    
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

                    part_line_segment = joint_two_loc - joint_one_loc
                    norm = np.linalg.norm(part_line_segment)

                    norm = norm + 1e-8 if norm == 0 else norm # To make sure it is not equal to zero.

                    vector = (part_line_segment)/norm

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
                    mask = (cond_1 & cond_2 & cond_3).astype(np.float32)
                    
                    # put the values
                    paf[image_id % num_images, :, :, 0, limb] += np.transpose(mask) * vector[0]
                    paf[image_id % num_images, :, :, 1, limb] += np.transpose(mask) * vector[1]
    return paf


def gen_data(all_keypoints, batch_size = 64, im_width = 224, im_height = 224, val=False):
    
    # Necessary imports
    import cv2
    import os
    import numpy as np
    
    from utilities.constants import dataset_dir
    from utilities.helper import generate_confidence_maps, get_image_name, generate_paf
    
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
    
        yield images, conf_maps, pafs
    
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
        
        yield images, conf_maps, pafs
        
# -------EVALUATION FUNCTIONS-------------------------
def find_joints(confidence_maps, threshold = 0.7):
    """
    Finds the list of peaks with the confidence scores from a confidence map 
    for an image. The confidence map has all the heatmaps for all joints.
    Inputs:
        confidence_map: A confidence map for all joints of a single image.
            expected shape: (1, im_width, im_height, num_joints)
        threshold: A number less than one used to eliminate low probability 
            detections.
    Output:
        joints_list: A list of all the detected joints. 
        It is a list of (num_joints) where each element is a list of detected 
        joints. Each joint is of following format: (x, y, confidence_score)
    """
    
    import numpy as np
    from scipy.ndimage.filters import maximum_filter
    from scipy.ndimage.morphology import generate_binary_structure
    
    # Check if the input is of expected shape
    assert len(confidence_maps.shape) == 3, "Wrong input confidence map shape."
    
    joints_list = []
    
    for joint_num in range(confidence_maps.shape[2]):
        # Detected joints for this type
        joints = []
        
        # Get the map for the joint and reshape to 2-d
        conf_map = confidence_maps[:, :, joint_num].reshape(confidence_maps.shape[0], confidence_maps.shape[1])
        # Threshold the map to eliminate low probability locations.
        conf_map = np.where(conf_map >= threshold, conf_map, 0)
        # Apply a 2x1 convolution(kind of).
        # This replaces a 2x1 window with the max of that window.
        peaks = maximum_filter(conf_map.astype(np.float64), footprint=generate_binary_structure(2, 1))
        peaks = np.where(peaks == 0, 0.1, peaks)
        # Now equate the peaks with joint_one.
        # This works because the maxima in joint_one will be at a single place and
        # equating with peaks will result in a single peak with all others as 0. 
        peaks = np.where(peaks == conf_map, peaks, 0)
        y_indices, x_indices = np.where(peaks != 0)

        for x, y in zip(x_indices, y_indices):
            joints.append((x, y, conf_map[y, x]))
        
        joints_list.append(joints)
        
    return joints_list