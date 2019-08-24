# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 18:48:23 2019

@author: myidispg
"""

import torch

import numpy as np
import cv2
import sys

from scipy.ndimage.filters import gaussian_filter, maximum_filter
from scipy.ndimage.morphology import generate_binary_structure

from models.paf_model_v2 import StanceNet
from utilities.constants import threshold, BODY_PARTS, num_joints

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

print('Loading the pre-trained model')
model = StanceNet(18, 38).eval()
model.load_state_dict(torch.load('trained_models/trained_model.pth'))
model = model.to(device)
print(f'Loading the model complete.')

#img = cv2.imread(os.path.join('Coco_Dataset', 'val2017', '000000008532.jpg'))
orig_img = cv2.imread('test_images/footballers.jpg')
orig_img_shape = orig_img.shape
img = orig_img.copy()/255
img = cv2.resize(img, (400, 400))
#cv2.imshow('image', orig_img)
#cv2.waitKey()
#cv2.destroyAllWindows()

img = torch.from_numpy(img).view(1, img.shape[0], img.shape[1], img.shape[2]).permute(0, 3, 1, 2)

img = img.to(device).float()

paf, conf = model(img)

paf = paf.cpu().detach().numpy()
conf = conf.cpu().detach().numpy()

# Remove the extra dimension of batch size
conf = np.squeeze(conf.transpose(2, 3, 1, 0))
paf = np.squeeze(paf.transpose(2, 3, 1, 0))

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
        for x, y in zip(x_index, y_index):
            confidence = joint_map[y, x]
            sub_list.append((x, y, confidence, counter))
            counter += 1
        
        joints_list.append(sub_list)
    return joints_list

joints_list = find_joint_peaks(conf, orig_img_shape, threshold)

#def find_joint_peaks(heatmap, original_image, threshold):
#    """
#    Given a heatmap, find the peaks of the detected joints with 
#    confidence score > threshold.
#    Inputs:
#        heatmap: The heatmaps for all the joints. It is of shape: h, w, num_joints.
#        orig_img_shape: The shape of the original image: (x, y, num_channels)
#        threshold: The minimum score for each joint
#    Output:
#        A list of the detected joints. There is a sublist for each joint 
#        type (18 in total) and each sublist contains a tuple with 4 values:
#        (x, y, confidence score, unique_id). The number of tuples is equal
#        to the number of joints detected of that type. For example, if there
#        are 3 nose detected (joint type 0), then there will be 3 tuples in the 
#        nose sublist            
#    """
#    
#    joints_list = list()
#    counter = 0
#    heatmap = cv2.resize(heatmap, (orig_img_shape[1], orig_img_shape[0]),
#                                    interpolation=cv2.INTER_CUBIC)
#    for i in range(heatmap.shape[2]-1):
#        sub_list = list()
#        joint_map = heatmap[:, :, i]
#        joint_map = gaussian_filter(joint_map, sigma=3)
#        
#        structure = generate_binary_structure(2, 1)
#        joint_map = (joint_map > 0.01) * joint_map
#        peaks = maximum_filter(joint_map, footprint=structure) == joint_map
##        peaks = maximum_filter(joint_map, footprint=np.ones((5, 5))) == joint_map
#        peaks = peaks * joint_map
#        y_index, x_index = np.where(peaks != 0)
#        
#        for x, y in zip(x_index, y_index):
#            confidence = joint_map[y, x]
#            sub_list.append((x, y, confidence, counter))
#            counter += 1
#        joints_list.append(sub_list)
#    
#    return joints_list
#        
#joints_list = find_joint_peaks(conf, orig_img_shape, threshold)

for joint_type in joints_list:
    for tuple_ in joint_type:
        x_index = tuple_[0]
        y_index = tuple_[1]
        cv2.circle(orig_img, (x_index, y_index), 3, (255, 0, 0))

cv2.imshow('img', orig_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# -------Now, we can use the PAFs to connect joints.------
#for i in range(len(BODY_PARTS)):
#    paf_index = [i*2, i*2+1] # get the indices of the pafs.
#    paf_limb = paf[:, :, paf_index[0]: paf_index[1]+1] # get the corresponding heatmaps.
#    map_ = paf_limb[:, :, 0] + paf_limb[:, :, 1]
#    map_ = np.where(map_ > 0.1, map_, 0)
#    cv2.imshow('paf', cv2.resize(map_*255, (400, 400)))
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()

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
            
    
# We need the PAF's upsampled to the to original image resolution.
paf_upsampled = cv2.resize(paf, (orig_img_shape[1], orig_img_shape[0]))

connected_limbs = get_connected_joints(paf_upsampled, joints_list)

# Visualize the limbs
for limb_type in connected_limbs:   
    for limb in limb_type:
        src, dest = limb[3], limb[4]
        cv2.line(orig_img, src, dest, (0, 255, 0), 2)
    
cv2.imshow('img', orig_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('readme_media/footballers.png', orig_img)

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

joint_list_unraveled = np.array([tuple(peak) + (joint_type,) for joint_type,
                                                           joint_peaks in enumerate(joints_list) for peak
                           in joint_peaks])

people = find_people(connected_limbs, joint_list_unraveled)
                

for person_joint_info in people: # For person in all people
    for limb_type in range(len(BODY_PARTS)): # For each limb in possible types.
        limb_src_index, limb_dest_index = BODY_PARTS[limb_type] # Get the index of src and destination joints
#        print(f'Person: {person_joint_info}, src: {limb_src_index}, dest: {limb_dest_index}')
        # Index of src joint for limb in unraveled list
        src_joint_index_joints_list = int(person_joint_info[limb_src_index])
        # Index of dest joitn fot limb in unraveled list
        dest_joint_index_joints_list = int(person_joint_info[limb_dest_index])
        if src_joint_index_joints_list == -1 or dest_joint_index_joints_list == -1:
            continue
        
        joint_src = int(joint_list_unraveled[src_joint_index_joints_list][0]), int(joint_list_unraveled[src_joint_index_joints_list][1])
        joint_dest = int(joint_list_unraveled[dest_joint_index_joints_list][0]), int(joint_list_unraveled[dest_joint_index_joints_list][1])
        
        # Draw the joints by a circle
        # Circle for source
        cv2.circle(orig_img, joint_src, 2, (255, 0, 0))
        # Circle for dest
        cv2.circle(orig_img, joint_dest, 2, (255, 0, 0))
        
        # Draw the limbs
        cv2.line(orig_img, joint_src, joint_dest, (0, 255, 0), 2)
        
        
cv2.imshow('img', orig_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ------ Work on video--------
ERASE_LINE = '\x1b[2K'
CURSOR_UP_ONE = '\x1b[1A'

vid = cv2.VideoCapture('test_images/action_scene.mp4')
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0,
                      (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                       int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))))

total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

print(f'Video file loaded and working on detecting joints.')

frames_processed = 0
while(vid.isOpened()):
    print(f'Processed {frames_processed} frames out of {total_frames}')
    ret, orig_img = vid.read()
    orig_img_shape = orig_img.shape
    img = orig_img.copy()/255
    img = cv2.resize(img, (400, 400))
    # Convert the frame to a torch tensor
    img = torch.from_numpy(img).view(1, img.shape[0], img.shape[1], img.shape[2]).permute(0, 3, 1, 2)
    img = img.to(device).float()
    # Get the model's output
    paf, conf = model(img)
    # Convert back to numpy
    paf = paf.cpu().detach().numpy()
    conf = conf.cpu().detach().numpy()
    # Remove the extra dimension of batch size
    conf = np.squeeze(conf.transpose(2, 3, 1, 0))
    paf = np.squeeze(paf.transpose(2, 3, 1, 0))

    # Get the joints
    joints_list = find_joint_peaks(conf, orig_img_shape, threshold)
    # Draw joints on the orig_img
    for joint_type in joints_list:
        for tuple_ in joint_type:
            x_index = tuple_[0]
            y_index = tuple_[1]
            cv2.circle(orig_img, (x_index, y_index), 3, (255, 0, 0))
            
    # Upsample the paf
    paf_upsampled = cv2.resize(paf, (orig_img_shape[1], orig_img_shape[0]))
    # Get the connected limbs
    connected_limbs = get_connected_joints(paf_upsampled, joints_list)
    
    # Draw the limbs too.
    for limb_type in connected_limbs:   
        for limb in limb_type:
            src, dest = limb[3], limb[4]
            cv2.line(orig_img, src, dest, (0, 255, 0), 2)
            
    out.write(orig_img)
    frames_processed += 1
    sys.stdout.write(CURSOR_UP_ONE)
    sys.stdout.write(ERASE_LINE)
    
    break
