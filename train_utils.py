# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 19:24:20 2019

@author: myidispg
"""

import torch
import os

import numpy as np

import utilities.helper as helper
import utilities.constants as constants

def train_epoch(model, criterion_conf, criterion_paf, optimizer, 
                keypoints, batch_size, losses, device, val=False,
                print_every=5000):
    """
    This method trains the model for a single epoch.
    Inputs:
        model: The model object to be trained. Instance of OpenPoseModel class.
        criterion_conf: The criterion for confidence maps
        criterion_paf: The criterion for Parts Affinity Fields
        optimizer: The optimizer instance used for updating the model weights.
        keypoints: The list of keypoints for the images. Used to generate batches.
        batch_size: The batch size to be used for training.
        losses: A list in which to append the losses. 
            They will be used for plotting purposes.
        device: A torch.device() object. To see if training on cuda(GPU) or CPU.
        val: A bool to see if this is the training or validation set. 
            True if validation set. Default is False
        print_every: After how many batches to print the loss. Default=5000
    Outputs:
        model: The model trained for one epoch.
        optimizer: The optimizer object after one epoch training. It is saved
            because this contains buffers and parameters that are updated as 
            the model trains.
        losses: The list of losses for this epoch.
    """
    
    # A variable to accumulate losses until print_every is triggered.
    running_loss = 0
    # Count the batch_number
    batch_num = 1
    
    # Use the DataGenerator to generate batches of data and perform training.
    for images, conf_maps, pafs in helper.gen_data(keypoints,
                                                   batch_size=batch_size,
                                                   val=val):
        # Convert all to PyTorch Tensors and move to the training device
        images = torch.from_numpy(images).view(2, 3, 224, 224).float().to(device)
        conf_maps = torch.from_numpy(conf_maps).float().to(device).view(2,
                                    constants.num_joints,
                                    56, 56)
        pafs = torch.from_numpy(pafs).float().to(device).view(2,
                               constants.num_limbs,
                               2, 56, 56)
        
        # Perform a forward pass through the model
        outputs = model(images)
        # Outputs is a dictionary with 3 keys for each stage. Keys are 1, 2, 3.
        # Each key has another dictionary. Each sub-dictionary has 2 keys:
        # 'paf' or 'conf' for storing output of respective stage.
        
        # Define loss for both conf and paf individually. They will be equal to
        # sum of losses of all stages.
        loss_conf_total = 0
        loss_paf_total = 0
        for i in range(1, 4): # Three stages: 1, 2 and 3
            conf_out = outputs[i]['conf']
            paf_out = outputs[i]['paf']
            paf_out = paf_out.reshape(
                    paf_out.shape[0],
                    paf_out.shape[1] // 2,
                    2,
                    paf_out.shape[2],
                    paf_out.shape[3])
            loss_conf_total += criterion_conf(conf_out, conf_maps)
            loss_paf_total += criterion_paf(paf_out, pafs)
        # Calculate the total loss for both the outputs: PAF and Conf map.
        loss = loss_conf_total + loss_paf_total
        # Set grads to zero to prevent accumulation.
        optimizer.zero_grad()
        # Calculate the grads of all the parameters with respect to loss.
        loss.backward()
        # Take an optimization step.
        optimizer.step()
        # Add to running_loss
        running_loss += loss.item()
        # Append losses to the optimizer dictionary.
        losses.append(loss.item())
        
        batch_num += 1
        
        # Print statistics
        if batch_num % print_every == 0:
            print(f'Batch number: {batch_num}\tAverage loss:' \
                  f'{running_loss/batch_num}')
            running_loss = 0
    print('Finished epoch.')
    
    return model, optimizer, losses
        
        
        
    