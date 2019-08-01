# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 19:21:54 2019

@author: myidispg
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vgg_model import VGGFeatureExtractor

class ConvolutionalBlock(nn.Module):
    
    def __init__(self, input_channels, output_channels=64):
        super(ConvolutionalBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, output_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        
        output1 = F.relu(self.conv1(x), inplace=True)
        output2 = F.relu(self.conv2(output1), inplace=True)
        output3 = F.relu(self.conv3(output2), inplace=True)
        
        output3 = torch.cat([output1, output2, output3], 1)
        
        return output3

class OpenPoseModel(nn.Module):
  
  def __init__(self, num_limbs, num_joints):
    super(OpenPoseModel, self).__init__()
    
    self.num_limb_joint_features = (num_limbs*2) + num_joints + 256
    
    self.vgg = VGGFeatureExtractor()
    
    # Stage 1
    self.paf_1_stage_1 = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)
    self.paf_2_stage_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
    self.paf_3_stage_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
    self.paf_4_stage_1 = nn.Conv2d(64, 64, kernel_size=1)
    self.paf_5_stage_1 = nn.Conv2d(64, num_limbs*2, kernel_size=1)
    
    self.conf_1_stage_1 = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)
    self.conf_2_stage_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
    self.conf_3_stage_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
    self.conf_4_stage_1 = nn.Conv2d(64, 64, kernel_size=1)
    self.conf_5_stage_1 = nn.Conv2d(64, num_joints, kernel_size=1)
    
    # Stage 2
    self.paf_1_stage_2 = nn.Conv2d(self.num_limb_joint_features, 64, kernel_size=7, stride=1, padding=3)
    self.paf_2_stage_2 = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3)
    self.paf_3_stage_2 = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3)
    self.paf_4_stage_2 = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3)
    self.paf_5_stage_2 = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3)
    self.paf_6_stage_2 = nn.Conv2d(64, 64, kernel_size=1)
    self.paf_7_stage_2 = nn.Conv2d(64, num_limbs * 2, kernel_size=1)
    
    self.conf_1_stage_2 = nn.Conv2d(self.num_limb_joint_features, 64, kernel_size=7, stride=1, padding=3)
    self.conf_2_stage_2 = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3)
    self.conf_3_stage_2 = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3)
    self.conf_4_stage_2 = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3)
    self.conf_5_stage_2 = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3)
    self.conf_6_stage_2 = nn.Conv2d(64, 64, kernel_size=1)
    self.conf_7_stage_2 = nn.Conv2d(64, num_joints, kernel_size=1)
    
    # Stage 3
    self.paf_1_stage_3 = nn.Conv2d(self.num_limb_joint_features, 64, kernel_size=7, stride=1, padding=3)
    self.paf_2_stage_3 = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3)
    self.paf_3_stage_3 = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3)
    self.paf_4_stage_3 = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3)
    self.paf_5_stage_3 = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3)
    self.paf_6_stage_3 = nn.Conv2d(64, 64, kernel_size=1)
    self.paf_7_stage_3 = nn.Conv2d(64, num_limbs * 2, kernel_size=1)
    
    self.conf_1_stage_3 = nn.Conv2d(self.num_limb_joint_features, 64, kernel_size=7, stride=1, padding=3)
    self.conf_2_stage_3 = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3)
    self.conf_3_stage_3 = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3)
    self.conf_4_stage_3 = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3)
    self.conf_5_stage_3 = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3)
    self.conf_6_stage_3 = nn.Conv2d(64, 64, kernel_size=1)
    self.conf_7_stage_3 = nn.Conv2d(64, num_joints, kernel_size=1)
    
    # Stage 4
    self.paf_1_stage_4 = nn.Conv2d(self.num_limb_joint_features, 64, kernel_size=7, stride=1, padding=3)
    self.paf_2_stage_4 = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3)
    self.paf_3_stage_4 = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3)
    self.paf_4_stage_4 = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3)
    self.paf_5_stage_4 = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3)
    self.paf_6_stage_4 = nn.Conv2d(64, 64, kernel_size=1)
    self.paf_7_stage_4 = nn.Conv2d(64, num_limbs * 2, kernel_size=1)
    
    self.conf_1_stage_4 = nn.Conv2d(self.num_limb_joint_features, 64, kernel_size=7, stride=1, padding=3)
    self.conf_2_stage_4 = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3)
    self.conf_3_stage_4 = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3)
    self.conf_4_stage_4 = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3)
    self.conf_5_stage_4 = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3)
    self.conf_6_stage_4 = nn.Conv2d(64, 64, kernel_size=1)
    self.conf_7_stage_4 = nn.Conv2d(64, num_joints, kernel_size=1)
    
    # Stage 5
    self.paf_1_stage_5 = nn.Conv2d(self.num_limb_joint_features, 64, kernel_size=7, stride=1, padding=3)
    self.paf_2_stage_5 = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3)
    self.paf_3_stage_5 = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3)
    self.paf_4_stage_5 = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3)
    self.paf_5_stage_5 = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3)
    self.paf_6_stage_5 = nn.Conv2d(64, 64, kernel_size=1)
    self.paf_7_stage_5 = nn.Conv2d(64, num_limbs * 2, kernel_size=1)
    
    self.conf_1_stage_5 = nn.Conv2d(self.num_limb_joint_features, 64, kernel_size=7, stride=1, padding=3)
    self.conf_2_stage_5 = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3)
    self.conf_3_stage_5 = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3)
    self.conf_4_stage_5 = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3)
    self.conf_5_stage_5 = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3)
    self.conf_6_stage_5 = nn.Conv2d(64, 64, kernel_size=1)
    self.conf_7_stage_5 = nn.Conv2d(64, num_joints, kernel_size=1)
    
  def forward_stage_1_paf(self, input):
    out = F.relu(self.paf_1_stage_1(input), inplace=True)
    out = F.relu(self.paf_2_stage_1(out), inplace=True)
    out = F.relu(self.paf_3_stage_1(out), inplace=True)
    out = F.relu(self.paf_4_stage_1(out), inplace=True)
    out = F.relu(self.paf_5_stage_1(out), inplace=True)
    
    return out
  
  def forward_stage_1_conf(self, input):
    out = F.relu(self.conf_1_stage_1(input), inplace=True)
    out = F.relu(self.conf_2_stage_1(out), inplace=True)
    out = F.relu(self.conf_3_stage_1(out), inplace=True)
    out = F.relu(self.conf_4_stage_1(out), inplace=True)
    out = F.relu(self.conf_5_stage_1(out), inplace=True)
    
    return out
  
  def forward_stage_2_paf(self, input):
    out = F.relu(self.paf_1_stage_2(input), inplace=True)
    out = F.relu(self.paf_2_stage_2(out), inplace=True)
    out = F.relu(self.paf_3_stage_2(out), inplace=True)
    out = F.relu(self.paf_4_stage_2(out), inplace=True)
    out = F.relu(self.paf_5_stage_2(out), inplace=True)
    out = F.relu(self.paf_6_stage_2(out), inplace=True)
    out = F.relu(self.paf_7_stage_2(out), inplace=True)
    
    return out
  
  def forward_stage_3_paf(self, input):
    out = F.relu(self.paf_1_stage_3(input), inplace=True)
    out = F.relu(self.paf_2_stage_3(out), inplace=True)
    out = F.relu(self.paf_3_stage_3(out), inplace=True)
    out = F.relu(self.paf_4_stage_3(out), inplace=True)
    out = F.relu(self.paf_5_stage_3(out), inplace=True)
    out = F.relu(self.paf_6_stage_3(out), inplace=True)
    out = F.relu(self.paf_7_stage_3(out), inplace=True)
    
    return out
  
  def forward_stage_4_paf(self, input):
    out = F.relu(self.paf_1_stage_4(input), inplace=True)
    out = F.relu(self.paf_2_stage_4(out), inplace=True)
    out = F.relu(self.paf_3_stage_4(out), inplace=True)
    out = F.relu(self.paf_4_stage_4(out), inplace=True)
    out = F.relu(self.paf_5_stage_4(out), inplace=True)
    out = F.relu(self.paf_6_stage_4(out), inplace=True)
    out = F.relu(self.paf_7_stage_4(out), inplace=True)
    
    return out
  
  def forward_stage_5_paf(self, input):
    out = F.relu(self.paf_1_stage_5(input), inplace=True)
    out = F.relu(self.paf_2_stage_5(out), inplace=True)
    out = F.relu(self.paf_3_stage_5(out), inplace=True)
    out = F.relu(self.paf_4_stage_5(out), inplace=True)
    out = F.relu(self.paf_5_stage_5(out), inplace=True)
    out = F.relu(self.paf_6_stage_5(out), inplace=True)
    out = F.relu(self.paf_7_stage_5(out), inplace=True)
    
    return out
  
  def forward_stage_2_conf(self, input):
    out = F.relu(self.conf_1_stage_2(input), inplace=True)
    out = F.relu(self.conf_2_stage_2(out), inplace=True)
    out = F.relu(self.conf_3_stage_2(out), inplace=True)
    out = F.relu(self.conf_4_stage_2(out), inplace=True)
    out = F.relu(self.conf_5_stage_2(out), inplace=True)
    out = F.relu(self.conf_6_stage_2(out), inplace=True)
    out = F.relu(self.conf_7_stage_2(out), inplace=True)
    
    return out
  
  def forward_stage_3_conf(self, input):
    out = F.relu(self.conf_1_stage_3(input), inplace=True)
    out = F.relu(self.conf_2_stage_3(out), inplace=True)
    out = F.relu(self.conf_3_stage_3(out), inplace=True)
    out = F.relu(self.conf_4_stage_3(out), inplace=True)
    out = F.relu(self.conf_5_stage_3(out), inplace=True)
    out = F.relu(self.conf_6_stage_3(out), inplace=True)
    out = F.relu(self.conf_7_stage_3(out), inplace=True)
    
    return out
    
  def forward_stage_4_conf(self, input):
    out = F.relu(self.conf_1_stage_4(input), inplace=True)
    out = F.relu(self.conf_2_stage_4(out), inplace=True)
    out = F.relu(self.conf_3_stage_4(out), inplace=True)
    out = F.relu(self.conf_4_stage_4(out), inplace=True)
    out = F.relu(self.conf_5_stage_4(out), inplace=True)
    out = F.relu(self.conf_6_stage_4(out), inplace=True)
    out = F.relu(self.conf_7_stage_4(out), inplace=True)
    
    return out
  
  def forward_stage_5_conf(self, input):
    out = F.relu(self.conf_1_stage_5(input), inplace=True)
    out = F.relu(self.conf_2_stage_5(out), inplace=True)
    out = F.relu(self.conf_3_stage_5(out), inplace=True)
    out = F.relu(self.conf_4_stage_5(out), inplace=True)
    out = F.relu(self.conf_5_stage_5(out), inplace=True)
    out = F.relu(self.conf_6_stage_5(out), inplace=True)
    out = F.relu(self.conf_7_stage_5(out), inplace=True)
    
    return out
  
  def forward(self, x):
    vgg_out = self.vgg(x)
    
    outputs = {1: {'paf': None, 'conf': None},
                   2: {'paf': None, 'conf': None},
                   3: {'paf': None, 'conf': None},
                   4: {'paf': None, 'conf': None},
                   5: {'paf': None, 'conf': None},}
    
    # Stage 1
    out_paf = self.forward_stage_1_paf(vgg_out)
    out_conf = self.forward_stage_1_conf(vgg_out)
    outputs[1]['paf'] = out_paf
    outputs[1]['conf'] = out_conf
    
    # Stage 2
    out_paf = self.forward_stage_2_paf(torch.cat([out_paf, out_conf, vgg_out], 1))
    out_conf = self.forward_stage_2_conf(torch.cat([out_paf, out_conf, vgg_out], 1))
    outputs[2]['paf'] = out_paf
    outputs[2]['conf'] = out_conf
    
    # Stage 3
    out_paf = self.forward_stage_3_paf(torch.cat([out_paf, out_conf, vgg_out], 1))
    out_conf = self.forward_stage_3_conf(torch.cat([out_paf, out_conf, vgg_out], 1))
    outputs[3]['paf'] = out_paf
    outputs[3]['conf'] = out_conf
    
    # Stage 4
    out_paf = self.forward_stage_4_paf(torch.cat([out_paf, out_conf, vgg_out], 1))
    out_conf = self.forward_stage_4_conf(torch.cat([out_paf, out_conf, vgg_out], 1))
    outputs[4]['paf'] = out_paf
    outputs[4]['conf'] = out_conf
    
    # Stage 5
    out_paf = self.forward_stage_5_paf(torch.cat([out_paf, out_conf, vgg_out], 1))
    out_conf = self.forward_stage_5_conf(torch.cat([out_paf, out_conf, vgg_out], 1))
    outputs[5]['paf'] = out_paf
    outputs[5]['conf'] = out_conf
    
    return outputs
#
#model = OpenPoseModel(15, 17).to(torch.device('cuda'))
#outputs = model(torch.from_numpy(np.ones((2, 3, 368, 368))).float().to(torch.device('cuda')))
#for j in range(10):
#    print(f'Round {j+1}')
#    outputs = model(torch.from_numpy(np.ones((1, 3, 400, 400))).float().to(torch.device('cuda')))
#    for i in range(1, 4):
#        print(f'Stage {i} paf: {outputs[i]["paf"].shape}')
#        print(f'Stage {i} conf: {outputs[i]["conf"].shape}')
#    print()
#model = OpenPoseModel(15, 17).to(torch.device('cuda'))
#gpu_memory = torch.cuda.memory_allocated()
#gpu_memory /= 1024*1024
#print(f'GPU memory used before training: {gpu_memory}MB')
#for i in range(10):
#    print(i+1)
#    outputs = model(torch.from_numpy(np.ones((2, 3, 224, 224))).float().to(torch.device('cuda')))
#    print(f"Stage 1 paf: {outputs[1]['paf'].shape}")
#    print(f"Stage 2 paf: {outputs[2]['paf'].shape}")
#    print(f"Stage 3 paf: {outputs[3]['paf'].shape}")
#    print(f"Stage 1 conf: {outputs[1]['conf'].shape}")
#    print(f"Stage 2 conf: {outputs[2]['conf'].shape}")
#    print(f"Stage 3 conf: {outputs[3]['conf'].shape}")
#    gpu_memory = torch.cuda.memory_allocated()
#    gpu_memory /= 1024*1024
#    print(f'GPU memory used at step {i+1} is: {gpu_memory}')
#    print()
#        
