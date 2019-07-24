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
        
        self.vgg = VGGFeatureExtractor(False)
        
        # ------PAF BLOCK----------------------
        self.paf_block1_stage1 = ConvolutionalBlock(256)
        
        self.paf_block2_stage1 = ConvolutionalBlock(128)
        self.paf_block3_stage1 = ConvolutionalBlock(128)
        self.paf_block4_stage1 = ConvolutionalBlock(128)
        self.paf_block5_stage1 = ConvolutionalBlock(128)
        
        self.paf_conv1_stage1 = nn.Conv2d(128, 32, kernel_size=1)
        self.paf_conv2_stage1 = nn.Conv2d(32, num_limbs*2, kernel_size=1)
        
        self.paf_block1_stage2 = ConvolutionalBlock(286)
        self.paf_block2_stage2 = ConvolutionalBlock(128)
        self.paf_block3_stage2 = ConvolutionalBlock(128)
        self.paf_block4_stage2 = ConvolutionalBlock(128)
        self.paf_block5_stage2 = ConvolutionalBlock(128)
        
        self.paf_conv1_stage2 = nn.Conv2d(128, 32, kernel_size=1)
        self.paf_conv2_stage2 = nn.Conv2d(32, num_limbs*2, kernel_size=1)
        
        self.paf_block1_stage3 = ConvolutionalBlock(286)
        self.paf_block2_stage3 = ConvolutionalBlock(128)
        self.paf_block3_stage3 = ConvolutionalBlock(128)
        self.paf_block4_stage3 = ConvolutionalBlock(128)
        self.paf_block5_stage3 = ConvolutionalBlock(128)
        
        self.paf_conv1_stage3 = nn.Conv2d(128, 32, kernel_size=1)
        self.paf_conv2_stage3 = nn.Conv2d(32, num_limbs*2, kernel_size=1)
        
        # ---------CONFIDENCE MAPS BLOCK--------
        self.conf_block1_stage1 = ConvolutionalBlock(286)
        self.conf_block2_stage1 = ConvolutionalBlock(128)
        self.conf_block3_stage1 = ConvolutionalBlock(128)
        self.conf_block4_stage1 = ConvolutionalBlock(128)
        self.conf_block5_stage1 = ConvolutionalBlock(128)
        
        self.conf_conv1_stage1 = nn.Conv2d(128, 32, kernel_size=1)
        self.conf_conv2_stage1 = nn.Conv2d(32, num_joints, kernel_size=1)
        
        self.conf_block1_stage2 = ConvolutionalBlock(273)
        self.conf_block2_stage2 = ConvolutionalBlock(128)
        self.conf_block3_stage2 = ConvolutionalBlock(128)
        self.conf_block4_stage2 = ConvolutionalBlock(128)
        self.conf_block5_stage2 = ConvolutionalBlock(128)
        
        self.conf_conv1_stage2 = nn.Conv2d(128, 32, kernel_size=1)
        self.conf_conv2_stage2 = nn.Conv2d(32, num_joints, kernel_size=1)
        
        self.conf_block1_stage3 = ConvolutionalBlock(273)
        self.conf_block2_stage3 = ConvolutionalBlock(128)
        self.conf_block3_stage3 = ConvolutionalBlock(128)
        self.conf_block4_stage3 = ConvolutionalBlock(128)
        self.conf_block5_stage3 = ConvolutionalBlock(128)
        
        self.conf_conv1_stage3 = nn.Conv2d(128, 32, kernel_size=1)
        self.conf_conv2_stage3 = nn.Conv2d(32, num_joints, kernel_size=1)
        
    def forward_stage_1_pafs(self, input_data):
        out = F.relu(self.paf_block1_stage1(input_data), inplace = True)
        out = F.relu(self.paf_block2_stage1(out), inplace = True)
        out = F.relu(self.paf_block3_stage1(out), inplace = True)
        out = F.relu(self.paf_block4_stage1(out), inplace = True)
        out = F.relu(self.paf_block5_stage1(out), inplace = True)
        out = F.relu(self.paf_conv1_stage1(out), inplace = True)
        return self.paf_conv2_stage1(out)
    
    def forward_stage_2_pafs(self, input_data):
        out = F.relu(self.paf_block1_stage2(input_data), inplace = True)
        out = F.relu(self.paf_block2_stage2(out), inplace = True)
        out = F.relu(self.paf_block3_stage2(out), inplace = True)
        out = F.relu(self.paf_block4_stage2(out), inplace = True)
        out = F.relu(self.paf_block5_stage2(out), inplace = True)
        out = F.relu(self.paf_conv1_stage2(out), inplace = True)
        return self.paf_conv2_stage2(out)
        
    def forward_stage_3_pafs(self, input_data):
        out = F.relu(self.paf_block1_stage3(input_data), inplace = True)
        out = F.relu(self.paf_block2_stage3(out), inplace = True)
        out = F.relu(self.paf_block3_stage3(out), inplace = True)
        out = F.relu(self.paf_block4_stage3(out), inplace = True)
        out = F.relu(self.paf_block5_stage3(out), inplace = True)
        out = F.relu(self.paf_conv1_stage3(out), inplace = True)
        return self.paf_conv2_stage3(out)
    
    def forward_stage_1_conf(self, input_data):
        out = F.relu(self.conf_block1_stage1(input_data), inplace=True)
        out = F.relu(self.conf_block2_stage1(out), inplace=True)
        out = F.relu(self.conf_block3_stage1(out), inplace=True)
        out = F.relu(self.conf_block4_stage1(out), inplace=True)
        out = F.relu(self.conf_block5_stage1(out), inplace=True)
        out = F.relu(self.conf_conv1_stage1(out), inplace=True)
        return self.conf_conv2_stage1(out)
    
    def forward_stage_2_conf(self, input_data):
        out = F.relu(self.conf_block1_stage2(input_data), inplace=True)
        out = F.relu(self.conf_block2_stage2(out), inplace=True)
        out = F.relu(self.conf_block3_stage2(out), inplace=True)
        out = F.relu(self.conf_block4_stage2(out), inplace=True)
        out = F.relu(self.conf_block5_stage2(out), inplace=True)
        out = F.relu(self.conf_conv1_stage2(out), inplace=True)
        return self.conf_conv2_stage2(out)
    
    def forward_stage_3_conf(self, input_data):
        out = F.relu(self.conf_block1_stage3(input_data), inplace=True)
        out = F.relu(self.conf_block2_stage3(out), inplace=True)
        out = F.relu(self.conf_block3_stage3(out), inplace=True)
        out = F.relu(self.conf_block4_stage3(out), inplace=True)
        out = F.relu(self.conf_block5_stage3(out), inplace=True)
        out = F.relu(self.conf_conv1_stage3(out), inplace=True)
        return self.conf_conv2_stage3(out)
        
    def forward(self, x):
#        print(f'Input shape: {x.shape}')
        
        vgg_out = self.vgg(x)
#        print(f'VGG Output shape: {vgg_out.shape}')        
        
        outputs = {1: {'paf': None, 'conf': None},
                   2: {'paf': None, 'conf': None},
                   3: {'paf': None, 'conf': None}}
        
        # PAF BLOCK
        out_stage1_pafs = self.forward_stage_1_pafs(vgg_out)
        outputs[1]['paf'] = out_stage1_pafs
        out_stage1_pafs = torch.cat([out_stage1_pafs, vgg_out], 1)
        
        out_stage2_pafs = self.forward_stage_2_pafs(out_stage1_pafs)
        outputs[2]['paf'] = out_stage2_pafs
        out_stage2_pafs = torch.cat([out_stage2_pafs, vgg_out], 1)
        
        out_stage3_pafs = self.forward_stage_3_pafs(out_stage2_pafs)
        outputs[3]['paf'] = out_stage3_pafs
        out_stage3_pafs = torch.cat([out_stage3_pafs, vgg_out], 1)
#        
#        # CONF BLOCK
        out_stage1_conf = self.forward_stage_1_conf(out_stage3_pafs)
        outputs[1]['conf'] = out_stage1_conf
        out_stage1_conf = torch.cat([out_stage1_conf, vgg_out], 1)
        
        out_stage2_conf = self.forward_stage_2_conf(out_stage1_conf)
        outputs[2]['conf'] = out_stage2_conf
        out_stage2_conf = torch.cat([out_stage2_conf, vgg_out], 1)
        
        out_stage3_conf = self.forward_stage_3_conf(out_stage2_conf)
        outputs[3]['conf'] = out_stage3_conf
        
        return outputs

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
