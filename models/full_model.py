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
    
    def __init__(self, input_channels, output_channels=128):
        super(ConvolutionalBlock, self).__init__()
        
        
        
        self.conv1 = nn.Conv2d(input_channels, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, output_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
#        print(f'input shape: {x.shape}')
        
        output1 = F.relu(self.conv1(x), inplace=True)
#        print(f'output1 shape: {output1.shape}')
        output2 = F.relu(self.conv2(output1), inplace=True)
#        print(f'output2 shape: {output2.shape}')
        output3 = F.relu(self.conv3(output2), inplace=True)
#        print(f'output3 shape: {output3.shape}')
        
        output3 = torch.cat([output1, output2, output3], 1)
#        print(f'output3 shape after concat: {output3.shape}')
        
        return output3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#model = ConvolutionalBlock(512).to(device)
#output = model(torch.from_numpy(np.ones((8, 512, 14, 14))).float().to(device))

class OpenPoseModel(nn.Module):
    
    def __init__(self, input_channels, num_limbs, num_parts, num_stages):
        super(OpenPoseModel, self).__init__()
        
        self.vgg = VGGFeatureExtractor(False)
        
        self.num_stages = num_stages
        
        # ------PAF BLOCK----------------------
        self.paf_block1_stage1 = ConvolutionalBlock(256)
        
        self.paf_block2_stage1 = ConvolutionalBlock(512)
        self.paf_block3_stage1 = ConvolutionalBlock(512)
        self.paf_block4_stage1 = ConvolutionalBlock(512)
        self.paf_block5_stage1 = ConvolutionalBlock(512)
        
        self.paf_conv1_stage1 = nn.Conv2d(512, 32, kernel_size=1)
        self.paf_conv2_stage1 = nn.Conv2d(32, num_limbs, kernel_size=1)
        
        self.paf_block1_stage2 = ConvolutionalBlock(273)
        self.paf_block2_stage2 = ConvolutionalBlock(512)
        self.paf_block3_stage2 = ConvolutionalBlock(512)
        self.paf_block4_stage2 = ConvolutionalBlock(512)
        self.paf_block5_stage2 = ConvolutionalBlock(512)
        
        self.paf_conv1_stage2 = nn.Conv2d(512, 32, kernel_size=1)
        self.paf_conv2_stage2 = nn.Conv2d(32, num_limbs, kernel_size=1)
        
        self.paf_block1_stage3 = ConvolutionalBlock(529)
        self.paf_block2_stage3 = ConvolutionalBlock(512)
        self.paf_block3_stage3 = ConvolutionalBlock(512)
        self.paf_block4_stage3 = ConvolutionalBlock(512)
        self.paf_block5_stage3 = ConvolutionalBlock(512)
        
        self.paf_conv1_stage3 = nn.Conv2d(512, 32, kernel_size=1)
        self.paf_conv2_stage3 = nn.Conv2d(32, num_limbs, kernel_size=1)
        
        # ---------CONFIDENCE MAPS BLOCK--------
        self.conf_block1 = ConvolutionalBlock(273)
        self.conf_block2 = ConvolutionalBlock(512)
        self.conf_block3 = ConvolutionalBlock(512)
        self.conf_block4 = ConvolutionalBlock(512)
        self.conf_block5 = ConvolutionalBlock(512)
        
        self.conf_conv1 = nn.Conv2d(512, 32, kernel_size=1)
        self.conf_conv2 = nn.Conv2d(32, num_limbs, kernel_size=1)
        
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
    
    def forward(self, x):
        print(f'Input shape: {x.shape}')
        
        vgg_out = self.vgg(x)
        print(f'VGG Output shape: {vgg_out.shape}')        
        
        pafs = dict()
        
        # PAF BLOCK
        out_stage1_pafs = self.forward_stage_1_pafs(vgg_out)
        pafs[1] = out_stage1_pafs
        out_stage1_pafs = torch.cat([out_stage1_pafs, vgg_out], 1)
        
        out_stage2_pafs = self.forward_stage_2_pafs(out_stage1_pafs)
        pafs[2] = out_stage2_pafs
        out_stage2_pafs = torch.cat([out_stage1_pafs, vgg_out], 1)
        
        out_stage3_pafs = self.forward_stage_3_pafs(out_stage2_pafs)
        pafs[3] = out_stage3_pafs
#        x = torch.cat([x, vgg_out], 1)
#        
#        # CONF BLOCK
#        x = self.conf_block1(x)
#        x = self.conf_block2(x)
#        x = self.conf_block3(x)
#        x = self.conf_block4(x)
#        x = self.conf_block5(x)
#        
#        x = self.conf_conv1(x)
#        x = self.conf_conv2(x)
        
        return out_stage3_pafs, pafs

model = OpenPoseModel(512, 17, 15, 3).to(device)
output, pafs = model(torch.from_numpy(np.ones((1, 3, 224, 224))).float().to(device))
print(output.shape)
print(pafs[1].shape)
        
