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
    
    def __init__(self, input_channels, output_channels=32):
        super(ConvolutionalBlock, self).__init__()
        
        
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, output_channels, kernel_size=3, stride=1, padding=1)
    
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
    
    def __init__(self, input_channels, num_limbs):
        super(OpenPoseModel, self).__init__()
        
        self.vgg = VGGFeatureExtractor(False)
        
        self.paf_block1 = ConvolutionalBlock(256)
        self.paf_block2 = ConvolutionalBlock(96)
        self.paf_block3 = ConvolutionalBlock(96)
        self.paf_block4 = ConvolutionalBlock(96)
        self.paf_block5 = ConvolutionalBlock(96)
        
        self.paf_conv1 = nn.Conv2d(96, 32, kernel_size=1)
        self.paf_conv2 = nn.Conv2d(32, num_limbs, kernel_size=1)
        
    def forward(self, x):
        print(f'Input shape: {x.shape}')
        
        vgg_out = self.vgg(x)
        print(f'VGG Output shape: {vgg_out.shape}')        
        
        x = self.paf_block1(vgg_out)
        print('here')
        x = self.paf_block2(x)
        print('here')
        x = self.paf_block3(x)
        print('here')
        x = self.paf_block4(x)
        print('here')
        x = self.paf_block5(x)
        print('here')
        x = self.paf_conv1(x)
        print(f'After first 1x1 conv: {x.shape}')
        x = self.paf_conv2(x)
        print(f'After second 1x1 conv: {x.shape}')
        
        return x

model = OpenPoseModel(512, 17).to(device)
output = model(torch.from_numpy(np.ones((1, 3, 896, 896))).float().to(device))
        
