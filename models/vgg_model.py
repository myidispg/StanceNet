# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 15:04:52 2019

@author: myidispg
"""
import numpy as np

import torch.nn as nn
import torchvision.models as models

# Extract the first 10 layers of the VGG-19 model.
class VGGFeatureExtractor(nn.Module):
    def __init__(self, batch_normalization=True):
        """
        Download and use the first 10 layers of the pre-trained VGG-19 model.
        """
        
        super(VGGFeatureExtractor, self).__init__()
        
        print('Downloading pre-trained VGG 19 model...')
        vgg = models.vgg19_bn(pretrained=True)
        vgg = list(list(vgg.children())[0].children())[:23]
        
        self.vgg = nn.Sequential(*vgg)
    
    def forward(self, image):
        features = self.vgg(image)
        
        return features
