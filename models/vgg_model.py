# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 15:04:52 2019

@author: myidispg
"""
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vgg = models.vgg19(pretrained=True)
list(vgg.children())[0]

model = nn.Sequential(nn.ConvTranspose2d(17, 17, 55, 13)).to(device)
output = model(torch.from_numpy(np.ones((8, 17, 14, 14), dtype=np.float16)).float().to(device))
print(output.shape)

# Extract the first 10 layers of the VGG-19 model.
class VGGFeatureExtractor(nn.Module):
    def __init__(self, batch_normalization=True):
        """
        Download and use the first 10 layers of the pre-trained VGG-19 model.
        The model comes in two versions:
            1. No batch normalization.
            2. With batch normalization.
        Inputs:
            batch_normalization: boolean to use normalized or simple version.            
        """
        
        super(VGGFeatureExtractor, self).__init__()
        if batch_normalization:
            print('Downloading pre-trained VGG 19 model...')
            vgg = models.vgg19(pretrained=True)
            vgg = list(list(vgg.children())[0].children())[:11]
        else:
            print('Downloading pre-trained VGG 19 model...')
            vgg = models.vgg19_bn(pretrained=True)
            vgg = list(list(vgg.children())[0].children())[:23]
        
        self.vgg = nn.Sequential(*vgg)
    
    def forward(self, image):
        features = self.vgg(image)
        
        return features
            

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#import numpy as np

#torch.cuda.empty_cache()
#vgg_model1 = VGGFeatureExtractor(False).to(device)
#features1 = vgg_model1.forward(torch.from_numpy(np.ones((8, 3, 224, 224), dtype=np.float16)).float().to(device))
#print(features1.shape)


#torch.cuda.empty_cache()
# Output features are of the shape: (batch_size, 521, 14, 14)
