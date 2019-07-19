# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 15:04:52 2019

@author: myidispg
"""
import torch
import torch.nn as nn
import torchvision.models as models

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
            vgg = models.vgg19(pretrained=True, progress=True)
            vgg = list(list(vgg.children())[0].children())[:33]
        else:
            print('Downloading pre-trained VGG 19 model...')
            vgg = models.vgg19_bn(pretrained=True, progress=True)
            vgg = list(list(model.children())[0].children())[:23]
        
        self.vgg = nn.Sequential(*vgg)
    
    def forward(self, image):
        features = self.vgg(image)
        
        return features
            
vgg_model1 = VGGFeatureExtractor(True).double()
features1 = vgg_model1.forward(torch.from_numpy(np.ones((1, 3, 224, 224))))

vgg_model2 = VGGFeatureExtractor(True).double()
features2 = vgg_model2.forward(torch.from_numpy(np.ones((1, 3, 224, 224))))


# Output features are of the shape: (batch_size, 521, 14, 14)
torch.all(torch.eq(features1, features2))