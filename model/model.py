# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 15:04:52 2019

@author: myidispg
"""
import torch
import torchvision.models as models

model = models.vgg19_bn(pretrained=True, progress=True)

model.features[:10]

for i in range(10):
    print(model.parameters())
    
list(model.parameters())