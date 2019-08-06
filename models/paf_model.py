# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 17:13:21 2019

@author: myidispg
"""

import torch.nn as nn
import torch
from models.helper import init, make_standard_block
from models.vgg_model import VGGFeatureExtractor

class StanceNet(nn.Module):
    def __init__(self, n_joints, n_limbs, n_stages=7):
        super(StanceNet, self).__init__()
        assert(n_stages > 0), "Number of stages cannot be less than 1."
        self.vgg = VGGFeatureExtractor()
        self.n_stages = n_stages
        stages = [Stage(128, n_joints, n_limbs, stage1 = True)]
        for i in range(n_stages - 1):
            stages.append(Stage(128, n_joints, n_limbs, False))
        self.stages = nn.ModuleList(stages)
        
    def forward(self, x):
        img_feats = self.vgg(x)
        concat_feats = img_feats
        outputs = dict()
        for i in range(self.n_stages):
            outputs[i] = dict()
            
        for i, stage in enumerate(self.stages):
            heatmap_out, paf_out = stage(concat_feats)
            outputs[i]['paf'] = paf_out
            outputs[i]['conf'] = heatmap_out
            concat_feats = torch.cat([img_feats, heatmap_out, paf_out], 1)
        return outputs
    
class Stage(nn.Module):
    def __init__(self, vgg_output_features, n_joints, n_limbs, stage1):
        super(Stage, self).__init__()
        input_features = vgg_output_features
        if stage1:
            self.block1 = make_paf_block_stage1(input_features, n_joints)
            self.block2 = make_paf_block_stage1(input_features, n_limbs)
        else:
            input_features = vgg_output_features + n_joints + n_limbs
            self.block1 = make_paf_block_stage2(input_features, n_joints)
            self.block2 = make_paf_block_stage2(input_features, n_limbs)
        
        init(self.block1)
        init(self.block2)
        
    def forward(self, x):
        heatmap = self.block1(x)
        paf = self.block2(x)
        
        return heatmap, paf

def make_paf_block_stage1(inp_feats, output_feats):
    layers = [make_standard_block(inp_feats, 128, 3),
              make_standard_block(128, 128, 3),
              make_standard_block(128, 128, 3),
              make_standard_block(128, 512, 1, 1, 0)]
    layers += [nn.Conv2d(512, output_feats, 1, 1, 0)]
    return nn.Sequential(*layers)


def make_paf_block_stage2(inp_feats, output_feats):
    layers = [make_standard_block(inp_feats, 128, 7, 1, 3),
              make_standard_block(128, 128, 7, 1, 3),
              make_standard_block(128, 128, 7, 1, 3),
              make_standard_block(128, 128, 7, 1, 3),
              make_standard_block(128, 128, 7, 1, 3),
              make_standard_block(128, 128, 1, 1, 0)]
    layers += [nn.Conv2d(128, output_feats, 1, 1, 0)]
    return nn.Sequential(*layers)
        

#import numpy as np
#
#model = StanceNet(17, 30, 3).to(torch.device('cuda'))
#input_ = torch.from_numpy(np.zeros((4, 3, 368, 368))).float().to(torch.device('cuda'))
#outputs = model(input_)
#
#for i in range(3):
#    print(f'{i}, paf: {outputs[i]["paf"].shape}')
#    print(f'{i}, conf: {outputs[i]["conf"].shape}')