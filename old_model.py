# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 09:30:50 2019

@author: myidispg
"""

import cv2
import numpy as np

paf = np.random.rand(15, 2, 100, 100)
paf = paf.transpose(2,3,0,1)
paf = paf.reshape(paf.shape[0], paf.shape[1], paf.shape[2] * paf.shape[3])
paf = cv2.resize(paf, (46, 46),interpolation=cv2.INTER_CUBIC)

class OpenPoseModel(nn.Module):
    
    def __init__(self, num_limbs, num_joints):
        super(OpenPoseModel, self).__init__()
        
        self.vgg = VGGFeatureExtractor(False)
        
        # ------PAF BLOCK----------------------
        
        self.paf_block1_stage1 = ConvolutionalBlock(256)
        self.paf_block2_stage1 = ConvolutionalBlock(448)
        self.paf_block3_stage1 = ConvolutionalBlock(448)
        self.paf_block4_stage1 = ConvolutionalBlock(448)
        self.paf_block5_stage1 = ConvolutionalBlock(448)
        
        self.paf_conv1_stage1 = nn.Conv2d(448, 64, kernel_size=1)
        self.paf_conv2_stage1 = nn.Conv2d(64, num_limbs*2, kernel_size=1)
        
        self.paf_block1_stage2 = ConvolutionalBlock(286)
        self.paf_block2_stage2 = ConvolutionalBlock(448)
        self.paf_block3_stage2 = ConvolutionalBlock(448)
        self.paf_block4_stage2 = ConvolutionalBlock(448)
        self.paf_block5_stage2 = ConvolutionalBlock(448)
        
        self.paf_conv1_stage2 = nn.Conv2d(448, 64, kernel_size=1)
        self.paf_conv2_stage2 = nn.Conv2d(64, num_limbs*2, kernel_size=1)
        
        self.paf_block1_stage3 = ConvolutionalBlock(286)
        self.paf_block2_stage3 = ConvolutionalBlock(448)
        self.paf_block3_stage3 = ConvolutionalBlock(448)
        self.paf_block4_stage3 = ConvolutionalBlock(448)
        self.paf_block5_stage3 = ConvolutionalBlock(448)
        
        self.paf_conv1_stage3 = nn.Conv2d(448, 64, kernel_size=1)
        self.paf_conv2_stage3 = nn.Conv2d(64, num_limbs*2, kernel_size=1)
        
        self.paf_block1_stage4 = ConvolutionalBlock(286)
        self.paf_block2_stage4 = ConvolutionalBlock(448)
        self.paf_block3_stage4 = ConvolutionalBlock(448)
        self.paf_block4_stage4 = ConvolutionalBlock(448)
        self.paf_block5_stage4 = ConvolutionalBlock(448)
        
        self.paf_conv1_stage4 = nn.Conv2d(448, 64, kernel_size=1)
        self.paf_conv2_stage4 = nn.Conv2d(64, num_limbs*2, kernel_size=1)
        
        self.paf_block1_stage5 = ConvolutionalBlock(286)
        self.paf_block2_stage5 = ConvolutionalBlock(448)
        self.paf_block3_stage5 = ConvolutionalBlock(448)
        self.paf_block4_stage5 = ConvolutionalBlock(448)
        self.paf_block5_stage5 = ConvolutionalBlock(448)
        
        self.paf_conv1_stage5 = nn.Conv2d(448, 64, kernel_size=1)
        self.paf_conv2_stage5 = nn.Conv2d(64, num_limbs*2, kernel_size=1)
        
        # ---------CONFIDENCE MAPS BLOCK--------
        self.conf_block1_stage1 = ConvolutionalBlock(286)
        self.conf_block2_stage1 = ConvolutionalBlock(448)
        self.conf_block3_stage1 = ConvolutionalBlock(448)
        self.conf_block4_stage1 = ConvolutionalBlock(448)
        self.conf_block5_stage1 = ConvolutionalBlock(448)
        
        self.conf_conv1_stage1 = nn.Conv2d(448, 64, kernel_size=1)
        self.conf_conv2_stage1 = nn.Conv2d(64, num_joints, kernel_size=1)
        
        self.conf_block1_stage2 = ConvolutionalBlock(273)
        self.conf_block2_stage2 = ConvolutionalBlock(448)
        self.conf_block3_stage2 = ConvolutionalBlock(448)
        self.conf_block4_stage2 = ConvolutionalBlock(448)
        self.conf_block5_stage2 = ConvolutionalBlock(448)
        
        self.conf_conv1_stage2 = nn.Conv2d(448, 64, kernel_size=1)
        self.conf_conv2_stage2 = nn.Conv2d(64, num_joints, kernel_size=1)
        
        self.conf_block1_stage3 = ConvolutionalBlock(273)
        self.conf_block2_stage3 = ConvolutionalBlock(448)
        self.conf_block3_stage3 = ConvolutionalBlock(448)
        self.conf_block4_stage3 = ConvolutionalBlock(448)
        self.conf_block5_stage3 = ConvolutionalBlock(448)
        
        self.conf_conv1_stage3 = nn.Conv2d(448, 64, kernel_size=1)
        self.conf_conv2_stage3 = nn.Conv2d(64, num_joints, kernel_size=1)
        
        self.conf_block1_stage4 = ConvolutionalBlock(273)
        self.conf_block2_stage4 = ConvolutionalBlock(448)
        self.conf_block3_stage4 = ConvolutionalBlock(448)
        self.conf_block4_stage4 = ConvolutionalBlock(448)
        self.conf_block5_stage4 = ConvolutionalBlock(448)
        
        self.conf_conv1_stage4 = nn.Conv2d(448, 64, kernel_size=1)
        self.conf_conv2_stage4 = nn.Conv2d(64, num_joints, kernel_size=1)
        
        self.conf_block1_stage5 = ConvolutionalBlock(273)
        self.conf_block2_stage5 = ConvolutionalBlock(448)
        self.conf_block3_stage5 = ConvolutionalBlock(448)
        self.conf_block4_stage5 = ConvolutionalBlock(448)
        self.conf_block5_stage5 = ConvolutionalBlock(448)
        
        self.conf_conv1_stage5 = nn.Conv2d(448, 64, kernel_size=1)
        self.conf_conv2_stage5 = nn.Conv2d(64, num_joints, kernel_size=1)
        
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
    
    def forward_stage_4_pafs(self, input_data):
        out = F.relu(self.paf_block1_stage4(input_data), inplace = True)
        out = F.relu(self.paf_block2_stage4(out), inplace = True)
        out = F.relu(self.paf_block3_stage4(out), inplace = True)
        out = F.relu(self.paf_block4_stage4(out), inplace = True)
        out = F.relu(self.paf_block5_stage4(out), inplace = True)
        out = F.relu(self.paf_conv1_stage4(out), inplace = True)
        return self.paf_conv2_stage4(out)
      
    def forward_stage_5_pafs(self, input_data):
        out = F.relu(self.paf_block1_stage5(input_data), inplace = True)
        out = F.relu(self.paf_block2_stage5(out), inplace = True)
        out = F.relu(self.paf_block3_stage5(out), inplace = True)
        out = F.relu(self.paf_block4_stage5(out), inplace = True)
        out = F.relu(self.paf_block5_stage5(out), inplace = True)
        out = F.relu(self.paf_conv1_stage5(out), inplace = True)
        return self.paf_conv2_stage5(out)
    
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
      
    def forward_stage_4_conf(self, input_data):
        out = F.relu(self.conf_block1_stage4(input_data), inplace=True)
        out = F.relu(self.conf_block2_stage4(out), inplace=True)
        out = F.relu(self.conf_block3_stage4(out), inplace=True)
        out = F.relu(self.conf_block4_stage4(out), inplace=True)
        out = F.relu(self.conf_block5_stage4(out), inplace=True)
        out = F.relu(self.conf_conv1_stage4(out), inplace=True)
        return self.conf_conv2_stage4(out)
      
    def forward_stage_5_conf(self, input_data):
        out = F.relu(self.conf_block1_stage5(input_data), inplace=True)
        out = F.relu(self.conf_block2_stage5(out), inplace=True)
        out = F.relu(self.conf_block3_stage5(out), inplace=True)
        out = F.relu(self.conf_block4_stage5(out), inplace=True)
        out = F.relu(self.conf_block5_stage5(out), inplace=True)
        out = F.relu(self.conf_conv1_stage5(out), inplace=True)
        return self.conf_conv2_stage5(out)
        
    def forward(self, x):
#        print(f'Input shape: {x.shape}')
        
        vgg_out = self.vgg(x)
#        print(f'VGG Output shape: {vgg_out.shape}')        
        
        outputs = {1: {'paf': None, 'conf': None},
                   2: {'paf': None, 'conf': None},
                   3: {'paf': None, 'conf': None},
                   4: {'paf': None, 'conf': None},
                   5: {'paf': None, 'conf': None},}
        
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
        
        out_stage4_pafs = self.forward_stage_4_pafs(out_stage3_pafs)
        outputs[4]['paf'] = out_stage4_pafs
        out_stage4_pafs = torch.cat([out_stage4_pafs, vgg_out], 1)
        
        out_stage5_pafs = self.forward_stage_5_pafs(out_stage4_pafs)
        outputs[5]['paf'] = out_stage5_pafs
        out_stage5_pafs = torch.cat([out_stage5_pafs, vgg_out], 1)
      
#        # CONF BLOCK
        out_stage1_conf = self.forward_stage_1_conf(out_stage5_pafs)
        outputs[1]['conf'] = out_stage1_conf
        out_stage1_conf = torch.cat([out_stage1_conf, vgg_out], 1)
        
        out_stage2_conf = self.forward_stage_2_conf(out_stage1_conf)
        outputs[2]['conf'] = out_stage2_conf
        out_stage2_conf = torch.cat([out_stage2_conf, vgg_out], 1)
        
        out_stage3_conf = self.forward_stage_3_conf(out_stage2_conf)
        outputs[3]['conf'] = out_stage3_conf
        out_stage3_conf = torch.cat([out_stage3_conf, vgg_out], 1)
        
        out_stage4_conf = self.forward_stage_4_conf(out_stage3_conf)
        outputs[4]['conf'] = out_stage4_conf
        out_stage4_conf = torch.cat([out_stage4_conf, vgg_out], 1)
        
        out_stage5_conf = self.forward_stage_5_conf(out_stage4_conf)
        outputs[5]['conf'] = out_stage5_conf
        
        return outputs