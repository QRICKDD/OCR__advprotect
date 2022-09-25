import torch
from torch import nn
from model_DBnet.models.modules.shufflenetv2 import shufflenet_v2_x1_0
from model_DBnet.models.modules.segmentation_head import FPN,FPEM_FFM

backbone_dict = {
                 'shufflenetv2': {'models': shufflenet_v2_x1_0, 'out': [24, 116, 232, 464]}
                 }
"""
FPN 金字塔   FPEM 特征增强  FFM融合特征   PANnet结构，是PSEnet的优化版本

相关信息参考  https://blog.csdn.net/c991262331/article/details/109320811
"""
segmentation_head_dict = {'FPN': FPN, 'FPEM_FFM': FPEM_FFM}