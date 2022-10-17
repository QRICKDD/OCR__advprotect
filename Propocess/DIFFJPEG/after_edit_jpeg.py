import torch
import torch.nn as nn
from Propocess.DIFFJPEG.modules import modules_compression,modules_decompression
from utils import diff_round, quality_to_factor
import cv2
import matplotlib.pyplot as plt
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class DiffJPEG(nn.Module):
    def __init__(self, height=480, width=640, differentiable=True, quality=70):
        ''' Initialize the DiffJPEG layer
        Inputs:
            height(int): Original image hieght
            width(int): Original image width
            differentiable(bool): If true uses custom differentiable
                rounding function, if false uses standrard torch.round
            quality(float): Quality factor for jpeg compression scheme.
        '''
        super(DiffJPEG, self).__init__()
        if differentiable:
            rounding = diff_round
        else:
            rounding = torch.round
        factor = quality_to_factor(quality)
        self.compress = modules_compression.compress_jpeg(rounding=rounding, factor=factor)
        self.decompress = modules_decompression.decompress_jpeg(height, width, rounding=rounding, factor=factor)

    def forward(self, x):
        y, cb, cr = self.compress(x)
        recovered = self.decompress(y, cb, cr)
        return recovered

weight = 640
height = 480
image_path = '/home/luning/luning_code/luning/code/picture/trian/59.jpg'
