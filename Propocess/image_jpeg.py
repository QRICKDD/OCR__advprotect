import torch
import torch.nn as nn
from Propocess.DIFFJPEG.modules import modules_compression,modules_decompression
from Propocess.DIFFJPEG.utils import diff_round, quality_to_factor



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

if __name__=='__main__':
    from AllConfig.GConfig import test_img_path
    from Tools.Imagebasetool import *
    img=img_read(test_img_path)
    img_show3(img.numpy())
    height,width = img.shape[-2], img.shape[-1]

    dj80 = DiffJPEG(height=height,width=width,quality=80)
    dj70 = DiffJPEG(height=height, width=width, quality=70)
    dj60 = DiffJPEG(height=height, width=width, quality=60)




