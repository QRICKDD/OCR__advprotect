import torch
import torch.nn as nn
from Propocess.DIFFJPEG import compression, decompression
from Propocess.DIFFJPEG.utils import diff_round, quality_to_factor


class DiffJPEG(nn.Module):
    def __init__(self, height, width, differentiable=True, quality=80):
        # Initialize the DiffJPEG layer
        # Inputs:
        #     height(int): Original image height
        #     width(int): Original image width
        #     differentiable(bool): If true uses custom differentiable
        #         rounding function, if false uses standard torch.round
        #     quality(float): Quality factor for jpeg compression scheme.
        #
        super(DiffJPEG, self).__init__()
        if differentiable:
            rounding = diff_round
        else:
            rounding = torch.round
        factor = quality_to_factor(quality)
        self.compress = compression.compress_jpeg(rounding=rounding, factor=factor)
        self.decompress = decompression.decompress_jpeg(height, width, rounding=rounding, factor=factor)

    def forward(self, x):
        y, cb, cr = self.compress(x)
        recovered = self.decompress(y, cb, cr)
        return recovered


def diff_jpeg():
    from AllConfig.GConfig import test_img_path
    from Tools.Imagebasetool import img_read, img_grad_show
    img = img_read(test_img_path)
    img = img.unsqueeze_(0)
    h, w = img.shape[2:]
    # initlize DiffJPEG with quality=80
    dj80 = DiffJPEG(height=h, width=w, quality=80)

    img.requires_grad = True
    img_compress = dj80(img)

    # show grad
    img_grad_show(img_compress)


if __name__ == '__main__':
    diff_jpeg()
