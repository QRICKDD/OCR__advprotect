import numpy as np
import torch
import random
from torchvision import transforms
from Tools.Imagebasetool import img_grad_show
from Propocess.DIFFJPEG import compression, decompression
from Propocess.DIFFJPEG.utils import diff_round, quality_to_factor
import torch.nn as nn

class DiffJPEG(nn.Module):
    def __init__(self, height=230, width=224, differentiable=True, quality=80):
        # Initialize the DiffJPEG layer
        # Inputs:
        #     height(int): Original image height
        #     width(int): Original image width
        #     differentiable(bool): If true uses custom differentiable
        #         rounding function, if false uses standard torch.round
        #     quality(float): Quality factor for jpeg compression scheme.
        #
        super(DiffJPEG, self).__init__()
        print(height,width)
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

def random_image_resize(image: torch.Tensor, low=0.25, high=3):
    assert (len(image.shape) == 4 and image.shape[0] == 1)
    #assert image.requires_grad == True
    scale = random.random()
    shape = image.shape
    h, w = shape[-2], shape[-1]
    h, w = int(h * (scale * (high - low) + low)), int(w * (scale * (high - low) + low))
    image = transforms.Resize([h, w])(image)
    return image

def repeat_4D(patch: torch.Tensor, h_num: int, w_num: int, h_real, w_real) -> torch.Tensor:
    """
    :param x: (batch,channel,h,w)
    :param h_num:
    :param w_num:
    :param h_real:
    :param w_real:
    :return:
    """
    assert (len(patch.shape) == 4 and patch.shape[0] == 1)
    #assert patch.requires_grad == True
    patch = patch.repeat(1, 1, h_num, w_num)
    patch = patch[:, :, :h_real, :w_real]
    return patch

def random_offset_h(image: torch.Tensor, scale_range=0.1):
    assert (len(image.shape) == 4 and image.shape[0] == 1)
    #assert image.requires_grad == True
    scale = random.random()
    hoffset=int(image.shape[2]*scale_range*scale)+1
    new_image=torch.concat([image[:, :, hoffset:, :], image[:, :, :hoffset, :]], dim=2)
    assert new_image.shape==image.shape
    return new_image

def random_offset_w(image: torch.Tensor, scale_range=0.1):
    assert (len(image.shape) == 4 and image.shape[0] == 1)
    #assert image.requires_grad == True
    scale = random.random()
    woffset=int(image.shape[3]*scale_range*scale)+1
    new_image = torch.concat([image[:, :, :, woffset:], image[:, :, :, :woffset]], dim=3)
    assert new_image.shape==image.shape
    return new_image


def random_jpeg(image:torch.Tensor):
    assert (len(image.shape) == 4 and image.shape[0] == 1)
    #assert image.requires_grad == True
    qs=[75,80,85,90]
    q=random.choice(qs)
    h,w=image.shape[2:]
    h_resize=(h//112)*112
    w_resize = (w // 112) * 112
    image = transforms.Resize([h_resize, w_resize])(image)
    jpeg = DiffJPEG(h_resize, w_resize, differentiable=True, quality=q).cuda()
    image=jpeg(image)
    return image

def normlize_MeanVariance(image:torch.Tensor,device):
    assert (len(image.shape) == 4 and image.shape[0] == 1)
    mean = torch.Tensor([[[[0.485]],[[0.456]],[[ 0.406]]]])
    mean=mean.to(device)
    variance = torch.Tensor([[[[0.229]], [[0.224]], [[0.225]]]])
    variance=variance.to(device)
    image=(image-mean)/variance
    return image

def resize_aspect_ratio(image:torch.Tensor,device,square_size,mag_ratio=1.5):
    assert (len(image.shape) == 4 and image.shape[0] == 1)
    h,w=image.shape[2:]
    target_size = mag_ratio * max(h, w)
    if target_size>square_size:
        target_size=square_size
    ratio=target_size/max(h,w)
    target_h,target_w=int(h*ratio),int(w*ratio)
    image=transforms.Resize([target_h,target_w])(image)

    # make canvas and paste image
    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)
    resized = torch.zeros([1,3,target_h32, target_w32]).to(device)
    resized[:,:,0:target_h,0:target_w]=image

    #target_h, target_w = target_h32, target_w32
    #size_heatmap = (int(target_w / 2), int(target_h / 2))
    #return size_heatmap
    return resized,ratio




def test_random_jpeg():
    from AllConfig.GConfig import test_img_path
    from Tools.Imagebasetool import img_read,img_grad_show,img_tensortocv2,img_show3
    img=img_read(test_img_path)
    img=img.cuda()
    img.requires_grad=True
    img=random_jpeg(img)
    imcv=img_tensortocv2(img)
    img_show3(imcv)
    img_grad_show(img)


def test_repeat_4D():
    x = torch.Tensor([[[[0.4942, 0.1321],
                          [0.3797, 0.3320]]]])
    x.requires_grad = True
    img_h, img_w = 5, 5
    h_num = int(img_h / x.shape[2]) + 1
    w_num = int(img_w / x.shape[3]) + 1
    y = repeat_4D(x, h_num, w_num, img_h, img_w)

    referance=torch.Tensor([[[[0.4942, 0.1321,0.4942, 0.1321,0.4942],
                              [0.3797, 0.3320,0.3797, 0.3320,0.3797],
                              [0.4942, 0.1321,0.4942, 0.1321,0.4942],
                              [0.3797, 0.3320,0.3797, 0.3320,0.3797],
                              [0.4942, 0.1321,0.4942, 0.1321,0.4942],]]])
    assert (y==referance).all()
    img_grad_show(y)

def test_random_resize():
    from Tools.Imagebasetool import img_grad_show
    img = torch.randn(1, 3, 120, 100)
    img.requires_grad = True
    img = random_image_resize(img, low=0.1, high=3)
    img_grad_show(img)

def test_random_offset_h():
    from Tools.Imagebasetool import img_grad_show
    img=torch.Tensor([[[[0.4942, 0.1321],
          [0.3797, 0.3320]]]])
    img.requires_grad=True
    img,hoffset=random_offset_h(img,scale_range=1)
    #print(img)
    assert (img==torch.tensor([[[[0.3797, 0.3320],
                                [0.4942, 0.1321],]]])).all()
    img_grad_show(img)

def test_random_offset_w():
    from Tools.Imagebasetool import img_grad_show
    img = torch.Tensor([[[[0.4942, 0.1321],
                          [0.3797, 0.3320]]]])
    img.requires_grad=True
    img,woffset=random_offset_w(img,scale_range=0.1)
    assert (img==torch.tensor([[[[0.1321, 0.4942],
                                [0.3320, 0.3797],]]])).all()
    img_grad_show(img)


def test_resize_aspect_ratio():
    from Tools.Imagebasetool import img_grad_show
    img=torch.randn(1,3,5,5)
    img.requires_grad=True
    img=img.cuda()
    resize_image,ratio=resize_aspect_ratio(image=img,device=torch.device('cuda:0'),
                                           square_size=10,mag_ratio=1.5)
    print(resize_image.shape)
    print(resize_image[0,0,:10,0])
    img_grad_show(resize_image)
