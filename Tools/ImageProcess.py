import torch
import random
from torchvision import transforms
from Tools.Imagebasetool import img_grad_show


def random_image_resize(image: torch.Tensor, low=0.25, high=3):
    assert (len(image.shape) == 4 and image.shape[0] == 1)
    assert image.requires_grad == True
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
    assert patch.requires_grad == True
    patch = patch.repeat(1, 1, h_num, w_num)
    patch = patch[:, :, :h_real, :w_real]
    return patch

def random_offset_h(image: torch.Tensor, scale_range=0.1):
    assert (len(image.shape) == 4 and image.shape[0] == 1)
    assert image.requires_grad == True
    scale = random.random()
    hoffset=int(image.shape[2]*scale_range*scale)+1
    new_image=torch.concat([image[:, :, hoffset:, :], image[:, :, :hoffset, :]], dim=2)
    assert new_image.shape==image.shape
    return new_image, hoffset

def random_offset_w(image: torch.Tensor, scale_range=0.1):
    assert (len(image.shape) == 4 and image.shape[0] == 1)
    assert image.requires_grad == True
    scale = random.random()
    woffset=int(image.shape[3]*scale_range*scale)+1
    new_image = torch.concat([image[:, :, :, woffset:], image[:, :, :, :woffset]], dim=3)
    assert new_image.shape==image.shape
    return new_image, woffset




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