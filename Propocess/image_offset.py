import torch
import random
from torchvision import transforms
"""
image offset operation
just augmentation
random scale from default 1~image_size//4
"""

def image_offset(image:torch.Tensor):
    image.requires_grad = True
    h = image.shape[2]
    h_offset_step = int(h // 40)
    w = image.shape[3]
    w_offset_step = int(w // 40)
    for i in range(1,11):
        #h_offset
        new_image=torch.concat([image[:, :, i*h_offset_step:, :], image[:, :, :i*h_offset_step, :]], dim=2)
        assert new_image.shape==image.shape
        new_image = torch.concat([new_image[:, :, :, i*w_offset_step:], new_image[:, :, :, :i*w_offset_step]], dim=3)
        assert new_image.shape == image.shape
        #保存图片到指定位置
    image=transforms.Resize([h,w])(image)
    return image,h,w


if __name__=='__main__':
    x=torch.randn(1,3,10,10)
    print("=====x--",x.shape,"=====")
    y,h,w=random_image_resize(x,low=0.25,high=3)
    print("=====y--", y.shape, "=====")
    ygf=y.grad_fn
    while ygf!=():
        print(ygf)
        try:
            ygf=ygf.next_functions[0][0]
        except:
            break
