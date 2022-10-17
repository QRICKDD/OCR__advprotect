import torch
import random
from torchvision import transforms
"""
image resize operation
just augmentation
random 
"""
def random_image_resize(image:torch.Tensor,low=0.25,high=3):
    image.requires_grad = True
    scale=random.random()
    shape=image.shape
    h,w=shape[-2],shape[-1]
    h,w=int(h*(scale*(high-low)+low)),int(w*(scale*(high-low)+low))
    image=transforms.Resize([h,w])(image)
    return image


if __name__=='__main__':
    x=torch.randn(1,3,10,10)
    print("=====x--",x.shape,"=====")
    y=random_image_resize(x,low=0.25,high=3)
    print("=====y--", y.shape, "=====")
    ygf=y.grad_fn
    while ygf!=():
        print(ygf)
        try:
            ygf=ygf.next_functions[0][0]
        except:
            break
