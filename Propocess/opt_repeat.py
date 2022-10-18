import torch

def repeat_4D(patch:torch.Tensor,h_num:int,w_num:int,h_real,w_real)->torch.Tensor:
    """
    :param x: (batch,channel,h,w)
    :param h_num:
    :param w_num:
    :param h_real:
    :param w_real:
    :return:
    """
    assert patch.requires_grad==True
    patch=patch.repeat(1,1,h_num,w_num)
    patch=patch[:,:,:h_real,:w_real]
    return patch


if __name__=='__main__':
    from Tools.Imagebasetool import *
    x=torch.randn(1,3,2,2)
    x.requires_grad=True
    img_h,img_w=5,7
    h_num=int(img_h/x.shape[2])
    w_num = int(img_w / x.shape[3])
    y=repeat_4D(x,h_num,w_num,img_h,img_w)
    img_grad_show(y)

