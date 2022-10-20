import cv2
from torchvision import transforms

import torch
from AllConfig.GConfig import test_img_path

import matplotlib.pyplot as plt


def img_show3(img):
    plt.matshow(img)
    plt.show()


def img_show1(img):
    plt.matshow(img, cmap=plt.cm.gray)
    plt.show()


def img_read(image_path) -> torch.Tensor:
    transform = transforms.ToTensor()
    im = cv2.imread(image_path, 1)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # img_show3(im)
    img = transform(im)
    img = img.unsqueeze_(0)
    return img


def img_tensortocv2(img_tensor: torch.Tensor):
    assert (len(img_tensor.shape) == 4 and img_tensor.shape[0] == 1)
    img_tensor = img_tensor.detach().cpu()
    img_tensor = img_tensor.squeeze()
    img_tensor = img_tensor.mul_(255).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    img_cv = cv2.cvtColor(img_tensor, cv2.COLOR_RGB2BGR)
    return img_cv


def img_grad_show(img: torch.Tensor) -> None:
    assert img.requires_grad == True
    ygf = img.grad_fn
    print('')
    print('***********cyclic print grads**************')
    while ygf != ():
        print(ygf)
        try:
            ygf = ygf.next_functions[0][0]
        except:
            break


def img_extract_background(img_tensor: torch.Tensor):
    assert (len(img_tensor.shape) == 4 and img_tensor.shape[0] == 1)
    img_sum = torch.sum(img_tensor, dim=1)
    mask = (img_sum == 3)
    mask = mask + 0
    return mask.unsqueeze_(0)


def test_cv_tensor(img_path=test_img_path):
    img_cv2 = cv2.imread(img_path, 1)
    img_tensor = img_read(img_path)
    img_tensor2cv = img_tensortocv2(torch.unsqueeze(img_tensor, dim=0))
    assert (img_tensor2cv == img_cv2).all()
    print('')


if __name__ == '__main__':
    # img_read(test_img_path)
    test_cv_tensor()
