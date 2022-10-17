# Pytorch
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
from torch.autograd import Variable
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from torchvision import models
import math
# Local
import cv2
import sys
sys.path.append(r'/home/luning/PycharmProjects/mac_code/luning_experimental_code/DiffJPEG/modules')
import Propocess.DIFFJPEG.modules.modules_compression as compression
import Propocess.DIFFJPEG.modules.modules_decompression as decompression
from Propocess.DIFFJPEG.utils import diff_round, quality_to_factor
import matplotlib.pyplot as plt
import numpy as np
img_transform = transforms.Compose([
    transforms.ToTensor()
])
def show_images_diffrence(original_img, after_jpeg_img):
    plt.title('Original_img')
    plt.imshow(original_img)
    plt.axis('off')
    plt.show()
    plt.title('JPEG_img')
    plt.imshow(after_jpeg_img)
    plt.axis('off')
    plt.show()
class NIMA(nn.Module):
    """Neural IMage Assessment model by Google"""
    def __init__(self, base_model, num_classes=10):
        super(NIMA, self).__init__()
        self.features = base_model.features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.75),
            nn.Linear(in_features=25088, out_features=num_classes),
            nn.Softmax())

    def forward(self, x):
        out = self.features(x)
        out = out.reshape(out.size(0), -1)
        out = self.classifier(out)
        return out
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = models.vgg16(pretrained=True)
attack_model = NIMA(base_model).cuda()
attack_model.load_state_dict(
    torch.load('/home/luning/PycharmProjects/mac_code/luning_experimental_code/NIMA/epoch-82.pth'))
class DiffJPEG(nn.Module):
    def __init__(self, height=224, width=224, differentiable=True, quality=90):
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
        self.compress = compression.compress_jpeg(rounding=rounding, factor=factor)
        self.decompress = decompression.decompress_jpeg(height, width, rounding=rounding, factor=factor)

    def forward(self, x):
        y, cb, cr = self.compress(x)
        recovered = self.decompress(y, cb, cr)
        return recovered


def psnr(target, ref):
    # target:目标图像  ref:参考图像  scale:尺寸大小
    # print('targe.shape = ', target.shape)
    # print('ref.shape = ', ref.shape)
    diff = ref - target
    diff = diff.flatten('C')
    mse = math.sqrt(np.mean(diff ** 2.))
    return 20 * math.log10(1.0 / mse)
diff_img = DiffJPEG()
if __name__ == '__main__':
    image_path = '/home/luning/PycharmProjects/mac_code/luning_experimental_code/experiment_picture/STN/nima_kodim24.png'
    score = 0.0
    img = Image.open(image_path).convert('RGB')
    img = img_transform(img)
    img = img.unsqueeze(dim=0)
    # 压缩 1*3*224*224
    after_jpeg_img = diff_img(img)
    after_jpeg_img = Variable(after_jpeg_img.to(device))
    out = attack_model(after_jpeg_img)
    out = out.view(10, 1)
    for j, e in enumerate(out, 1):
        score += j * e
    print('score = ', score)
    # 展示经过JPEG压缩后的图片
    img = img.squeeze(dim=0)
    img = img.numpy()
    img = img.transpose(1, 2, 0)
    after_jpeg_img = after_jpeg_img.squeeze(dim=0)
    after_jpeg_img = after_jpeg_img.cpu().numpy()
    after_jpeg_img = after_jpeg_img.transpose(1, 2, 0)
    show_images_diffrence(img, after_jpeg_img)
    psnr_score = psnr(img, after_jpeg_img)
    print('psnr = ', psnr_score)




    # weight = 224
    # height = 224
    # img = cv2.imread(image_path)
    # img_orig = img
    # img = cv2.resize(img, (weight, height))
    # img_orig = cv2.resize(img_orig, (weight, height))
    # img = img.copy().astype(np.float32) #将像素值转换成float类型
    # img /= 255.0
    # img = img.transpose(2, 0, 1) #交换img的序列
    # img = np.expand_dims(img, axis=0)
    # img = torch.from_numpy(img)
    # print('img.shape = ', img.shape)
    # #压缩 1*3*480*640
    # diff_img = DiffJPEG()
    # after_jpeg_img = diff_img(img)
    # # #显示图像
    # print('after_jpeg_img.shape = ', after_jpeg_img.shape)
    # after_jpeg_img = after_jpeg_img.data.cpu().numpy()[0]
    # after_jpeg_img *= 255
    # after_jpeg_img = np.clip(after_jpeg_img, 0, 255).astype(np.uint8)#只保留0-255之间的值
    # after_jpeg_img = after_jpeg_img.transpose(1, 2, 0) #交换img的序列
    # #对比展现原始图片和经过JPEG压缩后的图片
    # def show_images_diffrence(original_img, adversarial_img):
    #     plt.title('Original_img')
    #     plt.imshow(img_orig)
    #     plt.axis('off')
    #     plt.show()
    #     plt.title('JPEG_img')
    #     plt.imshow(after_jpeg_img)
    #     plt.axis('off')
    #     plt.show()
    # show_images_diffrence(img_orig, after_jpeg_img)
