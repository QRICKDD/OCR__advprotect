import os
import cv2
import torch
import numpy as np
from Tools.TBTools import load_TBmodel,img_read, get_box
from Tools.DBTools import DB_draw_box
import warnings
warnings.filterwarnings("ignore")

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


if __name__ == '__main__':
    from AllConfig.GConfig import test_img_path
    os.environ["CUDA_VISIBLE_DEVICES"] = '1,2'
    thresh = 0.2
    net = load_TBmodel()
    x,shape = img_read(test_img_path,net.size)
    y = net(x.cuda())  # forward pass
    boxes = np.array(get_box(y.data,shape))
    print(boxes)
    img = cv2.imread(test_img_path)
    DB_draw_box(img, boxes=boxes, save_path=r"output_img\boxes.jpg")

