import torch
import cv2
import numpy as np
from torch.autograd import Variable
from model_textbox.ssd import build_ssd



def load_TBmodel(model_name = r'F:\OCR-TASK\OCR__advprotect\model_textbox\weight\vgg16_reducedfc.pth'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MTWI_CLASSES = ('text',)
    num_classes = len(MTWI_CLASSES) + 1  # +1 background
    net = build_ssd('test', 384, num_classes).to(device)
    net.load_state_dict(torch.load(model_name), False)
    net.eval()
    return net

def base_transform(image, size):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    x = cv2.resize(image, (size, size)).astype(np.float32)
    x = x.astype(np.float32)
    return x


class BaseTransform:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size), boxes, labels

def img_read(img_path,size):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    shape = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(BaseTransform(size)(img)[0]).permute(2, 0, 1)
    img = Variable(img.unsqueeze(0))
    return img, shape

def get_box(detections,shape,thresh = 0.2):
    scale = torch.Tensor(shape[1::-1]).repeat(6)
    boxes = []
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= thresh:
            pt = (detections[0, i, j, 1:].clamp(max=1, min=0) * scale).cpu().numpy()
            coords = np.array([[pt[4], pt[5]], [pt[6], pt[7]], [pt[8], pt[9]], [pt[10], pt[11]]])
            boxes.append(coords)
            j += 1
    return boxes