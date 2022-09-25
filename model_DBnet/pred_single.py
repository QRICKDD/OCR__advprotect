from model_DBnet.utils.utils import draw_bbox
from model_DBnet.utils.load_model import *
from AllConfig import GConfig
import os


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    DBnet_path=GConfig.DBnet_model_path
    img_path=GConfig.test_img_path

