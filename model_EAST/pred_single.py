import matplotlib.pyplot as plt
from Tools.EASTTools import *


if __name__ == '__main__':
    from Tools.Imagebasetool import *
    from AllConfig.GConfig import test_img_path
    from PIL import Image
    from detect import detect,plot_boxes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #加载模型
    EASTnet=load_EASTmodel()
    #加载图片
    img_path=test_img_path
    img = Image.open(img_path).convert('RGB')
    boxes = detect(img, EASTnet, device)
    plot_img = plot_boxes(img, boxes)
    plt.imshow(plot_img)
    plt.show()


