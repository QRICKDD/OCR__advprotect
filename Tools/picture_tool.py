img_test_1 = r"F:\OCR-TASK\Wsf\data\test\002.png"
img_test_3 = r"F:\OCR-TASK\Wsf\data\test\003.png"
img_test_2 = r"F:\OCR-TASK\Wsf\data\test\006.png"
from Tools.Imagebasetool import *
from Tools.ImageProcess import *


def test_fun_xxx():
    patch=torch.load(r"F:\OCR-TASK\OCR__advprotect\result_save\120_120\advtorch\advpatch_15.jpg")
    patch_h,patch_w=120,120
    img=img_read(img_test_1)
    mask=img_extract_background(img)
    img_show1(mask.numpy().squeeze())
    h,w=img.shape[2:]
    repeat_patch = repeat_4D(patch=patch, h_num=int(h / patch_h) + 1, w_num=int(w / patch_w) + 1,
                             h_real=h, w_real=w)
    new_img=img + repeat_patch
    #img_show3(new_img)
    cv_img=img_tensortocv2(new_img)
    cv2.imwrite(r"F:\OCR-TASK\OCR__advprotect\result_save\test_save\show_save.jpg",cv_img)

test_fun_xxx()