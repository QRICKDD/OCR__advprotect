from Tools.Imagebasetool import img_read,img_extract_background,img_tensortocv2
from Tools.ImageProcess import *


def get_image_hw(image_list):
    hw_list = []
    for item in image_list:
        hw_list.append(item.shape[2:])
    return hw_list
def get_random_resize_image(adv_image_lists, low=0.25, high=3.0):
    resize_adv_img_lists = []
    for img in adv_image_lists:
        resize_adv_img_lists.append(random_image_resize(img, low, high))
    for img in adv_image_lists:
        resize_adv_img_lists.append(random_image_resize(img, low, high))
    return resize_adv_img_lists
def get_random_jpeg_image(adv_image_lists):
    jpeg_adv_img_lists = []
    for img in adv_image_lists:
        jpeg_adv_img_lists.append(random_jpeg(img))
    return jpeg_adv_img_lists
def get_random_offset_h(adv_image_lists, scale_range=0.1):
    offset_h_adv_img_lists = []
    for img in adv_image_lists:
        offset_h_adv_img_lists.append(random_offset_h(img, scale_range=scale_range))
    return offset_h_adv_img_lists

def get_random_offset_w(adv_image_lists, scale_range=0.1):
    offset_w_adv_img_lists = []
    for img in adv_image_lists:
        offset_w_adv_img_lists.append(random_offset_w(img, scale_range=scale_range))
    return offset_w_adv_img_lists

def get_augm_image(adv_images):
    resize_images = get_random_resize_image(adv_images)
    jpeg_images = get_random_jpeg_image(adv_images)
    offset_h_images = get_random_offset_h(adv_images)
    offset_w_images = get_random_offset_w(adv_images)
    return adv_images + resize_images + jpeg_images + offset_h_images + offset_w_images
