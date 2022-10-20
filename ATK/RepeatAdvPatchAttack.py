import math
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import tqdm
from model_DBnet.pred_single import *
from Tools.Imagebasetool import img_read
from AllConfig.GConfig import test_img_path
from Tools.ImageProcess import repeat_4D,random_image_resize,random_offset_h,random_offset_w
import random

class RepeatAdvPatch_Attack():
    def __init__(self,
                 train_path, savedir,
                 eps=100/255, alpha=1/255,decay=1.0,
                 epoches=100, batch_size=8,
                 adv_patch_size=(1,3,100,100),
                 is_test=True,):
        self.DBmodel = load_DBmodel()
        # hyper-parameters
        self.eps = eps
        self.alpha = alpha
        self.decay = decay

        #train settings
        self.epoches = epoches
        self.batch_size=batch_size
        self.loss = nn.MSELoss()

        # path process
        self.savedir = savedir
        if os.path.exists(self.savedir) == False:
            os.makedirs(self.savedir)
        self.train_dataset=[os.path.join(train_path,name) for name in os.listdir(train_path)]
        if is_test:
            self.train_dataset=self.train_dataset[:batch_size*2]
        self.train_images=[img_read(path) for path in self.train_dataset]#[(1,3,h,w),...]

        #initiation
        self.adv_patch=torch.zeros(list(adv_patch_size))

    def get_image_hw(self,image_list):
        hw_list=[]
        for item in image_list:
            hw_list.append(item.shape[2:])
        return hw_list

    def get_image_backgroud_mask(self,image_list):
        mask_list=[]
        for item in image_list:
            mask_list.append(img_extract_background(item))
        return mask_list

    def list_to_cuda(self,data):
        data_cuda=[]
        for item in data:
            data_cuda.append(item.cuda())
        return data_cuda

    def get_merge_image(self,patch:torch.Tensor,mask_list:list,image_list:list,hw_list:list):
        assert patch.requires_grad==True
        patch_h,patch_w=patch.shape[2:]
        adv_image_list=[]
        for mask,image,[h,w] in zip(mask_list,image_list,hw_list):
            repeat_patch=repeat_4D(patch=patch,h_num=int(h/patch_h)+1,w_num=int(w/patch_w)+1,
                                   h_real=h,w_real=w)
            adv_image_list.append(image+repeat_patch*mask.cuda())
        return adv_image_list

    def get_augm_image(self,adv_images):
        pass

    def get_DB_results(self,aug_images):

    def attack_(self):
        for epoch in range(self.epoches):
            # 每次epoch都打乱样本库
            epoch_images=random.shuffle(self.train_images)# this epoch
            batchs= int(len(self.train_dataset)/self.batch_size)

            #初始化扰动
            adv_patch=self.adv_patch.clone().detach().cuda()
            adv_patch.requires_grad=True
            for i in range(batchs):
                #拿到batchsize数据并存放到cuda
                batchs_images=epoch_images[i*self.batch_size:i+1*self.batch_size]
                batchs_images=self.list_to_cuda(batchs_images)
                hw_list = self.get_image_hw(batchs_images)
                masks_list = self.get_image_backgroud_mask(batchs_images)#提取背景
                #嵌入扰动
                adv_images = self.get_merge_image(adv_patch,mask_list=masks_list,
                                                  image_list=batchs_images,hw_list=hw_list)
                #数据扩增
                aug_images = []








    def attack_single(self, step=100, patch_size=(1,3,60,60), decay=1,alpha=1/255,eps=50/255):
        #加载原图
        img=img_read(test_img_path)
        img = torch.unsqueeze(img, dim=0)
        h,w=img.shape[2:]
        #提取背景mask
        background_mask=img_extract_background(img)
        cv2.imwrite("../result_save/mask.jpg",img_tensortocv2(torch.cat([background_mask,background_mask,background_mask],dim=1)))
        #初始化patch
        adv_patch=torch.zeros(list(patch_size))
        #初始化动量
        momentum=torch.zeros_like(adv_patch).detach().cuda()

        #初始化损失
        loss = nn.MSELoss()


        for i in range(step):
            #克隆中间变量   生成advimg
            imc = img.clone().detach().cuda()
            patchc=adv_patch.clone().detach().cuda()
            patchc.requires_grad=True#可导
            patch_repeat=repeat_4D(patchc,h_num=int(h/patch_size[2])+1,
                                   w_num=int(w/patch_size[3])+1,h_real=h,w_real=w)

            advimg=imc+patch_repeat*background_mask.cuda()
            #非常重要的操作***********
            advimg=torch.clamp(advimg,min=0,max=1)

            #输入模型计算结果
            preds=self.DBmodel(advimg)[0]
            prob_map=preds[0]

            #初始化target_label
            target_prob_map=torch.zeros_like(prob_map)
            target_prob_map=target_prob_map.cuda()
            cost = -loss(prob_map, target_prob_map)
            print("step:{}, cost：{}".format(i,cost))

            #计算梯度 更新 动量
            grad = torch.autograd.grad(cost, patchc,
                                       retain_graph=False, create_graph=False)[0]
            grad = grad / torch.mean(torch.abs(grad), dim=(1), keepdim=True)
            grad = grad + momentum * decay
            momentum = grad

            #更新中间扰动temp_patch 进行裁剪等操作
            temp_patch = patchc.clone().detach().cpu() + alpha * grad.sign().cpu()
            #并不对图像进行最大最小值的剪裁 仅仅对扰动大小进行修建
            #temp_advimg = torch.clamp(repeat_4D(temp_patch)+img,min=0,max=1)
            temp_patch = torch.clamp(temp_patch,min=-eps,max=0)

            #更新外部扰动
            adv_patch=temp_patch

            #保存中间结果
            if i!=0 and i%50==0:
                temp_adv_patch=adv_patch.detach().clone().cpu()
                #计算扰动图像并修剪  保存
                temp_adv_img=repeat_4D(temp_adv_patch, h_num=int(h / patch_size[2]) + 1,
                          w_num=int(w / patch_size[3]) + 1, h_real=h, w_real=w)*background_mask + img
                #保存adv patch 但是记得要用负数保存
                adv_patch_cv2 = img_tensortocv2(temp_adv_patch+1)
                cv2.imwrite("../result_save/adv_patch_{}.jpg".format(i), adv_patch_cv2)

                adv_image_cv2=img_tensortocv2(temp_adv_img)
                cv2.imwrite("../result_save/adv_img_{}.jpg".format(i), adv_image_cv2)

                #绘制框
                dilates, boxes = get_DB_dilateds_boxes(preds, h, w, min_area=100)
                DB_draw_dilated(adv_image_cv2, dilateds=dilates, save_path=r"..\result_save\test_save\dilated_{}.jpg".format(i))
                DB_draw_box(adv_image_cv2, boxes=boxes, save_path=r"..\result_save\test_save\boxes_{}.jpg".format(i))

    def attack_multi(self):
        pass

if __name__ == '__main__':
    RAT=Repeat_Attack(train_set=None,savedir=None)
    RAT.attack_single(step=101,patch_size=(1,3,100,100),decay=1,alpha=1/255,eps=200/255)