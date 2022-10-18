import math
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import tqdm
from model_DBnet.pred_single import *
from Tools.Imagebasetool import *
from AllConfig.GConfig import test_img_path
from Propocess.opt_repeat import repeat_4D


class Repeat_Attack():
    def __init__(self, train_set, savedir,
                 eps=0.05, alpha=0.001, steps=500, decay=1.0):
        self.DBmodel = load_DBmodel()
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.decay = decay

        self.savedir = savedir
        if os.path.exists(self.savedir) == False:
            os.makedirs(self.savedir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.soft = torch.nn.Softmax(dim=-1)

    def attack_single(self,step=50,patch_size=(1,3,60,60),decay=1,alpha=1/255,eps=20/255):
        #读取cv2原图
        imcv2=cv2.imread(test_img_path)
        #加载原图
        img=img_read(test_img_path)
        h,w=img.shape[2:]
        img=torch.unsqueeze(img,dim=0)
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
            patch_repeat=repeat_4D(patchc,h_num=int(h/patch_size[2]),
                                   w_num=int(w/patch_size[3]),h_real=h,w_real=w)
            advimg=imc+patch_repeat

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
            temp_patch = patchc.clone().detach().cpu() - alpha * grad.sign().cpu()
            temp_advimg = torch.clamp(temp_patch+img,min=0,max=1)
            temp_patch = torch.clamp(img-temp_advimg,min=-eps,max=eps)

            #更新外部扰动
            adv_patch=temp_patch

            #保存中间结果
            if i!=0 and i%10==0:
                adv_patch_cv2=img_tensortocv2(adv_patch)
                cv2.imwrite("../result_save/adv_patch_{}.jpg".format(i),adv_patch_cv2)
                adv_image_cv2=img_tensortocv2(adv_patch+img)
                cv2.imwrite("../result_save/adv_img_{}.jpg".format(i), adv_image_cv2)

                dilates, boxes = get_DB_dilateds_boxes(preds, h, w, min_area=100)
                DB_draw_dilated(imcv2, dilateds=dilates, save_path=r"..\result_save\test_save\dilated_{}.jpg".format(i))
                DB_draw_box(imcv2, boxes=boxes, save_path=r"..\result_save\test_save\boxes_{}.jpg".format(i))

    # def attack_(self, images, labels):
    #     r"""
    #     Overridden.
    #     """
    #     images = images.clone().detach().to(self.device)
    #     target_labels = labels.clone().detach().to(self.device)
    #     momentum = torch.zeros_like(images).detach().to(self.device)
    #
    #     loss = nn.CrossEntropyLoss()
    #
    #     adv_images = images.clone().detach()
    #
    #     cstep = 0
    #     for _ in range(self.steps):
    #         adv_images.requires_grad = True
    #
    #
    #
    #         output = self.model(adv_images)
    #
    #
    #
    #
    #
    #
    #         cost = -loss(output, target_labels) * self.c_norm
    #         cost = cost - loss(outputs_more, more_labels) * self.c_norm_more
    #         cost = cost - loss(outputs_few, few_labels) * self.c_norm_few
    #
    #
    #         grad = torch.autograd.grad(cost, adv_images,
    #                                    retain_graph=False, create_graph=False)[0]
    #
    #         grad = grad / torch.mean(torch.abs(grad), dim=(1), keepdim=True)
    #         grad = grad + momentum * self.decay
    #         momentum = grad
    #
    #         adv_images = adv_images.detach() + self.alpha * grad.sign()
    #         delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
    #         adv_images = torch.clamp(images + delta, min=-1, max=1).detach()
    #
    #         cstep += 1
    #     return adv_images
    #
    # def attack(self):
    #     allnum = 0
    #     allsuccess = 0
    #     for item in tqdm.tqdm(self.attackfiles):
    #         # 读数据  切片成需要的长度
    #         x, sr = sf.read(os.path.join(self.attackdir, item))
    #         x = x[:self.signlelength]
    #         x = np.expand_dims(torch.Tensor(x).numpy(), 0)
    #
    #         # 判断本来是非被预测错误：输入全部是假样本，预测为真则本来就错误
    #         prey = self.model(torch.Tensor(x).cuda()).detach().cpu().numpy()
    #         # print(prey)
    #         if np.argmax(prey, axis=1).item() != 0:
    #             print("原始分类错误")
    #             continue
    #         allnum += 1
    #
    #         xadv = self.attack_(torch.Tensor(x), torch.Tensor([[0., 1.]]))
    #         # 判断是否攻击成功
    #         prey = self.model(xadv)
    #         temp_norm_res = np.argmax(prey.detach().cpu().numpy(), axis=1).item()
    #         # 判断是否攻击成功2
    #         temp_few_xadv = xadv[:, :self.signlelength_few]
    #         temp_few_prey = self.model_few(temp_few_xadv)
    #         temp_few_res = np.argmax(temp_few_prey.detach().cpu().numpy(), axis=1).item()
    #         # 判断是否成功3
    #         temp_more_xadv = torch.cat((xadv, xadv[:, :self.signlelength_more - self.signlelength]), dim=-1)
    #         temp_more_prey = self.model_more(temp_more_xadv)
    #         temp_more_res = np.argmax(temp_more_prey.detach().cpu().numpy(), axis=1).item()
    #
    #         xadv = xadv.detach().cpu().numpy()
    #         xadv = xadv.squeeze(0)
    #         if temp_norm_res != 1 or temp_few_res != 1 or temp_more_res != 1:  # 如果预测不是真则攻击失败
    #             print("攻击失败")
    #         else:
    #             allsuccess += 1
    #             np.save(os.path.join(self.savedir, flctonpy(item)), xadv)
    #     # 输出攻击成功率
    #     print("攻击成功率:", allsuccess / allnum)