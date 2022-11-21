import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import tqdm
from model_DBnet.pred_single import *
from model_CRAFT.pred_single import *
from Tools.Imagebasetool import img_read, img_extract_background, img_tensortocv2
from Tools.ImageProcess import *
import random
from ATK.ImageAugum import *


class RepeatAdvPatch_Attack():
    def __init__(self,
                 train_path, test_path, savedir,
                 eps=100 / 255, alpha=1 / 255, decay=1.0,
                 epoches=101, batch_size=8,
                 adv_patch_size=(1, 3, 100, 100),
                 is_test=True, ):
        self.DBmodel = load_DBmodel(GConfig.DB_device)
        self.CRAFTmodel=load_CRAFTmodel(device=GConfig.CRAFT_device)
        # hyper-parameters
        self.eps = eps
        self.alpha = alpha
        self.decay = decay

        # train settings
        self.epoches = epoches
        self.batch_size = batch_size
        self.loss = nn.MSELoss()

        # path process
        self.savedir = savedir
        if os.path.exists(self.savedir) == False:
            os.makedirs(self.savedir)
        self.train_dataset = [os.path.join(train_path, name) for name in os.listdir(train_path)]
        self.test_dataset = [os.path.join(test_path, name) for name in os.listdir(test_path)]

        if is_test:
            self.train_dataset = self.train_dataset[:batch_size * 2]
        self.train_images = [img_read(path) for path in self.train_dataset]  # [(1,3,h,w),...]
        self.test_images = [img_read(path) for path in self.test_dataset]

        # initiation
        self.adv_patch = torch.zeros(list(adv_patch_size))

    def get_image_backgroud_mask(self, image_list):
        mask_list = []
        for item in image_list:
            mask_list.append(img_extract_background(item))
        return mask_list
    def list_to_cuda(self, data):
        data_cuda = []
        for item in data:
            data_cuda.append(item.cuda())
        return data_cuda

    def get_merge_image(self, patch: torch.Tensor, mask_list: list,
                        image_list: list, hw_list: list):
        assert patch.requires_grad == True
        patch_h, patch_w = patch.shape[2:]
        adv_image_list = []
        for mask, image, [h, w] in zip(mask_list, image_list, hw_list):
            repeat_patch = repeat_4D(patch=patch, h_num=int(h / patch_h) + 1, w_num=int(w / patch_w) + 1,
                                     h_real=h, w_real=w)
            adv_image_list.append(image + repeat_patch * mask.cuda())
        return adv_image_list

    def get_DB_results(self, aug_images):
        db_results = []
        for img in aug_images:
            preds = self.DBmodel(img)[0]
            prob_map = preds[0]
            db_results.append(prob_map)
        return db_results
    def get_DB_single_result(self,aug_image):
        preds = self.DBmodel(aug_image)[0]
        prob_map = preds[0]
        return prob_map
    def get_DB_mean_loss(self, resultes):
        cost = torch.Tensor([0])
        for res in resultes:
            target_prob_map = torch.zeros_like(res)
            cost += -self.loss(res, target_prob_map.cuda())
        cost = cost / len(resultes)
        return cost
    def get_DB_single_loss(self,res):
        target_prob_map = torch.zeros_like(res)
        cost = -self.loss(res, target_prob_map.cuda())
        return cost


    def train(self):
        print("start training-====================")
        for epoch in range(self.epoches):
            print("epoch: ", epoch)
            # 每个epoch都初始化动量
            momentum = 0
            # 每次epoch都打乱样本库
            random.shuffle(self.train_images)  # this epoch
            batchs = int(len(self.train_dataset) / self.batch_size)
            # 初始化扰动
            adv_patch = self.adv_patch.clone().detach().cuda()
            adv_patch.requires_grad = True
            for i in range(batchs):
                # 拿到batchsize数据并存放到cuda
                batchs_images = self.train_images[i * self.batch_size: i + 1 * self.batch_size]
                batchs_images = self.list_to_cuda(batchs_images)
                hw_list = get_image_hw(batchs_images)
                masks_list = self.get_image_backgroud_mask(batchs_images)  # 提取背景
                # 嵌入扰动
                adv_images = self.get_merge_image(adv_patch, mask_list=masks_list,
                                                  image_list=batchs_images, hw_list=hw_list)
                # 数据扩增
                aug_images = get_augm_image(adv_images)

                # 遍历，扩增数据分别分别输入到两个模型中
                log_DB_logits_loss=0
                log_CRAFT_logits_loss = 0
                #当前的batch的grad
                sum_grad=torch.zeros_like(adv_patch)
                for a_image in aug_images:
                    # 计算db_logit损失
                    db_result = self.get_DB_single_result(a_image)
                    db_single_loss = self.get_DB_single_loss(db_result)
                    log_DB_logits_loss+=db_single_loss.clone().detach().cpu().item()
                    grad_db = torch.autograd.grad(db_single_loss, adv_patch,
                                               retain_graph=False, create_graph=False)[0]
                    # 计算craft_logit损失
                    score_text, score_link, target_ratio = get_CRAFT_pred(self.CRAFTmodel, img=aug_images,
                                                                          square_size=1280,
                                                                          device=GConfig.CRAFT_device, is_eval=True)
                    boxes = get_CRAFT_box(score_text, score_link, target_ratio,
                                          text_threshold=0.7, link_threshold=0.4, low_text=0.4)

                    grad_craft = torch.autograd.grad(craft_single_loss, adv_patch,
                                                  retain_graph=False, create_graph=False)[0]
                    sum_grad+=grad_db
                    sum_grad += grad_craft

                print("batch:{}, db_mean_loss:{}".format(epoch, db_mean_loss))

                # 计算梯度 更新 动量
                grad = torch.autograd.grad(db_mean_loss, adv_patch,
                                           retain_graph=False, create_graph=False)[0]
                grad = grad / torch.mean(torch.abs(grad), dim=(1), keepdim=True)  # 有待考证
                grad = grad + momentum * self.decay
                momentum = grad

                # 更新adv_patch
                temp_patch = adv_patch.clone().detach().cpu() + self.alpha * grad.sign().cpu()
                temp_patch = torch.clamp(temp_patch, min=-self.eps, max=0)
                adv_patch = temp_patch
            print("batch:{}, db_mean_loss:{}".format(epoch, db_mean_loss))
            # epoch结束 更新self.adv_patch
            self.adv_patch = adv_patch
            # 保存epoch结果
            if epoch != 0 and epoch % 20 == 0:
                self.evauate_db_test_path(epoch)
                temp_save_path = os.path.join(self.savedir, "advpatch")
                if os.path.exists(temp_save_path):
                    os.makedirs(temp_save_path)
                self.save_adv_patch_img(self.adv_patch, os.path.join(temp_save_path, "advpatch_{}.jpg".format(epoch)))

    def evaulae_db_draw(self, adv_images, path, epoch):
        adv_images_cuda = self.list_to_cuda(adv_images)
        db_results = []
        for img in adv_images_cuda:
            preds = self.DBmodel(img)[0]
            db_results.append(preds)
        hw_lists = get_image_hw(adv_images)
        i = 0
        save_name = "DB_{}_{}.jpg"
        for adv_image, pred, [h, w] in zip(adv_images, db_results, hw_lists):
            _, boxes = get_DB_dilateds_boxes(pred, h, w)
            adv_image_cv2 = img_tensortocv2(adv_image)
            DB_draw_box(adv_image_cv2, boxes=boxes, save_path=os.path.join(path, save_name.format(epoch, i)))
            i += 1
        return db_results

    def save_adv_patch_img(self, img_tensor, path):
        img_cv = img_tensortocv2(img_tensor)
        cv2.imwrite(path, img_cv)

    def evauate_db_test_path(self, epoch):
        save_dir = os.path.join(self.savedir, "eval", "orgin")
        save_resize_dir = os.path.join(self.savedir, "eval", "resize")
        save_jpeg_dir = os.path.join(self.savedir, "eval", "jepg")
        if os.path.exists(save_dir):
            os.makedirs(save_dir)
        if os.path.exists(save_jpeg_dir):
            os.makedirs(save_jpeg_dir)
        if os.path.exists(save_resize_dir):
            os.makedirs(save_resize_dir)
        hw_s = get_image_hw(self.test_images)
        mask_s = self.get_image_backgroud_mask(self.test_images)
        adv_images = self.get_merge_image(self.adv_patch.clone().detach().cpu(),
                                          mask_s, self.test_images, hw_s)
        self.evaulae_db_draw(adv_images=adv_images, path=save_dir, epoch=epoch)
        resize_adv_images = get_random_resize_image(adv_image_lists=adv_images, low=0.4, high=3)
        self.evaulae_db_draw(adv_images=resize_adv_images, path=save_resize_dir, epoch=epoch)
        jpeg_adv_image = get_random_jpeg_image(adv_image_lists=adv_images)
        self.evaulae_db_draw(adv_images=jpeg_adv_image, path=save_jpeg_dir, epoch=epoch)


if __name__ == '__main__':
    RAT = RepeatAdvPatch_Attack(train_path=r'F:\OCR-TASK\Wsf\data\train', test_path=r'F:\OCR-TASK\Wsf\data\test',
                                savedir='../result_save/100_100',
                                eps=100 / 255, alpha=1 / 255, decay=0.5,
                                epoches=101, batch_size=8,
                                adv_patch_size=(1, 3, 100, 100),
                                is_test=True)
    RAT.train()
