import sys
sys.path.append("..")
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import tqdm
from model_CRAFT.pred_single import *
from model_DBnet.pred_single import *
from Tools.Imagebasetool import img_read, img_extract_background, img_tensortocv2
from Tools.ImageProcess import *
import random
from ATK.ImageAugum import *
from Mylog.logfun import logger_config
import datetime

class RepeatAdvPatch_Attack():
    def __init__(self,
                 train_path, test_path, savedir,
                 eps=100 / 255, alpha=1 / 255, decay=1.0,
                 epoches=101, batch_size=8,
                 adv_patch_size=(1, 3, 100, 100),
                 is_test=True, ):
        #DB,CRAFT
        self.weights = [1.0]
        self.DBmodel = load_DBmodel(GConfig.DB_device)
        self.CRAFTmodel = load_CRAFTmodel(device=GConfig.CRAFT_device)
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
        self.start_epoch=0
        # recover adv patch
        recover_adv_path,epoch=self.recover_adv_patch()
        if recover_adv_path!=None:
            self.adv_patch=recover_adv_path
            self.start_epoch=epoch



        # transform列表
        self.functions_name=[None,'get_random_resize_image_single',
                             'get_random_resize_image_single',
                             'get_random_noised_image',
                             'get_random_noised_image',
                            'get_random_jpeg_image_single',
                            'get_random_offset_h_single','get_random_offset_w_single',]
        self.logger=logger_config()

    def recover_adv_patch(self):
        temp_save_path = os.path.join(self.savedir, "advpatch")
        if os.path.exists(temp_save_path):
            files=os.listdir(temp_save_path)
            if len(files)==0:
                return None,None
            files=sorted(files,key=lambda x:int(x.split('.')[0].split("_")[-1]))
            epoch=int(files[-1].split('.')[0].split("_")[-1])
            keyfile=os.path.join(temp_save_path,files[-1])
            return img_read(keyfile),epoch
        return None,None

    def get_image_backgroud_mask(self, image_list):
        mask_list = []
        for item in image_list:
            mask_list.append(img_extract_background(item))
        return mask_list

    def list_to_cuda(self, data,device):
        data_cuda = []
        for item in data:
            item=item.to(device)
            data_cuda.append(item)
        return data_cuda

    def get_merge_image_list(self, patch: torch.Tensor, mask_list: list,
                        image_list: list, hw_list: list):
        patch_h, patch_w = patch.shape[2:]
        adv_image_list = []
        for mask, image, [h, w] in zip(mask_list, image_list, hw_list):
            repeat_patch = repeat_4D(patch=patch, h_num=int(h / patch_h) + 1, w_num=int(w / patch_w) + 1,
                                     h_real=h, w_real=w)
            adv_image_list.append(image + repeat_patch)
        return adv_image_list

    def get_merge_image(self, patch: torch.Tensor, mask,image,hw,device):
        assert patch.requires_grad == True
        patch_h, patch_w = patch.shape[2:]
        [h,w]=hw
        repeat_patch = repeat_4D(patch=patch, h_num=int(h / patch_h) + 1, w_num=int(w / patch_w) + 1,
                                     h_real=h, w_real=w)
        mask=mask.to(device)
        return image + repeat_patch * mask


    def get_DB_results(self, aug_images):
        db_results = []
        for img in aug_images:
            preds = self.DBmodel(img)[0]
            prob_map = preds[0]
            db_results.append(prob_map)
        return db_results

    def get_DB_single_result(self, aug_image):
        preds = self.DBmodel(aug_image)[0]
        prob_map = preds[0]
        return prob_map

    def get_DB_mean_loss(self, resultes):
        cost = torch.Tensor([0])
        for res in resultes:
            target_prob_map = torch.zeros_like(res)
            target_prob_map.to(GConfig.DB_device)
            cost += -self.loss(res, target_prob_map)
        cost = cost / len(resultes)
        return cost

    def get_DB_single_loss(self, res, device):
        target_prob_map = torch.zeros_like(res)
        target_prob_map = target_prob_map.to(device)
        cost = -self.loss(res, target_prob_map)
        return cost

    def get_CRAFT_single_loss(self, score_text, score_link, device):
        target_link = torch.ones_like(score_link)*-0.1
        target_text = torch.ones_like(score_text)*-0.1
        target_text, target_link = target_text.to(device), target_link.to(device)
        cost = -0.5 * self.loss(score_link, target_link) - 0.5 * self.loss(score_text, target_text)
        return cost

    def get_DB_grad(self,adv_image,adv_patch):
        db_result = self.get_DB_single_result(adv_image)
        db_single_loss = self.get_DB_single_loss(db_result, device=GConfig.DB_device)
        grad_db = torch.autograd.grad(db_single_loss, adv_patch,
                                      retain_graph=False, create_graph=False)[0]
        return db_single_loss.detach().cpu().item(),grad_db.detach().cpu()

    def get_CRAFT_grad(self,adv_image,adv_patch):
        score_text, score_link, target_ratio = get_CRAFT_pred(self.CRAFTmodel, img=adv_image,
                                                              square_size=1280,
                                                              device=GConfig.CRAFT_device, is_eval=False)
        craft_single_loss = self.get_CRAFT_single_loss(score_text=score_text, score_link=score_link,
                                                       device=GConfig.CRAFT_device)
        grad_craft = torch.autograd.grad(craft_single_loss, adv_patch,
                                         retain_graph=False, create_graph=False)[0]
        return craft_single_loss.detach().cpu().item(),grad_craft.detach().cpu()
    #快捷初始化
    def inner_init_adv_patch_image(self,mask,image,hw, device):
        adv_patch = self.adv_patch.clone().detach()
        adv_patch=adv_patch.to(device)
        adv_patch.requires_grad = True
        image=image.to(device)
        adv_image = self.get_merge_image(adv_patch, mask=mask,
                                         image=image, hw=hw,device=device)
        return adv_patch,adv_image

    #循环遍历,提供batch_images,等，返回grad_sum,和loss
    def inner_batch(self,mask,image,hw):
        sum_grad = torch.zeros_like(self.adv_patch)
        fun_DB_loss=0
        fun_CRAFT_loss = 0
        for item in self.functions_name:
            if item==None:
                adv_patch, adv_image = self.inner_init_adv_patch_image(mask, image, hw,device=GConfig.DB_device)
                temp_db_loss, temp_db_grad = self.get_DB_grad(adv_image, adv_patch)  # DB
                fun_DB_loss+=temp_db_loss
                sum_grad+=temp_db_grad
                del (adv_patch, adv_image,temp_db_grad,temp_db_loss)
                torch.cuda.empty_cache()
                adv_patch, adv_image = self.inner_init_adv_patch_image(mask, image, hw,device=GConfig.CRAFT_device)
                temp_craft_loss, temp_craft_grad = self.get_CRAFT_grad(adv_image, adv_patch)  # CRAFT
                fun_CRAFT_loss+=temp_craft_loss
                sum_grad += temp_craft_grad
                del(adv_patch, adv_image,temp_craft_grad,temp_craft_loss)
                torch.cuda.empty_cache()
            else:
                now_fun=eval(item)

                adv_patch, adv_image = self.inner_init_adv_patch_image(mask, image, hw,device=GConfig.DB_device)
                if item=='get_random_jpeg_image_single':
                    adv_image = now_fun(adv_image,GConfig.DB_device)
                else:
                    adv_image = now_fun(adv_image)
                temp_db_loss, temp_db_grad = self.get_DB_grad(adv_image, adv_patch)  # DB
                fun_DB_loss += temp_db_loss
                sum_grad += temp_db_grad
                del (adv_patch, adv_image,temp_db_grad, temp_db_loss)
                torch.cuda.empty_cache()

                adv_patch, adv_image = self.inner_init_adv_patch_image(mask, image, hw,device=GConfig.CRAFT_device)
                if item == 'get_random_jpeg_image_single':
                    adv_image = now_fun(adv_image, GConfig.CRAFT_device)
                else:
                    adv_image = now_fun(adv_image)
                temp_craft_loss, temp_craft_grad = self.get_CRAFT_grad(adv_image, adv_patch)  # CRAFT
                sum_grad += temp_craft_grad
                fun_CRAFT_loss += temp_craft_loss
                del (adv_patch, adv_image,temp_craft_grad, temp_craft_loss)
                torch.cuda.empty_cache()

        return fun_DB_loss,fun_CRAFT_loss,sum_grad
    def train(self):
        print("start training-====================")
        for epoch in range(self.start_epoch+1,self.epoches):
            print("epoch: ", epoch)
            # 每个epoch都初始化动量
            momentum = 0
            # 每次epoch都打乱样本库
            random.shuffle(self.train_images)  # this epoch

            batchs = int(len(self.train_dataset) / self.batch_size)


            log_epoch_DB_loss=0
            log_epoch_CRAFT_loss = 0
            #这里遍历一个epoch内的batchs
            ti=0
            for i in range(batchs):
                # 拿到batchsize数据并存放到cuda
                batchs_images = self.train_images[i * self.batch_size: (i + 1) * self.batch_size]
                hw_list = get_image_hw(batchs_images)
                masks_list = self.get_image_backgroud_mask(batchs_images)  # 提取背景

                #这边遍历单个batch
                sum_grad=torch.zeros_like(self.adv_patch)
                log_batch_DB_loss=0
                log_batch_CRAFT_loss = 0
                for b_index,b_img in enumerate(batchs_images):
                    #print(self.train_dataset[ti])
                    ti+=1
                    fun_DB_loss,fun_CRAFT_loss,temp_sum_grad=self.inner_batch(masks_list[b_index],batchs_images[b_index],
                                                                        hw=hw_list[b_index])
                    log_batch_CRAFT_loss+=fun_DB_loss/len(self.functions_name)
                    log_batch_DB_loss+=fun_CRAFT_loss/len(self.functions_name)
                    sum_grad+=temp_sum_grad/len(self.functions_name)
                    del(fun_DB_loss,fun_CRAFT_loss,temp_sum_grad)

                #加到epoch的loss里面
                log_epoch_DB_loss+=log_batch_DB_loss
                log_epoch_CRAFT_loss += log_batch_CRAFT_loss
                # 计算梯度 更新 动量
                grad = sum_grad / torch.mean(torch.abs(sum_grad), dim=(1), keepdim=True)  # 有待考证
                grad = grad + momentum * self.decay
                momentum = grad

                # 单个batch结束 更新adv_patch
                temp_patch = self.adv_patch.clone().detach().cpu() + self.alpha * grad.sign()
                temp_patch = torch.clamp(temp_patch, min=-self.eps, max=0)
                self.adv_patch = temp_patch
                e="batch_loss==db_loss:{},craft_loss:{}===".format(log_batch_DB_loss,log_batch_CRAFT_loss)
                self.logger.info(e + "------ " + str(datetime.time))
                del(batchs_images)
                del(hw_list)
                del(masks_list)
            # epoch结束 更新self.adv_patch
            # self.adv_patch = adv_patch
            # 打印epoch结果
            e="epoch:{}, db_loss:{},craft_loss:{}".format(epoch, log_epoch_DB_loss,log_epoch_CRAFT_loss)
            self.logger.info(e + "------ " + str(datetime.time))

            #保存advpatch
            temp_save_path = os.path.join(self.savedir, "advpatch")
            if os.path.exists(temp_save_path) == False:
                os.makedirs(temp_save_path)
            self.save_adv_patch_img(self.adv_patch + 1, os.path.join(temp_save_path, "advpatch_{}.jpg".format(epoch)))

            # 保存epoch结果
            if epoch != 0 and epoch % 5 == 0:
                self.evauate_test_path(epoch)


    def evaluate_db_draw(self, adv_images, path, epoch,device):
        adv_images_cuda = self.list_to_cuda(adv_images,device)
        db_results = []
        for img in adv_images_cuda:
            with torch.no_grad():
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
        return

    def evaluate_craft_draw(self,adv_images,path,epoch,device):
        adv_images_cuda = self.list_to_cuda(adv_images,device)
        craft_text_results=[]
        craft_link_results = []
        craft_ratio_results = []
        for img in adv_images_cuda:
            score_text,score_link,target_ratio=get_CRAFT_pred(self.CRAFTmodel,img=img,
                                                              device=device,is_eval=True)
            craft_text_results.append(score_text)
            craft_link_results.append(score_link)
            craft_ratio_results.append(target_ratio)
        i=0
        save_name = "CRAFT_{}_{}.jpg"
        for adv_img,text,link,ratio in zip(adv_images,craft_text_results,
                                           craft_link_results,craft_ratio_results):
            boxes = get_CRAFT_box(text, link, ratio,
                                  text_threshold=0.7, link_threshold=0.4, low_text=0.4)
            adv_image_cv2 = img_tensortocv2(adv_img)
            CRAFT_draw_box(adv_image_cv2, boxes=boxes, save_path=os.path.join(path,save_name.format(epoch,i)))
            i+=1
        return

    def save_adv_patch_img(self, img_tensor, path):
        img_cv = img_tensortocv2(img_tensor)
        cv2.imwrite(path, img_cv)

    def evauate_test_path(self, epoch):
        #============================DBNET=========================
        save_dir_db = os.path.join(self.savedir, "eval", "db_orgin")
        save_resize_dir = os.path.join(self.savedir, "eval", "db_resize")
        save_jpeg_dir = os.path.join(self.savedir, "eval", "db_jepg")
        save_offset_w_dir = os.path.join(self.savedir, "eval", "db_offset_w")
        save_offset_h_dir = os.path.join(self.savedir, "eval", "db_offset_h")
        if os.path.exists(save_dir_db)==False:
            os.makedirs(save_dir_db)
        if os.path.exists(save_jpeg_dir)==False:
            os.makedirs(save_jpeg_dir)
        if os.path.exists(save_resize_dir)==False:
            os.makedirs(save_resize_dir)
        if os.path.exists(save_offset_w_dir)==False:
            os.makedirs(save_offset_w_dir)
        if os.path.exists(save_offset_h_dir)==False:
            os.makedirs(save_offset_h_dir)
        hw_s = get_image_hw(self.test_images)
        mask_s = self.get_image_backgroud_mask(self.test_images)
        adv_images = self.get_merge_image_list(self.adv_patch.clone().detach().cpu(),
                                          mask_s, self.test_images, hw_s)
        self.evaluate_db_draw(adv_images=adv_images, path=save_dir_db, epoch=epoch,device=GConfig.DB_device)
        resize_adv_images = get_random_resize_image(adv_image_lists=adv_images, low=0.4, high=3)
        self.evaluate_db_draw(adv_images=resize_adv_images, path=save_resize_dir, epoch=epoch,device=GConfig.DB_device)
        jpeg_adv_image = get_random_jpeg_image(adv_image_lists=adv_images,device=GConfig.cpu_device)
        self.evaluate_db_draw(adv_images=jpeg_adv_image, path=save_jpeg_dir, epoch=epoch,device=GConfig.DB_device)
        offset_w_adv_images=get_random_offset_w(adv_image_lists=adv_images,scale_range=0.1)
        offset_h_adv_images = get_random_offset_h(adv_image_lists=adv_images, scale_range=0.1)
        self.evaluate_db_draw(adv_images=offset_w_adv_images,path=save_offset_w_dir,epoch=epoch,device=GConfig.DB_device)
        self.evaluate_db_draw(adv_images=offset_h_adv_images, path=save_offset_h_dir, epoch=epoch,device=GConfig.DB_device)

        # ============================CRAFT=========================
        save_dir_craft = os.path.join(self.savedir, "eval", "craft_orgin")
        save_resize_dir_craft = os.path.join(self.savedir, "eval", "craft_resize")
        save_jpeg_dir_craft = os.path.join(self.savedir, "eval", "craft_jepg")
        save_offset_w_dir_craft = os.path.join(self.savedir, "eval", "craft_offset_w")
        save_offset_h_dir_craft = os.path.join(self.savedir, "eval", "craft_offset_h")
        if os.path.exists(save_dir_craft)==False:
            os.makedirs(save_dir_craft)
        if os.path.exists(save_jpeg_dir_craft)==False:
            os.makedirs(save_jpeg_dir_craft)
        if os.path.exists(save_resize_dir_craft)==False:
            os.makedirs(save_resize_dir_craft)
        if os.path.exists(save_offset_w_dir_craft)==False:
            os.makedirs(save_offset_w_dir_craft)
        if os.path.exists(save_offset_h_dir_craft)==False:
            os.makedirs(save_offset_h_dir_craft)
        self.evaluate_craft_draw(adv_images=adv_images,path=save_dir_craft,epoch=epoch,device=GConfig.CRAFT_device)
        self.evaluate_craft_draw(adv_images=resize_adv_images, path=save_resize_dir_craft, epoch=epoch,device=GConfig.CRAFT_device)
        self.evaluate_craft_draw(adv_images=jpeg_adv_image, path=save_jpeg_dir_craft, epoch=epoch,device=GConfig.CRAFT_device)
        self.evaluate_craft_draw(adv_images=offset_h_adv_images, path=save_offset_h_dir_craft, epoch=epoch,device=GConfig.CRAFT_device)
        self.evaluate_craft_draw(adv_images=offset_w_adv_images, path=save_offset_w_dir_craft, epoch=epoch,device=GConfig.CRAFT_device)
        del(resize_adv_images,jpeg_adv_image,offset_w_adv_images,offset_h_adv_images,adv_images,hw_s,mask_s)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    RAT = RepeatAdvPatch_Attack(train_path=r'/share/home/yandiqun/djc/Wsf/data/train', test_path=r'/share/home/yandiqun/djc/Wsf/data/test',
                                savedir='../result_save/120_120',
                                eps=50/ 255, alpha=1 / 255, decay=0.05,
                                epoches=101, batch_size=20,
                                adv_patch_size=(1, 3, 120, 120),
                                is_test=False)
    RAT.train()

