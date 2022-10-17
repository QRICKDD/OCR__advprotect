from locale import normalize
import os

import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import random
import time
from torchvision.utils import save_image


from Propocess.DIFFJPEG.utils import diff_round, quality_to_factor
import Propocess.DIFFJPEG.compression as compression
import Propocess.DIFFJPEG.decompression as decompression

class DiffJPEG(nn.Module):
    def __init__(self, height, width, differentiable=True, quality=80):
        super(DiffJPEG, self).__init__()
        if differentiable:
            rounding = diff_round
        else:
            rounding = torch.round
        factor = quality_to_factor(quality)
        self.compress = compression.compress_jpeg(rounding=rounding, factor=factor)
        self.decompress = decompression.decompress_jpeg(height, width, rounding=rounding, factor=factor)

    def forward(self, x):
        y, cb, cr = self.compress(x)
        recovered = self.decompress(y, cb, cr)
        return recovered


######## For Model ###############

def load_model(model,params):
    if os.path.exists(params):
        checkpoint = torch.load(params)
        model.module.load_state_dict(checkpoint['model'], strict=True)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model_name = checkpoint['model_name']
        print('The { '+model_name+' } model load weight successful!')
        return start_epoch,best_acc
    else:
        print('The model path is not exists.')
        return 1,0

def save_model(model,params,epoch = 0,best_acc = 0,model_name = 'model_name'):
    checkpoint = {
        'best_acc': best_acc,    
        'epoch': epoch,
        'model': model.module.state_dict(),
        'model_name':model_name,
    }
    best_acc = round(best_acc,4)
    best_acc = str(best_acc)
    params =params.replace('version', best_acc)
    torch.save(checkpoint, params)
    print('save weight successful! Best score is:{}'.format(best_acc))

######## For train ###############
def train(train_loader, model, criterion1,optimizer, epoch, params):
    metric_monitor = MetricMonitor()
    model.train()
    stream = tqdm(train_loader)
    for i, (images, target,_) in enumerate(stream, start=1):
        images = images.cuda(non_blocking=params['non_blocking_'])
        target = target.cuda(non_blocking=params['non_blocking_'])
        target = target.permute(0, 3, 1, 2)
        output = model(images)
        qf = np.random.randint(low=71, high=95, size=None, dtype=int)
        jpeg = DiffJPEG(height=512, width=512, differentiable=True, quality=qf).cuda()
        output = jpeg(output)
        if i%50==0:
            res_orignal = target-images#target-images #abs(target-images)
            res_out = output-images
            psnr_score_list = psnr.psnr(output, target)
            psnr_score = torch.mean(psnr_score_list).detach().cpu().numpy()
            print(psnr_score.item())
            save_image(res_orignal, 'res_orignal.jpg',normalize=True)
            save_image(res_out, 'res_out.jpg',normalize=True)

        #output = torch.sigmoid(output)
        loss = criterion1(output, target)
        metric_monitor.update("Loss", loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        stream.set_description(
            "Epoch: {epoch}. Train. {metric_monitor} Time: {time}".format(epoch=epoch, metric_monitor=metric_monitor,time=time.strftime('%H:%M:%S', time.localtime()))
        )

######## For validate ###############
def validate(val_loader, model, params):
    model.eval()
    stream = tqdm(val_loader)
    # SSIM instantiable object
    ssim_criterion = ssim.SSIM().cuda()
    with torch.no_grad():
        psnr_list = []
        psnr_list_ = []
        #ssim_list = []
        for i, (images, target,_) in enumerate(stream, start=1):
            images = images.cuda(non_blocking=params['non_blocking_'])
            target = target.cuda(non_blocking=params['non_blocking_'])
            target = target.permute(0, 3, 1, 2)
            output = model(images)
            # PSNR function
            psnr_score = psnr.psnr(output, target)
            psnr_score = torch.mean(psnr_score).cpu().numpy()
            #ssim_score = ssim_criterion(output, target)
            psnr_list.append(psnr_score)
            #ssim_list.append(ssim_score)
            #psnr_score_ = psnr.psnr(images, target)
            #psnr_score_ = torch.mean(psnr_score_).cpu().numpy()
            #psnr_list_.append(psnr_score_)
            #ssim_score_ = ssim_criterion(images, target)


    avg_psnr = np.mean(psnr_list)
    #avg_psnr_ = np.mean(psnr_list_)
    #print('avg_psnr_:{}'.format(avg_psnr_))
    #print('avg_psnr:{}'.format(avg_psnr))
    #avg_ssim = np.mean(ssim_list)
    #socre = avg_psnr+ avg_ssim
    return avg_psnr

######## For predict ###############
def predict(val_loader, model, params,threshold):
    model.eval()
    stream = tqdm(val_loader)
    with torch.no_grad():
        avg_dice_list = []
        avg_iou_list = []
        for step, (batch_x_val,batch_y_val,w_s, h_s, name) in enumerate(stream, start=1):
            batch_x_val = batch_x_val.cuda(non_blocking=params['non_blocking_'])  
            batch_y_val = batch_y_val.cuda(non_blocking=params['non_blocking_'])  
            batch_y_val = batch_y_val.permute(0, 3, 1, 2)
            ############## 1 ##############
            output_val = model(batch_x_val)
            batch_x_val_h_flip = batch_x_val.clone().detach()
            batch_x_val_h_flip = torch.flip(batch_x_val_h_flip,[3])

            batch_x_val_v_flip = batch_x_val.clone().detach()
            batch_x_val_v_flip = torch.flip(batch_x_val_v_flip,[2])
            image_h_flip = model(batch_x_val_h_flip)
            image_v_flip = model(batch_x_val_v_flip)

            image_h_flip = torch.flip(image_h_flip,[3])
            image_v_flip = torch.flip(image_v_flip,[2])
            
            result_output_1 = (output_val+image_h_flip+image_v_flip)/3.
    
            result_output = result_output_1
            result_output = torch.sigmoid(result_output)
            for i in range(len(result_output)):
                orig_w = w_s[i]
                orig_h = h_s[i]
                result_output_ = F.interpolate(result_output[i:i+1], size=[orig_h,orig_w], mode="bicubic",align_corners=False)
                result_output[result_output >= threshold] = 1
                result_output[result_output < threshold] = 0
                str_ = name[i].split('/')[-1]
                name_str = str_.replace('.jpg', '.png')
                save_img_name = os.path.join(save_dir,name_str)
                save_image(result_output_, save_img_name)


######## For train_and_validate ###############
def train_and_validate(model, train_dataset,val_dataset, params,epoch_start = 1,best_acc=0):
    train_loader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=params["num_workers"],
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=params["test_batch_size"],
        shuffle=False,
        num_workers=params["num_workers"],
        pin_memory=True,
        drop_last=False,
    )
    # Define Loss 

    criterion = nn.MSELoss(reduction = 'sum').cuda()

    # Define optimizer 
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"])
    if params["mode"]=='train':
        for epoch in range(epoch_start, params["epochs"] + 1):
            train(train_loader, model,criterion,optimizer, epoch, params)
            # psnr_score = validate(val_loader,model,params)
            # #predict(val_loader,model,params,threshold=0.1)
            # #cur_acc,f1,iou = cal_score(val_gt_mask_dir,save_dir)
            # print('current score is:{} best score is:{}'.format(psnr_score,best_acc))
            # if psnr_score > best_acc:
            #     best_acc = psnr_score
            save_model(model,params["save_model_path"],epoch,best_acc,params["model_name"])
    elif params["mode"]=='val':
        threshold=0.1
        cur_acc,f1,iou = validate(val_loader,model,params,threshold=threshold)
        print('threshold:{} current score is:{} f1 score is:{} iou score is:{}'.format(threshold,cur_acc,f1,iou))

    elif params["mode"]=='predict':
        threshold=0.1
        predict(val_loader,model,params,threshold=threshold)
        cur_acc,f1,iou = cal_score(val_gt_mask_dir,save_dir)
        print('threshold:{} current score is:{} f1 score is:{} iou score is:{}'.format(threshold,cur_acc,f1,iou))

