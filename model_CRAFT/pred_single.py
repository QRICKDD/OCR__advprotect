from Tools.CRAFTTools import *

img_test_1 = r"F:\OCR-TASK\Wsf\data\test\002.png"
img_test_3 = r"F:\OCR-TASK\Wsf\data\test\003.png"
img_test_2 = r"F:\OCR-TASK\Wsf\data\test\006.png"
if __name__ == '__main__':
    from Tools.Imagebasetool import *
    from AllConfig.GConfig import test_img_path,CRAFT_device,test_device
    #加载模型
    CRAFTnet=load_CRAFTmodel(device=test_device)
    #加载图片
    # img_path=test_img_path
    img_path=img_test_3
    img=img_read(img_path)
    img=img.to(test_device)

    #预测结果
    score_text,score_link,target_ratio = get_CRAFT_pred(CRAFTnet,img=img,square_size=1280,
                                    device=GConfig.test_device,is_eval=False)
    boxes=get_CRAFT_box(score_text,score_link,target_ratio,
                        text_threshold=0.7,link_threshold=0.4,low_text=0.4)
    # 保存结果到cv图片
    img = cv2.imread(img_path)
    cv2.imwrite(r"..\result_save\test_save\craft_orgin.jpg",img)
    CRAFT_draw_box(img,boxes=boxes,save_path=r"..\result_save\test_save\craft_boxes.jpg")
