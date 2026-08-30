from model import CLIP_TSR
from tqdm import tqdm
import numpy as np
import cv2
import os

import warnings
warnings.filterwarnings("ignore")

# 保存结果函数
def save_result(image, labels, saved_images_path, saved_labels_path, file_name):
    labels_yolo = []
    for label in labels:
        index,x1,y1,x2,y2,category = label[0],label[1],label[2],label[3],label[4],label[5]
        w,h = 2048,2048
        new_x = str((x1 + x2)/(2*w))
        new_y = str((y1 + y2)/(2*h))
        new_w = str((x2 - x1)/w)
        new_h = str((y2 - y1)/h)
        labels_yolo.append([index,new_x,new_y,new_w,new_h])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, category, (x1, y1 - 10), font, 0.9, (0, 255, 0), 2)
    image = image[:, :, [2,1,0]]
    # 保存带标框的图片
    cv2.imwrite(saved_images_path+"\\"+file_name,image)
    # 保存对应的txt文件
    label_txt_path = saved_labels_path + "\\" + file_name.split('.')[0] + '.txt'  # 使用文件名生成txt文件路径
    with open(label_txt_path, 'w') as f:
        for label in labels_yolo:
            f.write(f"{int(label[0])} {label[1]} {label[2]} {label[3]} {label[4]}\n")

# 检测函数
def detect(model,image):
    # 创建保存的文件夹
    base_dir = "runs"
    detect_dir = os.path.join(base_dir, "detect")
    suffix = 0
    while os.path.exists(detect_dir):
        suffix += 1
        detect_dir = os.path.join(base_dir, f"detect{suffix}")
    os.makedirs(detect_dir, exist_ok=True)
    labels_dir = os.path.join(detect_dir, "labels")
    images_dir = os.path.join(detect_dir, "images")
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    # 如果只是图片
    if os.path.isdir(image):
        # 遍历文件夹中的所有图片文件
        for file_name in tqdm(os.listdir(image)):
            # file_path = os.path.join(image, file_name)
            if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                # 检测
                origin_image, results = model.scan(image+"/"+file_name)
                image_numpy = np.array(origin_image)
                save_result(image_numpy, results, images_dir, labels_dir, file_name)
        print(f"检测结果已经保存在：{detect_dir}")
    # 如果是个文件
    elif os.path.isfile(image):
        if image.lower().endswith((".jpg", ".jpeg", ".png")):
            # 检测
            origin_image, results = model.scan(image)
            image_numpy = np.array(origin_image)
            save_result(image_numpy, results, images_dir, labels_dir, image)
        else:
            print("不支持的文件格式，请提供图片文件 (.jpg, .jpeg, .png)")
        print(f"检测结果已经保存在：{detect_dir}")
    else:
        print("输入路径既不是文件也不是文件夹，请检查路径是否正确")
    
# 运行主函数
def main():
    model = CLIP_TSR()
    image = '/test_images/detect/images'
    detect(model,image)

if __name__ == "__main__":
    main()