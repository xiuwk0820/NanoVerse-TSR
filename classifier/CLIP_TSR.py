from ultralytics import YOLO
from clip import CLIP

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import numpy as np
import os
import yaml

class Clip_TSR:
    ''' 构造函数 '''
    def __init__(self,YOLO_path="yolo11s.pt", yaml_file_path=""):
        self.model_clip = CLIP()
        self.model_YOLO = YOLO(YOLO_path)
        self.set_caption_from_yaml(yaml_file_path)
    
    ''' 读取项目文件 '''
    def set_caption_from_yaml(self, yaml_file_path):
        # 读取 YAML 文件内容
        try:
            with open(yaml_file_path, "r") as file:
                config = yaml.safe_load(file)
                # 获取 "captions" 列表内容
                self.caption = config.get("captions", [])
                if not self.caption:
                    raise ValueError("YAML 文件中没有找到 'captions' 列表内容")
        except FileNotFoundError:
            print(f"错误：未找到 YAML 文件 {yaml_file_path}")
        except yaml.YAMLError as e:
            print(f"YAML 解析错误: {e}")

    '''利用clip分类'''
    def classify(self, image):
        probs = self.model_clip.detect_image(image, self.caption)
        probs = list(probs)
        return self.caption[probs.index(max(probs))]

    '''检测物体主函数 (不负责保存)'''
    def detect_single(self, image_path):
        image = Image.open(image_path).convert("RGB")
        results = self.model_YOLO.predict(source=np.array(image), verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        detection_results = []
        for box in boxes:
            # 裁剪图像
            x1, y1, x2, y2 = map(int, box)
            image_cropped = image.crop((x1, y1, x2, y2))

            # 调用分类函数
            category = self.classify(image_cropped)
            detection_results.append([x1, y1, x2, y2, category])

        return image, detection_results
    
    ''' 保存结果 '''
    def save_results(self, image_path, annotated_image, detection_results, save_txt=False, visual=False):
        # 保存YOLO格式的txt文件
        if save_txt:
            txt_file_path = os.path.join(self.labels_dir, os.path.basename(image_path).replace(".jpg", ".txt"))
            with open(txt_file_path, "w") as f:
                for x1, y1, x2, y2, category in detection_results:
                    ''' !!!!! '''
                    # class_idx = list(self.class_names.values()).index(category)
                    class_idx = self.caption.index(category)
                    # 归一化处理
                    img_width, img_height = annotated_image.size
                    x_center = ((x1 + x2) / 2) / img_width
                    y_center = ((y1 + y2) / 2) / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height
                    f.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        # 保存标注后的图像
        if visual:
            draw = ImageDraw.Draw(annotated_image)
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # 替换为系统中的字体路径
            font_size = 16
            try:
                font = ImageFont.truetype(font_path, font_size)
            except IOError:
                font = ImageFont.load_default()
            
            for x1, y1, x2, y2, category in detection_results:
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                draw.text((x1, max(0, y1 - 20)), category, fill="red", font=font)
            visual_path = os.path.join(self.images_dir, os.path.basename(image_path).replace(".jpg", "_annotated.jpg"))
            annotated_image.save(visual_path)

    # 检测函数
    def detect(self, image, save_txt=False, visual=True):
        # 创建结果保存目录结构
        base_dir = "runs"

        # 自动检测可用的detect文件夹名称
        detect_dir = os.path.join(base_dir, "detect")
        suffix = 0
        while os.path.exists(detect_dir):
            suffix += 1
            detect_dir = os.path.join(base_dir, f"detect{suffix}")
        os.makedirs(detect_dir, exist_ok=True)

        self.labels_dir = os.path.join(detect_dir, "labels")
        self.images_dir = os.path.join(detect_dir, "images")
        os.makedirs(self.labels_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)

        # 如果是个文件夹
        if os.path.isdir(image):
            # 遍历文件夹中的所有图片文件
            for file_name in tqdm(os.listdir(image)):
                file_path = os.path.join(image, file_name)
                if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    annotated_image, detection_results = self.detect_single(file_path)
                    self.save_results(file_path, annotated_image, detection_results, save_txt=save_txt, visual=visual)
            print(f"检测结果已经保存在：{detect_dir}")
        # 如果是个文件
        elif os.path.isfile(image):
            if image.lower().endswith((".jpg", ".jpeg", ".png")):
                annotated_image, detection_results = self.detect_single(image)
                self.save_results(image, annotated_image, detection_results, save_txt=save_txt, visual=visual)
            else:
                print("不支持的文件格式，请提供图片文件 (.jpg, .jpeg, .png)")
            print(f"检测结果已经保存在：{detect_dir}")
        else:
            print("输入路径既不是文件也不是文件夹，请检查路径是否正确")
    
model = Clip_TSR(yaml_file_path="project/cup.yaml")
model.detect('images',save_txt=True,visual=True)