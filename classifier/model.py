from pathlib import Path

from ultralytics import YOLO
from clip import CLIP
from PIL import Image
import numpy as np
import json

ROOT = Path(__file__).resolve().parent
DEFAULT_YOLO_WEIGHT = ROOT.parent / "data" / "weights" / "detector" / "yolo_detector_best.pt"
DEFAULT_PROJECT_JSON = ROOT / "project" / "TSR_new.json"


class CLIP_TSR:
    def __init__(
        self,
        yolo_path=str(DEFAULT_YOLO_WEIGHT),
        project_path=str(DEFAULT_PROJECT_JSON),
    ):
        self.model_clip = CLIP()
        self.model_YOLO = YOLO(yolo_path)
        self.sign_types, self.captions = self.get_captions(project_path)

    # 读取项目的json文件，获取字条
    def get_captions(self,path):
        # 解析并且返回captions
        try:
            with open(path,"r",encoding='utf8') as fp:
                json_data = json.load(fp)
                sign_types = list(json_data.keys())
                captions = []
                for sign in sign_types:
                    captions.append(json_data[sign])
                return sign_types, captions
        # 文件路径错
        except FileNotFoundError:
            print(f"错误：未找到 JSON 文件 {path}")
        # 文件解析错误
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")

    # 分类
    def classify(self,image):
        probs = self.model_clip.detect_image(image, self.captions)
        probs = probs.tolist()[0]
        # 返回类别以及对应的类别id
        return probs.index(max(probs)), self.sign_types[probs.index(max(probs))], self.captions[probs.index(max(probs))]
    
    # 扫描
    def scan(self,image_path):
        image = Image.open(image_path)
        results = self.model_YOLO.predict(image_path, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        detection_results = []
        for box in boxes:
            # 裁剪图像
            x1, y1, x2, y2 = map(int, box)
            image_cropped = image.crop((x1, y1, x2, y2))

            # 调用分类函数
            index,category,_ = self.classify(image_cropped)
            detection_results.append([index, x1, y1, x2, y2, category])

        return image, detection_results