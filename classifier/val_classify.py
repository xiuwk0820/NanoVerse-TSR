from PIL import Image
from model import CLIP_TSR
import json
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    model = CLIP_TSR(project_path="project/TSR_new.json")
    project_path = "test_images/classify"
    json_name = "new_test_data.json"
    with open(project_path+"/"+json_name) as fp:
        json_data = json.load(fp)
    corr  = 0
    false = 0
    for label in tqdm(json_data):
        image = Image.open(project_path+"/"+label["image"])
        _,_, result = model.classify(image)
        # print(result)
        # print(label["caption"][0])
        if result == label["caption"][0]:
            corr += 1
        else:
            false += 1
    print("Accuracy:",(corr/(corr+false)))

