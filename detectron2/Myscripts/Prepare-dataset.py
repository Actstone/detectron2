"""Prepare datasets"""
from detectron2.structures import BoxMode
import os
import json
def Get_class_dicts(img_dir):
    json_file = os.path.join(img_dir, "instances_train2017.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)
        # print(imgs_anns.values())
    
    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        print(v["file_name"])
        # record = {}

        # filename = os.path.join(img_dir, v["images"])
        # height, width = cv2.imread(filename).shape[:2]

if __name__ == '__main__':
    img_dir = "f:/coco/annotations"
    Get_class_dicts(img_dir)