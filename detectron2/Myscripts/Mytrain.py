import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np 
import os, json, cv2, random, glob
# from google_auth_oauthlib import cv2_imshow

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.checkpoint import DetectionCheckpointer

register_coco_instances("my_dataset_train", {}, r"F:\coco\annotations\instances_train2017.json", r"F:\coco\train2017")
register_coco_instances("my_dataset_val", {}, r"F:\coco\annotations\instances_val2017.json", r"F:\coco\val2017")



# for image in glob.glob(r"../../TestImage/*.jpg"):
if __name__ == '__main__':
    # im = cv2.imread(r"C:\Users\Actstone\Pictures\index.png")
    # cv2_imshow(im)
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 6
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.25
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 1000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1024
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
    cfg.OUTPUT_DIR = r"F:\coco"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok= True)
    trainer = DefaultTrainer(cfg)
    # model = trainer.build_model(cfg)
    # DetectionCheckpointer(model).load(cfg.OUTPUT_DIR+r"\model_final.pth")
    trainer.resume_or_load(resume=False)
    trainer.train()



    # predictor = DefaultPredictor(cfg)
    # outputs = predictor(im)
    # print(outputs["instances"])
    # v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale = 1.2)
    # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # out.save(r"C:\Users\Actstone\Pictures\index0.png")