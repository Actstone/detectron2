from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
import random
import glob
import cv2
import os
register_coco_instances("my_dataset_train", {}, r"F:\coco\annotations\instances_train2017.json", r"F:\coco\train2017")
register_coco_instances("my_dataset_test", {}, r"F:\coco\annotations\instances_test2017.json", r"F:\coco\test2017")
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_test")
cfg.DATALOADER.NUM_WORKERS = 4
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.25
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.25   # set a custom testing threshold
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 600
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6
cfg.OUTPUT_DIR = r"F:\coco"
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained

image_metadata = MetadataCatalog.get("my_dataset_test")
image_metadata.set(thing_classes=["1","2","3","4","5"])#["铁壳打火机","黑钉打火机","刀具","电源和电池","剪刀"]
predictor = DefaultPredictor(cfg)
flag = 0
for i in glob.glob(os.path.join(cfg.OUTPUT_DIR,"test2017/*.jpg")):
    im = cv2.imread(i)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=image_metadata, 
                   scale=0.5, 
                #    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    print(outputs["instances"])
    pred_image = out.get_image()[:,:,::-1]
    cv2.imwrite(os.path.join("f:/coco/Results/",str(flag)+".jpg"),pred_image)
    flag += 1
    if flag == 3:
        break