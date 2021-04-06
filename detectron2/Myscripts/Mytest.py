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
register_coco_instances("my_dataset_train", {}, "../../../VisDrone-Dataset/VisDrone2019-DET-train/json_annotaions/drone_anno.json", "../../../VisDrone-Dataset/VisDrone2019-DET-train/images")
register_coco_instances("my_dataset_test", {}, "../../../VisDrone-Dataset/VisDrone2019-DET-test/json_annotaions/drone_anno.json", "../../../VisDrone-Dataset/VisDrone2019-DET-test/images")
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_test")
cfg.DATALOADER.NUM_WORKERS = 6
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.025
cfg.SOLVER.MAX_ITER = 500
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1024
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.OUTPUT_DIR = "./Myoutput"
cfg.MODEL.WEIGHTS = os.path.join("../Mymodel", "model_final.pth")  # path to the model we just trained

image_metadata = MetadataCatalog.get("my_dataset_test")
image_metadata.set(thing_classes=["1","2"])
predictor = DefaultPredictor(cfg)
flag = 0
for i in glob.glob(os.path.join("../../../VisDrone-Dataset/VisDrone2019-DET-test/images","*.jpg")):
    print(i)
    im = cv2.imread(i)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=image_metadata, 
                   scale=0.5, 
                #    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    print(outputs["instances"])
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # print(outputs["instances"])
    pred_image = out.get_image()[:,:,::-1]
    cv2.imwrite(os.path.join("./Myoutput",str(flag)+".jpg"),pred_image)
    flag += 1
    if flag == 10:
        break