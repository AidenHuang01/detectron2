import torch, detectron2

# Version inspections:
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from tqdm import tqdm
import pickle
from pathlib import Path
import os



# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# Constances
IMAGE_DIR = "./images/"
DATA_PATH = "/yucheng/experiment/"
OUTPUT_PATH = "/yucheng/output/"

# load model
cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_C4_1x.yaml")
predictor = DefaultPredictor(cfg)
print("load model successful")

test_list = os.listdir(DATA_PATH)
print(f"start detection under {DATA_PATH}")
for test_name in tqdm(test_list):
  images_path = DATA_PATH + test_name + "/mtlb-pngFromBag"
  images_list = os.listdir(images_path)
  # forward
  outputs = None
  print(f"detecting on {test_name} with {len(images_list)} frames:")
  for i in tqdm(range(len(images_list))):
    image_name = images_list[i]
    image_num = image_name[:4]
    image_path = DATA_PATH + test_name + "/mtlb-pngFromBag/" + image_name
    output_file_name = OUTPUT_PATH + test_name + "/2D_boxes/" + image_num + ".pickle"

    im = cv2.imread(image_path)
    outputs = predictor(im)
    instances = outputs["instances"]
    outputs["instances"] = instances[instances.pred_classes == 2]
    pred_boxes = outputs["instances"].pred_boxes.tensor.cpu().detach().numpy()
    pred_scores = outputs["instances"].scores.cpu().detach().numpy()
    pred_classes = outputs["instances"].pred_classes.cpu().detach().numpy()
    pred_dict_2D = {}
    pred_dict_2D["boxes"] = pred_boxes
    pred_dict_2D["scores"] = pred_scores
    pred_dict_2D["classes"] = pred_classes
    output_file = Path(output_file_name)
    output_file.parent.mkdir(exist_ok=True, parents=True)
    with open(output_file_name, 'wb') as handle:
        pickle.dump(pred_dict_2D, handle)