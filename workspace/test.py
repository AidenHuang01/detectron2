import os
import pickle

DATA_PATH = "/yucheng/experiment/"
OUTPUT_PATH = "/yucheng/output/"
test_list = os.listdir(DATA_PATH)

from pathlib import Path


for test_name in test_list:
    images_path = DATA_PATH + test_name + "/mtlb-pngFromBag"
    images = os.listdir(images_path)
    for img in images:
        detectron_output = OUTPUT_PATH + test_name + "/2D_boxes/" + img
        to_save = []
        output_file = Path(detectron_output)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        with open(detectron_output, 'wb') as f:
            pickle.dump(to_save, f)
        print(detectron_output)