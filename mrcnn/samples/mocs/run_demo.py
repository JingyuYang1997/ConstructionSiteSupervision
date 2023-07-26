import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/mocs"))  # To find local version
import mocs


# Directory to save logs and trained model
MODEL_DIR = os.path.join("logs")

# Local path to trained weights file
MOCS_MODEL_PATH = os.path.join('./logs/mask_rcnn_mocs_0130.h5')

# Directory of images to run detection on
# IMAGE_DIR = os.path.join('../../../mocs/instances_val/')
IMAGE_DIR = './images'


class InferenceConfig(mocs.MOCSConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(MOCS_MODEL_PATH, by_name=True)

class_names = ['BG', 'Worker','Static crane','Hanging head','Crane','Roller','Bulldozer',
              'Excavator','Truck','Loader','Pump truck','Concrete mixer','Pile driving','Other vehicle']

file_names = next(os.walk(IMAGE_DIR))[2]
for file_name in file_names:
    # file_name = random.choice(file_names)
    image = skimage.io.imread(os.path.join(IMAGE_DIR, file_name))

    # Run detection
    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]
    print(file_name)
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                class_names, r['scores'],save_path='./new_results/{}.png'.format(file_name.split('.')[-2]))