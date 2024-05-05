
import numpy as np

# training parameters

EPOCHS = 250
BATCH_SIZE = 4
NUM_TRAIN = 3348
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
CHANNELS = 3
NUM_SAMPLE=3000   # number of pixels which selected on one image
THRESHOLD=1.0     # range of same depth
N_THRESHOLD = 15
val_THRESHOLD = 0.1*np.array(range(N_THRESHOLD))
LEARNING_RATE=0.0004

load_weights_before_training = False
load_weights_from_epoch = 0
save_frequency = 5

test_picture_dir = "path/to/root/"
test_images_during_training = False
training_results_save_dir = "./test_pictures/"
test_images_dir_list = ["", ""]

# When the iou value of the anchor and the real box is less than the IoU_threshold,
# the anchor is divided into negative classes, otherwise positive.
IOU_THRESHOLD = 0.6

# generate anchor
ASPECT_RATIOS = [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0, 1.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0, 1.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0, 1.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0, 1.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0, 1.0]]
MIN_SCALE = 0.2
MAX_SCALE = 0.95

# focal loss
alpha = 0.25
gamma = 2.0

reg_loss_weight = 0.5

# dataset
PASCAL_VOC_DIR = "path/to/pascalvoc/public/VOC2012/"

OBJECT_CLASSES = {}
# NUM_CLASSES = len(OBJECT_CLASSES) + 1
NUM_CLASSES = 2
for k in range(NUM_CLASSES):
    if k==0:
        continue
    OBJECT_CLASSES[str(k)]=k

TXT_DIR = "./train.txt"


MAX_BOXES_PER_IMAGE = 100


# nms
NMS_IOU_THRESHOLD = 0.1
CONFIDENCE_THRESHOLD = 0.5
MAX_BOX_NUM = 100


# directory of saving model
load_model_dir = 'path/to/git/light_diw/logs/RedWeb_remove_th1.0/'
model_save_dir = 'path/to/git/light_diw/logs/RedWeb_remove_th1.0_val/'
TFRECORDS = 'path/to/tfrecords/tfrecords/depth/RedWeb_V1/'

