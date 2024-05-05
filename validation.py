import tensorflow as tf

import time
import os
import cv2
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="1"
from configuration import *
import DIW
import augmentation as Aug


def print_model_summary(network):
    network.build(input_shape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    network.summary()

validation_folder = '/home/kdg/dev/dataset/depth/RedWeb_V1/validation/Imgs/'
# validation_folder = '/home/kdg/dev/tmp/test_tfrecords/'
output_folder = '/home/kdg/dev/tmp/test_depth/val_1.0/'

# GPU settings
gpus = tf.config.list_physical_devices("GPU")
# if gpus:
diw = DIW.DIW((1,IMAGE_HEIGHT,IMAGE_WIDTH,3))
print_model_summary(network=diw)
diw.load_weights(filepath=load_model_dir+"249-epoch")
print("Successfully load weights!")

file_list = os.listdir(validation_folder)
for file_ in file_list:
    input_img = cv2.imread(validation_folder+file_)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    input_img = cv2.resize(input_img,(224,224))
    input_img = np.expand_dims(input_img, axis=0)
    input_img = ((input_img/255.)-0.5)*2.
    _, _, logits = diw(input_img, training=False)
    ymin = tf.reduce_min(logits)
    ymax = tf.reduce_max(logits - ymin)
    depth = (logits - ymin) / ymax
    cv2.imwrite(output_folder+file_,np.uint8(np.squeeze(depth*255.)))