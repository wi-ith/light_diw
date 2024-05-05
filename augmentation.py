from configuration import *
import tensorflow as tf
import numpy as np

def _parse_function(example_proto):
    keys_to_features = {
        'image/height':
            tf.io.FixedLenFeature((), tf.int64),
        'image/width':
            tf.io.FixedLenFeature((), tf.int64),
        'image/encoded':
            tf.io.FixedLenFeature((), tf.string, default_value=''),
        'image/mask_encoded':
            tf.io.FixedLenFeature((), tf.string, default_value=''),
    }
    return tf.io.parse_single_example(example_proto, keys_to_features)


def decode_jpeg(image_buffer, channels, scope=None):
    with tf.name_scope(name=scope):
        image = tf.image.decode_image(image_buffer, channels)
        return image

def inputs(parsed_tfr):
    if not len(parsed_tfr['image/encoded']):
        batch_size = BATCH_SIZE

    # with tf.device('/cpu:0'):
    images_batch, boxes_batch = batch_inputs(parsed_tfr, len(parsed_tfr['image/encoded']))

    return images_batch, boxes_batch

def batch_inputs(parsed_tfr, batch_size):
    batch_labels=[]
    batch_images=[]
    for q in range(batch_size):
        one_image = decode_jpeg(parsed_tfr['image/encoded'][q], 3, scope='decode_jpeg')
        one_label = decode_jpeg(parsed_tfr['image/mask_encoded'][q], 1, scope='decode_mask_jpeg')
        one_label = tf.cast(one_label, dtype=tf.float32)
        one_image = tf.image.resize(one_image, tf.stack([IMAGE_HEIGHT, IMAGE_WIDTH]))
        one_image = tf.image.convert_image_dtype(one_image, dtype=tf.float32)
        one_image = (one_image/255.-0.5)*2.
        one_label = tf.cast(tf.image.resize(one_label, tf.stack([IMAGE_HEIGHT, IMAGE_WIDTH])),dtype=tf.int32)
        batch_images.append(one_image)
        batch_labels.append(one_label)
    batch_images = tf.stack(batch_images)
    batch_labels = tf.squeeze(tf.stack(batch_labels))
    return batch_images, batch_labels

