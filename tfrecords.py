import os
import io
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import random


from PIL import Image
from utils import dataset_util


def create_example(root_path, image_name):
    with tf.gfile.GFile(root_path+'Imgs/'+image_name, 'rb') as fid:
        encoded_image = fid.read()
    encoded_image_io = io.BytesIO(encoded_image)
    image = Image.open(encoded_image_io)
    width, height = image.size

    with tf.gfile.GFile(root_path+'RDs/'+image_name[:-4]+'.png', 'rb') as fid_mask:
        encoded_mask = fid_mask.read()
    encoded_mask_io = io.BytesIO(encoded_mask)
    mask = Image.open(encoded_mask_io)

    # if image.format != 'JPEG' or mask.format != 'JPEG':
    #     raise ValueError('Image format not JPEG')

    # create TFRecord Example
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(image_name.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image),
        'image/mask_encoded': dataset_util.bytes_feature(encoded_mask),
    }))
    return example


def main(_):
    image_folder = '/home/kdg/dev/dataset/depth/RedWeb_V1/'
    save_folder = '/home/kdg/dev/tfrecords/depth/RedWeb_V1/'
    train_path = image_folder+ 'train/'
    val_path = image_folder + 'validation/'
    writer_train = tf.python_io.TFRecordWriter(save_folder + 'train.record')
    writer_val = tf.python_io.TFRecordWriter(save_folder + 'val.record')
    init = (tf.global_variables_initializer(), tf.local_variables_initializer())
    sess = tf.Session()
    sess.run(init)
    train_image_list = os.listdir(train_path+'Imgs/')
    val_image_list = os.listdir(val_path + 'Imgs/')
    random.shuffle(train_image_list)  # shuffle files list
    random.shuffle(val_image_list)  # shuffle files list
    i = 1
    tst = 0  # to count number of images for evaluation
    trn = 0  # to count number of images for training

    # seg_list = os.listdir(root_folder)
    # train_folder = root_folder+seg_list[0]+'/'
    # val_folder = root_folder + seg_list[1] + '/'


    for a, img in enumerate(train_image_list):
        if a%200==0:
            print(a,' / ',len(train_image_list),' files done.')
        example = create_example(train_path,img)
        if example == None:
            continue

        writer_train.write(example.SerializeToString())
        trn = trn + 1
        i = i + 1

    for b, img in enumerate(val_image_list):
        if b%200==0:
            print(b,' / ',len(val_image_list),' files done.')
        example = create_example(val_path,img)
        if example == None:
            continue
        writer_val.write(example.SerializeToString())
        tst = tst + 1
        i = i + 1

    writer_val.close()
    writer_train.close()
    print('Successfully converted dataset to TFRecord.')
    print('training dataset: # ')
    print(trn)
    print('test dataset: # ')
    print(tst)

if __name__ == '__main__':
    tf.app.run()