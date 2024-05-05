import tensorflow as tf

import time
import os
import glob
import io
import cv2
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="1"
from configuration import *
from loss import Loss
from utils import eval
import DIW
import augmentation as Aug


def print_model_summary(network):
    network.build(input_shape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    network.summary()

if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.list_physical_devices("GPU")
    # if gpus:
    #     for gpu in gpus:
    #         tf.config.experimental.set_memory_growth(gpu, True)

    dataset_dir = TFRECORDS
    tf_record_pattern = glob.glob(dataset_dir + '*train*')
    val_pattern = glob.glob(dataset_dir + '*val*')
    dataset = tf.data.Dataset.from_tensor_slices([tf_record_pattern])
    val_dataset = tf.data.Dataset.from_tensor_slices([val_pattern])
    def _parse(x):
        x = tf.data.TFRecordDataset(x)
        return x
    SHUFFLE_BUFFER = 64
    PREFETCH = 256
    dataset = dataset.interleave(_parse, cycle_length=32,
                                 block_length=1,
                                 num_parallel_calls=32,
                                 deterministic=True)
    dataset = dataset.shuffle(SHUFFLE_BUFFER)
    dataset = dataset.map(Aug._parse_function, num_parallel_calls=32)
    final_dataset = dataset.prefetch(buffer_size=1000)
    final_dataset = final_dataset.cache()
    final_dataset = final_dataset.batch(BATCH_SIZE)
    val_dataset_iter = val_dataset.batch(4)

    diw = DIW.DIW((BATCH_SIZE,IMAGE_HEIGHT,IMAGE_WIDTH,3))
    print_model_summary(network=diw)

    train_summary_writer = tf.summary.create_file_writer(model_save_dir)
    @tf.function
    def summary_save(step, lr, loss_aux1, loss_aux2, final_loss):
        with train_summary_writer.as_default():
            tf.summary.scalar("learning_rate", lr, step=step)
            tf.summary.scalar("loss_aux1", loss_aux1, step=step)
            tf.summary.scalar("loss_aux2", loss_aux2, step=step)
            tf.summary.scalar("final_loss", final_loss, step=step)

    @tf.function
    def val_summary_save(step, whdr, rmse):
        with train_summary_writer.as_default():
            tf.summary.scalar("whdr", whdr, step=step)
            tf.summary.scalar("rmse", rmse, step=step)

    @tf.function
    def summary_image_save(step, prediction, input_image):
        with train_summary_writer.as_default():
            tf.summary.image("prediction", prediction, step=step)
            tf.summary.image("input_image", input_image, step=step)

    if load_weights_before_training:
        # checkpoint = tf.train.Checkpoint(model=diw)
        # status = checkpoint.restore(tf.train.latest_checkpoint(load_model_dir))
        # checkpoint = tf.train.Checkpoint(diw)
        # checkpoint.restore(load_model_dir).expect_partial()
        diw.load_weights(filepath=load_model_dir+'249-epoch')
        print("Successfully load weights!")
    else:
        load_weights_from_epoch = -1
    ##########################ckpt load compare
    # loss
    loss = Loss(threshold=THRESHOLD, num_samples=NUM_SAMPLE,mode='train')
    loss_val = Loss(threshold=THRESHOLD, num_samples=NUM_SAMPLE,mode='val')

    # optimizer
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=LEARNING_RATE,
                                                                 decay_steps=int((NUM_TRAIN/BATCH_SIZE)*50),
                                                                 decay_rate=0.5)
    lr_decay_rate = 0.5
    lr_decay_epochs = 50

    # metrics
    loss_metric = tf.metrics.Mean()
    cls_loss_metric = tf.metrics.Mean()
    reg_loss_metric = tf.metrics.Mean()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=model_save_dir,histogram_freq=10)
    for epoch in range(load_weights_from_epoch + 1, EPOCHS):
        lr_ = LEARNING_RATE*((lr_decay_rate)**(epoch//lr_decay_epochs))
        optimizer = tf.optimizers.Adam(learning_rate=lr_)
        start_time = time.time()
        for step, parsed_record in enumerate(final_dataset):

            if type(parsed_record)==tuple:
                parsed_record=parsed_record[0]

            images, labels = Aug.inputs(parsed_record)

            with tf.GradientTape() as tape:
                pre_logit1, pre_logit2, logits = diw(images, training=True)
                # print(pre_logit1.shape, pre_logit2.shape)
                loss_pre1 = loss.calc_loss(pre_logit1,labels)
                loss_pre2 = loss.calc_loss(pre_logit2,labels)
                loss_logits = loss.calc_loss(logits,labels)
                total_loss = loss_pre1*0.25 + loss_pre2*0.25 + loss_logits*0.5
                # if epoch ==0:
                # print(images.shape)
                # save_img = np.array(tf.cast((images[0, :, :, :]+1)*127.5, dtype=tf.uint8))
                # save_label = np.array(tf.cast(logits[0, :, :, :], dtype=tf.uint8))
                # ymin = tf.reduce_min(save_label)
                # ymax = tf.reduce_max(save_label - ymin)
                # depth = (save_label - ymin) / ymax
                # # print(np.max(depth),np.min(depth))
                # cv2.imwrite('./test_depth/'+str(step)+'_image.jpg',save_img)
                # cv2.imwrite('./test_depth/'+str(step)+'_label.jpg', np.uint8(depth*255))
                gradients = tape.gradient(total_loss, diw.trainable_variables)
                optimizer.apply_gradients(grads_and_vars=zip(gradients, diw.trainable_variables))
                loss_metric.update_state(values=total_loss)
            time_per_step = (time.time() - start_time) / (step + 1)
            if step%10==0:
                print("Epoch: {}/{}, step: {}/{}, {:.2f}s/step, loss: {:.5f}"
                      .format(epoch,
                              EPOCHS,
                              step,
                              tf.math.ceil(NUM_TRAIN / BATCH_SIZE),
                              time_per_step,
                              loss_metric.result()))
                summary_save(int(epoch*(NUM_TRAIN/BATCH_SIZE))+step,
                             lr_,
                             loss_pre1*0.25,
                             loss_pre2*0.25,
                             loss_logits*0.5)
            if step % 100 == 0:
                save_img = np.array(tf.cast((images[0, :, :, :]+1)*127.5, dtype=tf.uint8))
                save_label = np.array(tf.cast(logits[0, :, :, :], dtype=tf.uint8))
                ymin = tf.reduce_min(save_label)
                ymax = tf.reduce_max(save_label - ymin)
                depth = (save_label - ymin) / ymax
                # print(np.max(depth),np.min(depth))
                # cv2.imwrite('./test_depth/'+str(step)+'_image.jpg',save_img)
                # cv2.imwrite('./test_depth/'+str(step)+'_label.jpg', np.uint8(depth*255))
                summary_image_save(int(epoch*(NUM_TRAIN/BATCH_SIZE))+step,
                                   np.expand_dims(np.uint8(depth * 255),axis=0),
                                   np.expand_dims(save_img,axis=0)
                                   )
        diw.save_weights(filepath=model_save_dir+"{}-epoch".format(epoch))
        total_score = 0
        total_val_number = 0
        total_rmse = 0
        for val_step, val_parsed_record in enumerate(val_dataset_iter):
            if type(val_parsed_record)==tuple:
                val_parsed_record=val_parsed_record[0]
            num_exmple = val_parsed_record['image/encoded'].shape[0]
            total_val_number += num_exmple
            images, labels = Aug.inputs(val_parsed_record)
            _, _, val_logit = diw(images, training=False)
            for b_ in range(num_exmple):
                val_output = np.array(tf.cast(val_logit[b_,:,:,0], dtype=tf.uint8))
                ymin = tf.reduce_min(val_output)
                ymax = tf.reduce_max(val_output - ymin)
                one_val_depth = (val_output - ymin) / ymax
                one_label = np.float32(labels[b_,:,:])/255.
                one_rmse = np.mean((one_val_depth-one_label)**2)**(0.5)
                total_rmse += one_rmse
            total_score += loss_val.calc_loss(val_logit,labels)
        total_rmse = total_rmse/total_val_number
        print('total_score : ',total_score/total_val_number,'   ','rmse : ',total_rmse)
        val_summary_save(int(epoch*(NUM_TRAIN/BATCH_SIZE))+step,total_score/total_val_number,total_rmse)



