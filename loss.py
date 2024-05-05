import numpy as np
import tensorflow as tf
from configuration import *
import cv2
# tmp_label = tf.zeros([32,224,224,1])

class Loss():
    def __init__(self, threshold, num_samples, mode):
        self.mode = mode
        self.THRESHOLD = threshold
        self.NUM_SAMPLES = num_samples
        self.record = {
            'n_threshold': N_THRESHOLD,
            'eq_correct_count': np.zeros(N_THRESHOLD),
            'not_eq_correct_count': np.zeros(N_THRESHOLD),
            'eq_count': 0,
            'not_eq_count': 0,
            'threshold': 0.1 * np.array(range(N_THRESHOLD)),
            'WKDR': np.zeros((N_THRESHOLD, 4))
        }

    def get_random_sym_pairs(self, input_shape):
        sym_max_x = tf.cast(input_shape[2] / 2,dtype=tf.int32)
        sym_max_y = input_shape[1]
        sym_left_x = tf.random.uniform(shape=(), minval=0, maxval=sym_max_x, dtype=tf.int32)
        sym_right_x = input_shape[2] - sym_left_x - 1
        sym_y = tf.random.uniform(shape=(), minval=0, maxval=sym_max_y, dtype=tf.int32)
        sym_coordi = [[sym_y, sym_left_x], [sym_y, sym_right_x]]
        return sym_coordi

    def get_random_csct_pairs(self, input_shape):
        # csrt => constraint
        csrt_max_x = input_shape[2]
        csrt_max_y = input_shape[1]

        csrt_x1 = tf.random.uniform(shape=(), minval=0, maxval=csrt_max_x, dtype=tf.int32)
        csrt_y1 = tf.random.uniform(shape=(), minval=0, maxval=csrt_max_y, dtype=tf.int32)

        csrt_x2 = tf.random.uniform(shape=(), minval=0, maxval=csrt_max_x - 1, dtype=tf.int32)
        csrt_y2 = tf.random.uniform(shape=(), minval=0, maxval=csrt_max_y - 1, dtype=tf.int32)

        if csrt_x1 == csrt_x2 and csrt_y1 == csrt_y2:
            x_or_y = tf.random.uniform(shape=())
            if x_or_y > 0.5:
                if csrt_x1 > 0:
                    plus_minus = tf.random.uniform(shape=())
                    if plus_minus > 0.5:
                        csrt_x2 += 1
                    else:
                        csrt_x2 -= 1
                else:
                    csrt_x2 += 1
            else:
                if csrt_y1 > 0:
                    plus_minus = tf.random.uniform(shape=())
                    if plus_minus > 0.5:
                        csrt_y2 += 1
                    else:
                        csrt_y2 -= 1
                else:
                    csrt_y2 += 1

        csrt_coordi = [[csrt_y1, csrt_x1], [csrt_y2, csrt_x2]]
        return csrt_coordi

    tmp_output = tf.random.uniform(shape=([32,224,224,1]),minval=-20,maxval=20)


    def relative_loss(self, rand_pairs):
        """
        Returns:
            1-D Tensor - loss
        """
        za = rand_pairs[:,0]
        zb = rand_pairs[:,1]
        relation = rand_pairs[:,2]
        mask = tf.abs(relation)

        return mask * tf.math.log(1 + tf.exp(-relation * (za - zb))) + (1 - mask) * (za - zb) * (za - zb)


    def csrt_pairs(self, anchor_reshape, unique_array, depth, depth_label):
        first_pairs = tf.gather(anchor_reshape, unique_array[:, 0])
        second_pairs = tf.gather(anchor_reshape, unique_array[:, 1])
        first_x = tf.gather_nd(depth, first_pairs)
        second_x = tf.gather_nd(depth, second_pairs)

        first_y = tf.gather_nd(depth_label, first_pairs)
        second_y = tf.gather_nd(depth_label, second_pairs)

        label_relation_high = tf.cast(tf.greater_equal(first_y - second_y, THRESHOLD),
                                      dtype=tf.float32)
        label_relation_low = tf.cast(tf.less_equal(first_y - second_y, -1 * THRESHOLD),
                                     dtype=tf.float32)
        labels = tf.cast(label_relation_high - label_relation_low, dtype=tf.float32)

        output = tf.stack([first_x, second_x, labels], axis=-1)

        return output

    def csrt_pairs_val(self, anchor_reshape, unique_array, depth, depth_label):
        first_pairs = tf.gather(anchor_reshape, unique_array[:, 0])
        second_pairs = tf.gather(anchor_reshape, unique_array[:, 1])
        first_x = tf.gather_nd(depth, first_pairs)
        second_x = tf.gather_nd(depth, second_pairs)

        first_y = tf.gather_nd(depth_label, first_pairs)
        second_y = tf.gather_nd(depth_label, second_pairs)

        label_relation_high = tf.cast(tf.greater_equal(first_y - second_y, THRESHOLD),
                                      dtype=tf.float32)
        label_relation_low = tf.cast(tf.less_equal(first_y - second_y, -1 * THRESHOLD),
                                     dtype=tf.float32)
        labels = tf.cast(label_relation_high - label_relation_low, dtype=tf.float32)


        for k, one_thr in enumerate(self.record['threshold']):
            logit_relation_high = tf.cast(tf.greater_equal(first_x - second_x, one_thr),
                                          dtype=tf.float32)
            logit_relation_low = tf.cast(tf.less_equal(first_x - second_x, -1 * one_thr),
                                         dtype=tf.float32)
            val_output = tf.cast(logit_relation_high - logit_relation_low, dtype=tf.float32)

            correct_mask = tf.cast(tf.equal(labels,val_output),dtype=tf.float32)

            correct_labels = (labels+2)*correct_mask

            self.record['eq_correct_count'][k] = tf.reduce_sum(tf.cast(tf.equal(correct_labels, 2.),dtype=tf.int32))

            self.record['not_eq_correct_count'][k] = tf.reduce_sum(tf.cast(tf.logical_or(tf.equal(correct_labels, 1.),tf.equal(correct_labels, 3.)),dtype=tf.int32))

        self.record['eq_count'] = tf.reduce_sum(tf.cast(tf.equal(labels+2, 2.),dtype=tf.int32))

        self.record['not_eq_count'] = tf.reduce_sum(tf.cast(tf.logical_or(tf.equal(labels+2, 1.),tf.equal(labels+2, 3.)),dtype=tf.int32))

    def evaluate(self):
        record = self.record
        max_min = 0
        max_min_k = 1
        for tau_idx in range(record['n_threshold']):
            record['WKDR'][tau_idx][0] = record['threshold'][tau_idx]
            record['WKDR'][tau_idx][1] = float(
                record['eq_correct_count'][tau_idx] + record['not_eq_correct_count'][tau_idx]) / float(
                record['eq_count'] + record['not_eq_count'])
            record['WKDR'][tau_idx][2] = float(record['eq_correct_count'][tau_idx]) / float(record['eq_count'])
            record['WKDR'][tau_idx][3] = float(record['not_eq_correct_count'][tau_idx]) / float(record['not_eq_count'])

            if min(record['WKDR'][tau_idx][2], record['WKDR'][tau_idx][3]) > max_min:
                max_min = min(record['WKDR'][tau_idx][2], record['WKDR'][tau_idx][3])
                max_min_k = tau_idx

        return 1 - max_min


    def sym_pairs(self, anchor_reshape, unique_array, depth, depth_label):
        sym_first_pairs = tf.gather(anchor_reshape, unique_array[:, 0])
        sym_second_pairs = tf.stack([sym_first_pairs[:, 0], depth.shape[1] - sym_first_pairs[:, 1] - 1], axis=1)
        sym_pair_z1 = tf.gather_nd(depth, sym_first_pairs)
        sym_pair_z2 = tf.gather_nd(depth, sym_second_pairs)
        sym_pair_label1 = tf.gather_nd(depth_label, sym_first_pairs)
        sym_pair_label2 = tf.gather_nd(depth_label, sym_second_pairs)
        range_relation_high = tf.cast(tf.greater_equal(sym_pair_label1 - sym_pair_label2, 15),
                                      dtype=tf.float32)
        range_relation_low = tf.cast(tf.less_equal(sym_pair_label1 - sym_pair_label2, -1 * 15),
                                     dtype=tf.float32)
        range_label = range_relation_high - range_relation_low
        output = tf.stack([sym_pair_z1, sym_pair_z2, range_label], axis=-1)

        return output

    def calc_loss(self, batch_output,label):
        total_loss=0
        for batch_idx in range(batch_output.shape[0]):
            batch_shape = batch_output.shape
            coordi_range_y = tf.cast(tf.range(0, batch_shape[1], 1), dtype=tf.float32)
            coordi_range_x = tf.cast(tf.range(0, batch_shape[2], 1), dtype=tf.float32)

            y1 = tf.reshape(coordi_range_y, [batch_shape[1], 1])
            x1 = tf.reshape(coordi_range_x, [1, batch_shape[2]])

            y1 = tf.tile(y1, [1, batch_shape[1]])
            x1 = tf.tile(x1, [batch_shape[2], 1])

            y1 = tf.reshape(y1, [-1, batch_shape[1]])
            x1 = tf.reshape(x1, [-1, batch_shape[2]])

            anchor_y1x1 = tf.stack([y1, x1], axis=2)
            anchor_reshape = tf.cast(tf.reshape(anchor_y1x1,[-1,2]),dtype=tf.int64)

            # line_input = np.uint8(np.array(label[batch_idx]))
            # eroded = cv2.erode(line_input, (5, 5),iterations=2)
            # dilated = cv2.dilate(line_input, (5, 5),iterations=2)
            # line = np.abs(dilated - eroded)
            # zeros_coordi = np.array(np.where(line < THRESHOLD))
            #
            # trans_coordi = np.transpose(zeros_coordi, [1, 0])
            # y1_nonedge = tf.gather_nd(y1, trans_coordi)
            # x1_nonedge = tf.gather_nd(x1, trans_coordi)
            #
            # anchor_reshape = tf.cast(tf.stack([y1_nonedge, x1_nonedge], axis=1), dtype=tf.int32)

            depth_logit = tf.squeeze(batch_output[batch_idx])
            pairs = np.random.choice(anchor_reshape.shape[0], np.minimum(3000 * 2,anchor_reshape.shape[0]//2*2))
            pairs_reshape = np.reshape(pairs, [-1, 2])
            pairs_tuple = [tuple(row) for row in pairs_reshape]

            #remove duplication ex) (1,2) , (2,1) are same
            unique_list = np.unique(pairs_tuple, axis=0)
            unique_array = np.array(unique_list)

            resize_label = tf.image.resize(tf.expand_dims(label[batch_idx],axis=-1), batch_shape[1:3])
            resize_label = tf.squeeze(resize_label)

            sym_or_csct = tf.random.uniform(shape=())

            if self.mode == 'train':
                rand_pairs = tf.cond(tf.greater_equal(sym_or_csct,0.5),
                                     lambda:self.csrt_pairs(anchor_reshape, unique_array, depth_logit, resize_label),
                                     lambda:self.sym_pairs(anchor_reshape, unique_array, depth_logit, resize_label))

                total_loss += tf.reduce_sum(self.relative_loss(rand_pairs)) / rand_pairs.shape[0]
            else:
                self.csrt_pairs_val(anchor_reshape, unique_array, depth_logit, resize_label)

        if self.mode == 'train':
            return total_loss/batch_output.shape[0]
        else:
            return self.evaluate()










#
#
#
#
# import tensorflow as tf
#
# from configuration import reg_loss_weight, NUM_CLASSES, alpha, gamma
# # from utils.focal_loss import sigmoid_focal_loss
#
#
# class SmoothL1Loss:
#     def __init__(self):
#         pass
#
#     def __call__(self, y_true, y_pred, *args, **kwargs):
#         absolute_value = tf.math.abs(y_true - y_pred)
#         mask_boolean = tf.math.greater_equal(x=absolute_value, y=1.0)
#         mask_float32 = tf.cast(x=mask_boolean, dtype=tf.float32)
#         smooth_l1_loss = (1.0 - mask_float32) * 0.5 * tf.math.square(absolute_value) + mask_float32 * (absolute_value - 0.5)
#         return smooth_l1_loss
#
#
# class Loss:
#     def __init__(self):
#         self.smooth_l1_loss = SmoothL1Loss()
#         self.reg_loss_weight = reg_loss_weight
#         self.cls_loss_weight = 1 - reg_loss_weight
#         self.num_classes = NUM_CLASSES
#
#     @staticmethod
#     def __cover_background_boxes(true_boxes):
#         symbol = true_boxes[..., -1]
#         mask_symbol = tf.where(symbol < 0.5, 0.0, 1.0)
#         mask_symbol = tf.expand_dims(input=mask_symbol, axis=-1)
#         cover_boxes_tensor = tf.tile(input=mask_symbol, multiples=tf.constant([1, 1, 4], dtype=tf.dtypes.int32))
#         return cover_boxes_tensor
#
#     def relative_loss(za, zb, relation):
#         """
#         Returns:
#             1-D Tensor - loss
#         """
#         mask = tf.abs(relation)
#         return mask * tf.log(1 + tf.exp(-relation * (za - zb))) + (1 - mask) * (za - zb) * (za - zb)
#
#     def __call__(self, y_pred, y_true, *args, **kwargs):
#         logits_by_num_classes = tf.reshape(y_pred, [-1, NUM_CLASSES])
#         labels_flat = tf.reshape(y_true, [-1, ])
#         valid_indices = tf.cast((labels_flat <= NUM_CLASSES - 1),dtype=tf.int32)
#
#         condition = tf.equal(valid_indices, 1)
#         indices_ = tf.where(condition)
#
#         valid_logits = tf.gather(logits_by_num_classes, indices_)
#
#         valid_labels = tf.gather(labels_flat, indices_)
#
#         valid_logits = tf.squeeze(valid_logits)
#         valid_labels = tf.squeeze(valid_labels)
#
#         entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=valid_logits, labels=valid_labels)
#
#         return entropy_loss
#
