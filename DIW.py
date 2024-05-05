import tensorflow as tf

_WEIGHT_DECAY = 4e-5

# FLAGS = tf.flags.FLAGS

import tensorflow as tf
import numpy as np
from configuration import *

class inverted_residual(tf.keras.layers.Layer):
    def __init__(self, input_shape, up_sample_rate, atrous_rate, channels, subsample, index_):
        super(inverted_residual, self).__init__()
        # self.i += 1
        stride = 2 if subsample else 1
        self.up_sample_rate = up_sample_rate
        self.atrous_rate = atrous_rate
        self.channels = channels
        self.subsample = subsample
        self.stride = stride
        output_h = input_shape[1] // stride + input_shape[1] % stride
        output_w = input_shape[2] // stride + input_shape[2] % stride
        self.output_dims = [input_shape[0],output_h,output_w, channels]
        self.relu6 = tf.nn.relu6
        self.index_=index_

        if up_sample_rate > 1:
            self.expand = tf.keras.layers.Conv2D(filters=up_sample_rate * input_shape[-1],
                                               kernel_size=1, use_bias=False,
                                               padding='same', dilation_rate=(1, 1), name='expand')
            self.expand_batchnorm = tf.keras.layers.BatchNormalization(momentum=0.9997, epsilon=1e-3, center=True,
                                                                     scale=True, name='BatchNorm')

        self.depthwise = tf.keras.layers.DepthwiseConv2D(kernel_size=3,use_bias=False, strides=(stride, stride),
                                                     padding='same', dilation_rate=(atrous_rate, atrous_rate), name='depthwise')
        self.depthwise_batchnorm = tf.keras.layers.BatchNormalization(momentum=0.9997, epsilon=1e-3, center=True,
                                                                  scale=True, name='BatchNorm')
        self.pointwise = tf.keras.layers.Conv2D(filters=channels, kernel_size=1, use_bias=False,
                                            padding='same', dilation_rate=(1, 1), name='project')
        self.pointwise_batchnorm = tf.keras.layers.BatchNormalization(momentum=0.9997, epsilon=1e-3, center=True,
                                                                  scale=True, name='BatchNorm')
    # def call(self, inputs, training=None, *args, **kwargs):
    def __call__(self, input,training=None):
        if self.index_==0:
            name = 'expanded_conv'
        else:
            name = 'expanded_conv_{}'.format(self.index_)
        with tf.name_scope(name):
            if self.up_sample_rate > 1:
                expand_ = self.expand(input,training=training)
                with tf.name_scope('expand'):
                    expand_ = self.expand_batchnorm(expand_,training=training)
                    expand_ = self.relu6(expand_)
            else :
                expand_ = input
            depthwise_ = self.depthwise(expand_,training=training)
            with tf.name_scope('depthwise'):
                depthwise_ = self.depthwise_batchnorm(depthwise_,training=training)
                depthwise_ = self.relu6(depthwise_)
            project_ = self.pointwise(depthwise_,training=training)
            with tf.name_scope('project'):
                project_ =  self.pointwise_batchnorm(project_,training=training)
            if input.shape[-1] == self.channels:
                project_ =tf.keras.layers.add([input,project_])
        return expand_, depthwise_, project_

class CBR(tf.keras.layers.Layer):
    def __init__(self,input_dims, filters, kernel_size, use_bias, padding, dilation_rate, layer_name):
        super(CBR, self).__init__()
        self.input_dims = input_dims
        self.output_dims = [input_dims[0], input_dims[1], input_dims[2], filters]
        self.layer_name = layer_name
        self.relu = tf.nn.relu
        self.conv2d = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, use_bias=use_bias,
                                            padding=padding, dilation_rate=dilation_rate, name=layer_name)
        self.batch_norm = tf.keras.layers.BatchNormalization(momentum=0.9997, epsilon=1e-3, center=True,
                                                                  scale=True, name='BatchNorm')
    def __call__(self, input, relu=True ,training=None, *args, **kwargs):
        output = self.conv2d(input)
        with tf.name_scope(self.layer_name):
            output = self.batch_norm(output)
            if relu:
                output = self.relu(output)
        return output

class DecodingBlock(tf.keras.layers.Layer):
    def __init__(self,input_shape, atrous_rate, channels, upsample, index_, use_batchnorm=True):
        super(DecodingBlock,self).__init__()
        # self.i += 1
        self.atrous_rate = atrous_rate
        self.channels = channels
        self.mid_channels = input_shape[3]*3
        self.upsample = upsample[1:3] if upsample!=None else upsample
        # if upsample:
        #     self.resize_shape = (input_shape[1]*2, input_shape[2]*2)
        # else:
        #     self.resize_shape = input_shape[1:3]
        if upsample == None:
            self.output_dims = input_shape
        else:
            self.output_dims = upsample
        self.relu6 = tf.nn.relu6
        self.index_ = index_
        self.use_batchnorm = use_batchnorm


        self.expand = tf.keras.layers.Conv2D(filters=self.mid_channels,
                                             kernel_size=1, use_bias=False,
                                             padding='same', dilation_rate=(1, 1), name='expand')
        self.expand_batchnorm = tf.keras.layers.BatchNormalization(momentum=0.9997, epsilon=1e-3, center=True,
                                                                   scale=True, name='BatchNorm')

        self.depthwise = tf.keras.layers.DepthwiseConv2D(kernel_size=5, use_bias=False, strides=(1, 1),
                                                         padding='same', dilation_rate=(atrous_rate, atrous_rate),
                                                         name='depthwise')
        self.depthwise_batchnorm = tf.keras.layers.BatchNormalization(momentum=0.9997, epsilon=1e-3, center=True,
                                                                      scale=True, name='BatchNorm')
        self.pointwise = tf.keras.layers.Conv2D(filters=channels, kernel_size=1, use_bias=False,
                                                padding='same', dilation_rate=(1, 1), name='project')
        self.pointwise_batchnorm = tf.keras.layers.BatchNormalization(momentum=0.9997, epsilon=1e-3, center=True,
                                                                      scale=True, name='BatchNorm')
    def __call__(self,input, relu=True ,training=None, *args, **kwargs):

        if self.index_==0:
            name = 'decoder_layer'
        else:
            name = 'decoder_layer_{}'.format(str(self.index_))
        with tf.name_scope(name):
            expand_ = self.expand(input,training=training)
            with tf.name_scope('expand'):
                expand_ = self.expand_batchnorm(expand_,training=training)
                expand_ = self.relu6(expand_)
                if self.upsample != None:
                    expand_ = tf.image.resize(expand_,self.upsample,method='nearest')
            depthwise_ = self.depthwise(expand_,training=training)
            with tf.name_scope('depthwise'):
                depthwise_ = self.depthwise_batchnorm(depthwise_,training=training)
                depthwise_ = self.relu6(depthwise_)
            project_ = self.pointwise(depthwise_,training=training)
            with tf.name_scope('project'):
                if self.use_batchnorm:
                    project_ =  self.pointwise_batchnorm(project_,training=training)
            if input.shape[-1] == self.channels:
                project_ =tf.keras.layers.add([input,project_])
        return expand_, depthwise_, project_

class FeatureExtractor(tf.keras.Model):
    def __init__(self, input_dims):
        super(FeatureExtractor, self).__init__()
        self._name=''
        self.input_dims=input_dims
        output_h = input_dims[1] // 2 + input_dims[1] % 2
        output_w = input_dims[2] // 2 + input_dims[2] % 2
        output_dims = [input_dims[0],output_h,output_w, 32]
        self.conv_ = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), use_bias=False,
                                            padding='same', dilation_rate=(1, 1), name='Conv')
        self.conv_batchnorm = tf.keras.layers.BatchNormalization(momentum=0.9997, epsilon=1e-3, center=True,
                                                                  scale=True, name='BatchNorm')#112
        self.relu6 = tf.nn.relu6
        self.inverted_residual0 = inverted_residual(output_dims, 1, 1, 16, 0, 0) #layer1
        self.inverted_residual1 = inverted_residual(self.inverted_residual0.output_dims, 6, 1, 24, 1 ,1)
        self.inverted_residual2 = inverted_residual(self.inverted_residual1.output_dims, 6, 1, 24, 0, 2) #layer2
        self.inverted_residual3 = inverted_residual(self.inverted_residual2.output_dims, 6, 1, 32, 1, 3)
        self.inverted_residual4 = inverted_residual(self.inverted_residual3.output_dims, 6, 1, 32, 0, 4)
        self.inverted_residual5 = inverted_residual(self.inverted_residual4.output_dims, 6, 1, 32, 0, 5) #layer3
        self.inverted_residual6 = inverted_residual(self.inverted_residual5.output_dims, 6, 1, 64, 1, 6)
        self.inverted_residual7 = inverted_residual(self.inverted_residual6.output_dims, 6, 1, 64, 0, 7)
        self.inverted_residual8 = inverted_residual(self.inverted_residual7.output_dims, 6, 1, 64, 0, 8)
        self.inverted_residual9 = inverted_residual(self.inverted_residual8.output_dims, 6, 1, 64, 0, 9) #layer4
        self.inverted_residual10 = inverted_residual(self.inverted_residual9.output_dims, 6, 1, 96, 0, 10)
        self.inverted_residual11 = inverted_residual(self.inverted_residual10.output_dims, 6, 1, 96, 0, 11)
        self.inverted_residual12 = inverted_residual(self.inverted_residual11.output_dims, 6, 1, 96, 0, 12)
        self.inverted_residual13 = inverted_residual(self.inverted_residual12.output_dims, 6, 1, 160, 1, 13)
        self.inverted_residual14 = inverted_residual(self.inverted_residual13.output_dims, 6, 1, 160, 0, 14)
        self.inverted_residual15 = inverted_residual(self.inverted_residual14.output_dims, 6, 1, 160, 0, 15) #layer5
        self.inverted_residual16 = inverted_residual(self.inverted_residual15.output_dims, 6, 1, 320, 0, 16) #final
        self.final_dims = self.inverted_residual16.output_dims
    def call(self, inputs, training=None, mask=None):
        with tf.name_scope('MobilenetV2'):
            output = self.conv_(inputs,training=training)
            with tf.name_scope('Conv'):
                output = self.conv_batchnorm(output,training=training)
                output = self.relu6(output)
            _, _, layer1 = self.inverted_residual0(output,training=training)
            _, _, output = self.inverted_residual1(layer1,training=training)
            _, _, layer2 = self.inverted_residual2(output,training=training)
            _, _, output = self.inverted_residual3(layer2,training=training)
            _, _, output = self.inverted_residual4(output,training=training)
            _, _, layer3 = self.inverted_residual5(output,training=training)
            _, _, output = self.inverted_residual6(layer3,training=training)
            _, _, output = self.inverted_residual7(output,training=training)
            _, _, output = self.inverted_residual8(output,training=training)
            _, _, layer4 = self.inverted_residual9(output,training=training)
            _, _, output = self.inverted_residual10(output,training=training)
            _, _, output = self.inverted_residual11(output,training=training)
            _, _, output = self.inverted_residual12(output,training=training)
            _, _, output = self.inverted_residual13(output,training=training)
            _, _, output = self.inverted_residual14(output,training=training)
            _, _, layer5 = self.inverted_residual15(output,training=training)
            _, _, final = self.inverted_residual16(output,training=training)
        return layer1, layer2, layer3, layer4, layer5, final


class DIW(tf.keras.Model):
    def __init__(self, input_dims):
        super(DIW, self).__init__()
        self._name=''
        self.input_dims=input_dims
        self.backbone = FeatureExtractor(input_dims)
        self.final_conv2d = CBR(input_dims=self.backbone.final_dims ,filters=1024, kernel_size=1, use_bias=False,
                                            padding='same', dilation_rate=(1, 1), layer_name='final_conv')

        #input_shape, atrous_rate, channels, upsample, index_
        #160,64,32,24,16
        self.decoder_layer_1 = DecodingBlock(self.final_conv2d.output_dims, 1, 160, None, 1)
        self.decoder_layer_2 = DecodingBlock(self.decoder_layer_1.output_dims, 1, 64, self.backbone.inverted_residual9.output_dims, 2)
        self.decoder_layer_3 = DecodingBlock(self.decoder_layer_2.output_dims, 1, 32, self.backbone.inverted_residual5.output_dims, 3)
        self.decoder_layer_4 = DecodingBlock(self.decoder_layer_3.output_dims, 1, 24, self.backbone.inverted_residual2.output_dims, 4)
        self.decoder_layer_5 = DecodingBlock(self.decoder_layer_4.output_dims, 1, 16, self.backbone.inverted_residual0.output_dims, 5)

        self.interpred_decoder_1 = DecodingBlock(self.decoder_layer_2.output_dims, 1, 1, None, '2_1', use_batchnorm=False)
        self.interpred_decoder_2 = DecodingBlock(self.decoder_layer_3.output_dims, 1, 1, None, '3_1', use_batchnorm=False)
        self.prediction = tf.keras.layers.Conv2D(filters=1, kernel_size=1, use_bias=False,
                                                padding='same', dilation_rate=(1, 1), name='predictions')

    def call(self, inputs, training=None, mask=None):
        outputs_size = tf.shape(inputs)[1:3]
        layer_1, layer_2, layer_3, layer_4, layer5, final = self.backbone(inputs, training=training)
        end_point_features_final = tf.identity(final)

        output = self.final_conv2d(end_point_features_final)

        _, _, output = self.decoder_layer_1(output)

        output = tf.keras.layers.add([layer5, output])

        _, _, output = self.decoder_layer_2(output)
        output = tf.keras.layers.add([layer_4, output])

        _, _, inter_pred1 = self.interpred_decoder_1(output)

        _, _, output = self.decoder_layer_3(output)
        output = tf.keras.layers.add([layer_3, output])

        _, _, inter_pred2 = self.interpred_decoder_2(output)

        _, _, output = self.decoder_layer_4(output)
        output = tf.keras.layers.add([layer_2, output])

        _, _, output = self.decoder_layer_5(output)
        output = tf.keras.layers.add([layer_1, output])

        output = self.prediction(output)

        resized_output = tf.image.resize(output, outputs_size, name='upsample')

        return inter_pred1, inter_pred2, resized_output



