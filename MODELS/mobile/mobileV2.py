import os
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Activation, Conv2D, Dense, Dropout, BatchNormalization, ReLU, DepthwiseConv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Add
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v



# define the calculation of each 'inverted Res_Block'
def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id):
    prefix = 'block_{}_'.format(block_id)

    in_channels = inputs.shape[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs

    # Expand
    if block_id:
        x = Conv2D(expansion * in_channels, kernel_size=1, strides=1, padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", name=prefix + 'expand')(x)
        x = BatchNormalization(momentum=0.9, name=prefix + 'expand_BN')(x)
        x = ReLU(6., name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'

    # Depthwise
    x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None, use_bias=False, padding='same', kernel_initializer="he_normal", name=prefix + 'depthwise')(x)
    x = BatchNormalization(momentum=0.9, name=prefix + 'depthwise_BN')(x)
    x = ReLU(6., name=prefix + 'depthwise_relu')(x)

    # Project
    x = Conv2D(pointwise_filters, kernel_size=1, strides=1, padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", name=prefix + 'project')(x)
    x = BatchNormalization(momentum=0.9, name=prefix + 'project_BN')(x)


    if in_channels == pointwise_filters and stride == 1:
        return Add(name=prefix + 'add')([inputs, x])
    return x

#Create Build
def create_model(rows, cols, channels, level_0_blocks, level_1_blocks, level_2_blocks, num_classes, lr_initial):
  # encoder - input
    alpha=1.0
    include_top = True
    model_input = tf.keras.Input(shape=(rows, cols, channels), name='input_image')
    x           = model_input

    first_block_filters = _make_divisible(32 * alpha, 8)

    # model architechture
    x = Conv2D(first_block_filters, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_initializer="he_normal", name='Conv1')(model_input)

    x = _inverted_res_block(x, filters=16,  alpha=alpha, stride=1, expansion=1, block_id=0 )

    x = _inverted_res_block(x, filters=24,  alpha=alpha, stride=1, expansion=6, block_id=1 )
    x = _inverted_res_block(x, filters=24,  alpha=alpha, stride=1, expansion=6, block_id=2 )

    x = _inverted_res_block(x, filters=32,  alpha=alpha, stride=2, expansion=6, block_id=3 )
    x = _inverted_res_block(x, filters=32,  alpha=alpha, stride=1, expansion=6, block_id=4 )
    x = _inverted_res_block(x, filters=32,  alpha=alpha, stride=1, expansion=6, block_id=5 )

    x = _inverted_res_block(x, filters=64,  alpha=alpha, stride=2, expansion=6, block_id=6 )
    x = _inverted_res_block(x, filters=64,  alpha=alpha, stride=1, expansion=6, block_id=7 )
    x = _inverted_res_block(x, filters=64,  alpha=alpha, stride=1, expansion=6, block_id=8 )
    x = _inverted_res_block(x, filters=64,  alpha=alpha, stride=1, expansion=6, block_id=9 )
    x = Dropout(rate=0.25)(x)

    x = _inverted_res_block(x, filters=96,  alpha=alpha, stride=1, expansion=6, block_id=10)
    x = _inverted_res_block(x, filters=96,  alpha=alpha, stride=1, expansion=6, block_id=11)
    x = _inverted_res_block(x, filters=96,  alpha=alpha, stride=1, expansion=6, block_id=12)
    x = Dropout(rate=0.25)(x)

    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=13)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=14)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=15)
    x = Dropout(rate=0.25)(x)

    x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1, expansion=6, block_id=16)
    x = Dropout(rate=0.25)(x)


    # define filter size (last block)
    if alpha > 1.0:
        last_block_filters = _make_divisible(1280 * alpha, 8)
    else:
        last_block_filters = 1280

    x = Conv2D(last_block_filters, kernel_size=1, use_bias=False, kernel_initializer="he_normal", name='Conv_1')(x)
    x = BatchNormalization(momentum=0.9, name='Conv_1_bn')(x)
    x = ReLU(6., name='out_relu')(x)
    
    if include_top:
        x = GlobalAveragePooling2D(name='global_average_pool')(x)
        x = Dense(10, activation='softmax', use_bias=True, name='Logits')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # create model of MobileNetV2 (for CIFAR-10)
    model = Model(inputs=model_input, outputs=x, name='mobilenetv2_cifar10')

    #model.compile(optimizer=tf.keras.optimizers.Adam(lr_initial), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model