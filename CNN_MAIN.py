import os
import random
import warnings
import tensorflow as tf
import time
import pathlib
import argparse
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.compat.v2 as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.python.client import device_lib
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow.keras import layers
import MODELS.AlexNet as AlexNet
import MODELS.VGG16 as VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging
import MODELS.NEW_RESNET as NR
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.densenet import DenseNet169

warnings.filterwarnings(action='ignore')
logging.getLogger("tensorflow").setLevel(logging.DEBUG)

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

tf.enable_v2_behavior()
arr_acc = []
arr_acc_t1 = []
arr_acc_t5 = []
arr_tim = []

lst_acc = []
lst_time = []
lst_quant_16 = []

glo_batch_size = None
glo_train_images = None
glo_train_images_32 = None
temp_test_img = None
temp_test_lab = None        
                
# AlexNet
def AlexNet_On(input_shape, num_classes):
    return AlexNet.AlexNet(input_shape, num_classes)

# VGG16
def VGG16_On(input_shape, num_classes):
    return VGG16.VGG16(input_shape, num_classes)

# Random Seed Fix
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = '1'
    tf.random.set_seed(seed)

def mobilenetv2_block(x, filters, strides):
    # Depthwise Convolution
    x = DepthwiseConv2D(3, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Pointwise Convolution
    x = Conv2D(filters, 1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    return x

def MobileNetV2_CIFAR10(input_shape=(32, 32, 3), num_classes=10):
    input = Input(shape=input_shape)

    # Initial Conv Layer
    x = Conv2D(32, 3, padding='same')(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # MobileNetV2 Blocks
    x = mobilenetv2_block(x, 64, 1)
    x = mobilenetv2_block(x, 128, 2)
    x = mobilenetv2_block(x, 128, 1)
    x = mobilenetv2_block(x, 256, 2)
    x = mobilenetv2_block(x, 256, 1)
    x = mobilenetv2_block(x, 512, 2)
    for _ in range(5):
        x = mobilenetv2_block(x, 512, 1)
    x = mobilenetv2_block(x, 1024, 2)
    x = mobilenetv2_block(x, 1024, 1)
    
    x = GlobalAveragePooling2D()(x)
    output = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=input, outputs=output)
    return model
    
def NeuralNet(batch_size, epochs, typeFp, arr, modelname, lr, lamb, l2, rseed, resNum):
    tf.keras.backend.clear_session()
    print("Eager mode", tf.executing_eagerly())
    
    
    if typeFp == "mixed_float16" or typeFp == "bfloat16":
        tf.keras.mixed_precision.set_global_policy(typeFp)
    else:
        tf.keras.backend.set_floatx(typeFp)
    
    seed_everything(rseed)
    
    train_images, train_labels, test_images, test_labels = arr
    global glo_train_images 
    global glo_train_images_32
    global glo_batch_size
    glo_train_images = train_images
    glo_batch_size = batch_size
    glo_train_images_32 = glo_train_images.astype("float32")
    
    
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_dataset = train_dataset.shuffle(len(train_images)).batch(batch_size)
    test_set = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    test_batches = test_set.batch(batch_size)
    mem = []
    
    # Construct model
    if modelname == 'alex':
        model = AlexNet_On((32, 32, 3), 10)
    elif modelname == 'vgg':
        model = VGG16_On((32, 32, 3), 10)
    elif modelname == 'res':
        model = NR.ResNetModel(num_layers = resNum, weight_decay=lamb, l2_switch = l2)
        model.build_graph(train_images.shape)
    elif modelname == 'mob':
        model = MobileNetV2_CIFAR10((32, 32, 3), 10)
    elif modelname == "dense":
        if resNum == 121:
            print("DENSE 121")
            base_model = DenseNet121(include_top=False, weights=None, input_tensor=Input(shape=(32, 32, 3)))
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            predictions = Dense(10, activation='softmax')(x)
            model = Model(inputs=base_model.input, outputs=predictions)
        else:
            print("DENSE 169")
            base_model = DenseNet169(include_top=False, weights=None, input_tensor=Input(shape=(32, 32, 3)))
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            predictions = Dense(10, activation='softmax')(x)
            model = Model(inputs=base_model.input, outputs=predictions)

    optimizer = tf.keras.optimizers.SGD(learning_rate = lr, momentum = 0.9)
    start = time.time()
    
    model.compile(
        optimizer=optimizer, 
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(train_images, 
                        train_labels, 
                        batch_size = 256,
                        epochs = 100, 
                        validation_data=(test_images, test_labels)
                       )
    
    tacc = history.history['accuracy']
    acc = history.history['val_accuracy']
    loss = history.history['loss']
    timeX = time.time() - start
    
    lossdict = {
        "TACC" : tacc,
        "ACC" : acc,
        "LOSS" : loss,
        "TIME" : timeX / 100
    }
    
    lss = pd.DataFrame(lossdict)
    lss_name = "RESULTS/"+ typeFp + "_" + modelname + "_" + str(batch_size) + "_rand_" + str(rseed)+ "_resNum_" + str(resNum) +".csv"
    lss.to_csv(lss_name, index = False)
    tf.keras.backend.clear_session()

def normalization(train_images, test_images):
    mean = np.mean(train_images, axis=(0, 1, 2, 3))
    std = np.std(train_images, axis=(0, 1, 2, 3))
    train_images = (train_images - mean) / (std + 1e-7)
    test_images = (test_images - mean) / (std + 1e-7)
    return train_images, test_images    

def TestOn(mname, xtype, lr, batch_size, lamb, l2, rseed, resNum):
    typeFp = xtype
    modelname = mname
    batchsize = batch_size
    epochs = 1
    
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    (train_images, test_images) = normalization(train_images, test_images)
    
    if typeFp == "mixed_float16":
        train_images = train_images.astype("float32")
        train_labels = train_labels.astype("float32")
        test_images = test_images.astype("float32")
        test_labels = test_labels.astype("float32")
    else:
        train_images = train_images.astype(typeFp)
        train_labels = train_labels.astype(typeFp)
        test_images = test_images.astype(typeFp)
        test_labels = test_labels.astype(typeFp)
    
    # train_images, test_images = train_images / 255, test_images / 255
    datasets = [train_images, train_labels, test_images, test_labels]
    
    print()
    print("--------------------------------------------------------------")
    print("[] START", batchsize, epochs, typeFp, modelname, lr, lamb, l2)
    print("--------------------------------------------------------------")
    print()
    NeuralNet(batchsize, epochs, typeFp, datasets, modelname, lr, lamb, l2, rseed, resNum)
    tf.keras.backend.clear_session()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="LabProgram",
        description="This is the artifact"
    )

    parser.add_argument("modelType")
    parser.add_argument("precision")
    parser.add_argument("learningRate", type=float)
    parser.add_argument("batchSize", type=int)
    parser.add_argument("-rs", "--randomSeeds", type=int)
    parser.add_argument("-rn", "--resNum", type=int)

    args = parser.parse_args()
    TestOn(args.modelType, args.precision, args.learningRate, args.batchSize, 1e-4, True, (args.randomSeeds if None != args.randomSeeds else random.random(0, 100)), (args.resNum if args.modelType == 'res' else 0))
    

    