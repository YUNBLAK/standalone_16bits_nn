import os
import random
import tensorflow as tf
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.compat.v2 as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.python.client import device_lib
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization
from tensorflow.keras import Model
# import MODELS.BATCHNORM.BatchNormalizationLayer as BN

def AlexNet(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(64, (3,3), strides=(2,2), activation='relu', padding='same', input_shape=(input_shape)))
    #model.add(BN.BatchNormalizationLayerDense(x, x.dtype))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    
    model.add(Conv2D(96, (3,3), activation='relu', padding='same'))
    #model.add(BN.BatchNormalizationLayerDense(x, x.dtype))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))

    model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
    #model.add(BN.BatchNormalizationLayerDense(x, x.dtype))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model
