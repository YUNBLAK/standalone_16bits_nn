import os
import random
import cv2
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
from keras.layers import Reshape
import MODELS.BATCHNORM.BatchNormalization16 as BN16

def BatchNormalizationLayerConv(inputLayer, dtype = "float32"):
    l1 = inputLayer.shape[1]
    l2 = inputLayer.shape[2]
    l3 = inputLayer.shape[3]   
    newLayer = Flatten()(inputLayer)     
    if dtype == "float16":
        newLayer = BN16.BatchNormalization()(newLayer)
    else:
        newLayer = BatchNormalization()(newLayer)
    
    newLayer = layers.Reshape([l1, l2, l3])(newLayer)
    return newLayer

def BatchNormalizationLayerDense(inputLayer, dtype = "float32"):
    if dtype == "float16":
        newLayer = BN16.BatchNormalization()(inputLayer)
    else:
        newLayer = BatchNormalization()(inputLayer)
    return newLayer