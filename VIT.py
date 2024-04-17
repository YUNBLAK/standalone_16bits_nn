import os
import time
import pandas as pd
import numpy as np
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

arr1 = []
arr2 = []
arr3 = []

activationl = ""
num_classes = 10
input_shape = (32, 32, 3)
learning_rate = 0.01
weight_decay = 0.0001
batch_size = 256
num_epochs = 100
transformer_layers = 8
image_size = 32  # We'll resize input images to this size
patch_size = 6  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
mlp_head_units = [512, 512]  # Size of the dense layers of the final classifier
dropout_rate = 0.1
ff_dim = 512  # hidden layer size in feed forward network inside transformer

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = '1'
    tf.random.set_seed(seed)

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        
        global activationl
        
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="gelu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6) #CustomLayerNormalization() #layers.LayerNormalization(epsilon=1e-3)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6) #CustomLayerNormalization() #layers.LayerNormalization(epsilon=1e-3)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def create_vit_classifier(data_augmentation):
    inputs = layers.Input(shape=(32, 32, 3))
    # Augment data.
    augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        encoded_patches = TransformerBlock(projection_dim, num_heads, ff_dim, dropout_rate)(encoded_patches)

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    global activationl
    features = layers.Dense(units=mlp_head_units[0], activation="gelu")(representation)
    features = layers.Dense(units=mlp_head_units[1], activation="gelu")(features)
    # Classify outputs.
    logits = layers.Dense(num_classes)(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model

def run_experiment(model, x_train, y_train, x_test, y_test):
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)

    start = time.time()
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy")
        ],
    )
    
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_data=(x_test, y_test)
    )

    _, accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    timeX = time.time() - start
    print("time :", timeX)
    
    arr1.append(accuracy)
    arr3.append(timeX)
    return history

def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tfa.image.rotate(image, angles=np.random.uniform(-0.2, 0.2))

    return image, label

def neuralnet(dfp, act, l, r):
    global activationl
    activationl = act
    
    global transformer_layers
    transformer_layers = l
    
    seed_everything(r)
    fp = dfp

    if fp == "mixed_float16" or "bfloat16":
        tf.keras.mixed_precision.set_global_policy(fp)
    else:
        tf.keras.backend.set_floatx(fp)

    # Load the data
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    if fp == "mixed_float16" or "bfloat16":
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255
    else:
        x_train = x_train.astype(fp) / 255
        x_test = x_test.astype(fp) / 255
        y_train = y_train.astype(fp)
        y_test = y_test.astype(fp)
    
    # Preprocess the data
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.Resizing(image_size, image_size),
            layers.experimental.preprocessing.Normalization(),
        ]
    )

    # Compute the mean and the variance of the training data for normalization.
    vit_classifier = create_vit_classifier(data_augmentation)
    history = run_experiment(vit_classifier, x_train, y_train, x_test, y_test)
    
    acc = history.history['val_accuracy']
    loss = history.history['loss']
    
    dict = {
        "ACC" : acc,
        "LOSS" : loss
    }
    
    ndf = pd.DataFrame(dict)
    name = "RESULTS/" + fp + "_" + str(r) + "_" + str(l) + ".csv"
    ndf.to_csv(name, index = False)
    tf.keras.backend.clear_session()

def tester(ffpt, lr, bs, layerNum, random_seed):
    global learning_rate, batch_size, transformer_layers
    learning_rate = lr
    batch_size = bs
    transformer_layers = layerNum
    
    neuralnet(ffpt, "relu", layerNum, random_seed)