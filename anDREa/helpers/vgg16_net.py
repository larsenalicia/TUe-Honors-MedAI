import os
#import cv2
import glob
import pathlib
import PIL, PIL.Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from PIL import Image

def VGG16model(class_names, img_height, img_width):
    # pass

    num_classes = len(class_names)

    vgg16_model = keras.Sequential()

    vgg16_model.add(layers.Conv2D(input_shape=(img_height, img_width, 1), filters=64,kernel_size=(3, 3),
                                  padding="same", activation="relu"))
    vgg16_model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    vgg16_model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    vgg16_model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    vgg16_model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    vgg16_model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    vgg16_model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    vgg16_model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    vgg16_model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    vgg16_model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    vgg16_model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    vgg16_model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    vgg16_model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    vgg16_model.add(layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))

    vgg16_model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    vgg16_model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    vgg16_model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    vgg16_model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    vgg16_model.add(layers.Flatten())
    vgg16_model.add(layers.Dense(4096, activation="relu"))
    vgg16_model.add(layers.Dense(4096, activation="relu"))
    vgg16_model.add(layers.Dense(num_classes, activation="softmax"))
    vgg16_model.summary()

    opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001)
    vgg16_model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    return vgg16_model
