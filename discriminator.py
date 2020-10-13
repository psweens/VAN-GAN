import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_addons as tfa

from building_blocks import downsample

def get_discriminator(input_img_size=(64,64,512,1),
                      filters=64,
                      kernel_initializer=None,
                      num_downsampling=3,
                      name=None):

    img_input = layers.Input(shape=input_img_size, name=name + "_img_input")
    x = layers.Conv3D(
        filters,
        (4, 4, 4),
        strides=(2, 2, 2),
        padding="same",
        kernel_initializer=kernel_initializer,
    )(img_input)
    x = layers.LeakyReLU(0.2)(x)

    num_filters = filters
    for num_downsample_block in range(3):
        num_filters *= 2
        if num_downsample_block < 2:
            x = downsample(
                x,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4, 4, 4),
                strides=(2, 2, 2),
            )
        else:
            x = downsample(
                x,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4, 4, 4),
                strides=(1, 1, 1),
            )

    x = layers.Conv3D(1, (4, 4, 4), strides=(1, 1, 1), padding="same", kernel_initializer=kernel_initializer)(x)

    model = keras.models.Model(inputs=img_input, outputs=x, name=name)
    # plot_model(model, to_file='discriminator.png', show_shapes=True)
    model.summary()
    return model