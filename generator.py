import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils import plot_model

import tensorflow_addons as tfa

from building_blocks import downsample, upsample, residual_block, ReflectionPadding3D

def get_resnet_generator(
    input_img_size=(64,64,512,1),
    filters=32,
    num_downsampling_blocks=2,
    num_residual_blocks=9,
    num_upsample_blocks=2,
    gamma_initializer=None,
    kernel_initializer=None,
    name=None,
):
    img_input = layers.Input(shape=input_img_size, name=name + "_img_input")
    # x = ReflectionPadding3D(padding=(3, 3, 3))(img_input)
    x = tf.keras.layers.ZeroPadding3D(padding=1)(img_input)
    #  CHANGE!
    for i in range(0,1):
        x = layers.Conv3D(filters, (7, 7, 7), kernel_initializer=kernel_initializer,
                          use_bias=False)(x)
        x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
        x = layers.Activation("relu")(x)

    # Downsampling
    for _ in range(num_downsampling_blocks):
        filters *= 2
        x = downsample(x, filters=filters, activation=layers.Activation("relu"),
                       kernel_initializer=kernel_initializer,
                       gamma_initializer=gamma_initializer)

    # Residual blocks
    for _ in range(num_residual_blocks):
        x = residual_block(x, activation=layers.Activation("relu"),
                           kernel_initializer=kernel_initializer,
                           gamma_initializer=gamma_initializer)

    # Upsampling
    for _ in range(num_upsample_blocks):
        filters //= 2
        x = upsample(x, filters, activation=layers.Activation("relu"),
                     kernel_initializer=kernel_initializer,
                     gamma_initializer=gamma_initializer)

    # Final block
    # x = ReflectionPadding3D(padding=(3, 3, 3))(x)
    x = tf.keras.layers.ZeroPadding3D(padding=2)(x)
    x = layers.Conv3D(1, (7, 7, 7), padding="same")(x)
    x = layers.Activation("sigmoid")(x)

    model = keras.models.Model(img_input, x, name=name)
    # plot_model(model, to_file='resnet_generator.png', show_shapes=True)
    model.summary()
    return model

