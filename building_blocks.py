import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_addons as tfa

class ReflectionPadding3D(layers.Layer):
    """Implements Reflection Padding as a layer.

    Args:
        padding(tuple): Amount of padding for the
        spatial dimensions.

    Returns:
        A padded tensor with the same type as the input tensor.
    """

    def __init__(self, padding=(1, 1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding3D, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        padding_width, padding_height, padding_depth = self.padding
        padding_tensor = [
            [0, 0, 0],
            [padding_height, padding_height, padding_depth],
            [padding_width, padding_width, padding_depth],
            [0, 0, 0],
        ]
        return tf.pad(input_tensor, padding_tensor, mode="REFLECT")

def residual_block(
    x,
    activation,
    kernel_initializer=None,
    kernel_size=(3, 3, 3),
    strides=(1, 1, 1),
    padding="valid",
    gamma_initializer=None,
    use_bias=False
):
    dim = x.shape[-1]
    input_tensor = x

    # x = ReflectionPadding3D()(input_tensor)
    x = tf.keras.layers.ZeroPadding3D()(input_tensor)
    x = layers.Conv3D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = activation(x)

    # x = ReflectionPadding3D()(x)
    x = tf.keras.layers.ZeroPadding3D()(x)
    x = layers.Conv3D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = layers.add([input_tensor, x])
    return x


def downsample(
    x,
    filters,
    activation,
    kernel_initializer=None,
    kernel_size=(3, 3, 3),
    strides=(2, 2, 2),
    padding="same",
    gamma_initializer=None,
    use_bias=False
):
    x = layers.Conv3D(
        filters,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x


def upsample(
    x,
    filters,
    activation,
    kernel_size=(4,4,4),
    strides=(2, 2, 2),
    padding="same",
    kernel_initializer=None,
    gamma_initializer=None,
    use_bias=False,
):
    x = layers.Conv3DTranspose(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=kernel_initializer,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x