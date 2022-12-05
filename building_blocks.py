import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

import tensorflow_addons as tfa

def npy_padding(x, padding=(1,1,1), padtype='reflect'):
    return np.pad(x, ((padding[0],padding[0]), 
                      (padding[1],padding[1]), 
                      (padding[2],padding[2])),
                  'reflect')
    

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
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [padding_depth, padding_depth],
            [0, 0],
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

    x = ReflectionPadding3D()(input_tensor)

    x = layers.Conv3D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = activation(x)
    x = layers.SpatialDropout3D(0.5)(x)

    x = ReflectionPadding3D()(x)

    x = layers.Conv3D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = layers.add([input_tensor, x])
    return x


def downsample(
    x,
    filters,
    activation,
    kernel_initializer=None,
    kernel_size=(3, 3, 3),
    strides=(2, 2, 2),
    padding="valid",
    gamma_initializer=None,
    use_bias=False,
    use_dropout=True,
    use_SN=False,
    padding_size=(1, 1, 1),
    use_layer_noise=False,
    noise_std=0.1
):
    
    if padding == 'valid':
        x = ReflectionPadding3D(padding_size)(x)
        
    if use_layer_noise:
        x = layers.GaussianNoise(noise_std)(x)

    if use_SN:
        x = tfa.layers.SpectralNormalization(layers.Conv3D(
            filters,
            kernel_size,
            strides=strides,
            kernel_initializer=kernel_initializer,
            padding=padding,
            use_bias=use_bias
        ))(x)
    else:
        x = layers.Conv3D(
            filters,
            kernel_size,
            strides=strides,
            kernel_initializer=kernel_initializer,
            padding=padding,
            use_bias=use_bias
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        # x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
        
    if activation:
        x = activation(x)
        if use_dropout:
            x = layers.SpatialDropout3D(0.2)(x)
    return x


def deconv(
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
    x = tf.keras.layers.BatchNormalization()(x)
    # x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
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
    x = layers.UpSampling3D(
        size=2
        )(x)
    x = layers.Conv3D(
        filters,
        kernel_size,
        strides=(1,1,1),
        padding=padding,
        kernel_initializer=kernel_initializer,
        use_bias=use_bias,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x