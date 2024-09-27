import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers


def npy_padding(x, padding=(1, 1, 1)):
    return np.pad(x, ((padding[0], padding[0]),
                      (padding[1], padding[1]),
                      (padding[2], padding[2])),
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

    def call(self, input_tensor):
        padding_width, padding_height, padding_depth = self.padding
        padding_tensor = [
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [padding_depth, padding_depth],
            [0, 0],
        ]
        return tf.pad(input_tensor, padding_tensor, mode="REFLECT")


class ReflectionPadding2D(layers.Layer):
    """Implements Reflection Padding as a layer.

    Args:
        padding(tuple): Amount of padding for the
        spatial dimensions.

    Returns:
        A padded tensor with the same type as the input tensor.
    """

    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def call(self, input_tensor):
        padding_width, padding_height = self.padding
        padding_tensor = [
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],
        ]
        return tf.pad(input_tensor, padding_tensor, mode="REFLECT")


def residual_block(
        x,
        activation,
        kernel_initializer=None,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="valid",
        gamma_initializer=None,
        use_bias=False,
        dim=3  # Use 3 for 3D, 2 for 2D
):
    """
    Defines a residual block for use in both 2D and 3D convolutional neural networks.

    Args:
        x (tf.Tensor): The input tensor.
        activation (Callable): The activation function to be used.
        kernel_initializer (Optional[Callable], optional): The initializer for the kernel. Defaults to None.
        kernel_size (Tuple[int, int], optional): The kernel size. Defaults to (3, 3).
        strides (Tuple[int, int], optional): The stride size. Defaults to (1, 1).
        padding (str, optional): The padding type. Defaults to "valid".
        gamma_initializer (Optional[Callable], optional): The initializer for the gamma value. Defaults to None.
        use_bias (bool, optional): Whether to use a bias. Defaults to False.
        dim (int, optional): Whether the input is 2D or 3D. Use `dim=3` for 3D and `dim=2` for 2D.

    Returns:
        tf.Tensor: The output tensor.
    """
    dim_filters = x.shape[-1]
    input_tensor = x

    # Select padding layer based on dimensionality
    PaddingLayer = ReflectionPadding3D if dim == 3 else ReflectionPadding2D
    ConvLayer = layers.Conv3D if dim == 3 else layers.Conv2D

    x = PaddingLayer()(input_tensor)

    x = ConvLayer(
        dim_filters,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)

    x = tfa.layers.InstanceNormalization()(x)
    x = activation(x)

    x = PaddingLayer()(x)

    x = ConvLayer(
        dim_filters,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)

    x = tfa.layers.InstanceNormalization()(x)
    x = layers.add([input_tensor, x])

    return x


def downsample(
        x,
        filters,
        activation,
        kernel_initializer='he_normal',
        kernel_size=(3, 3, 3),
        strides=(2, 2, 2),
        padding="valid",
        gamma_initializer=None,
        use_bias=False,
        use_dropout=True,
        dropout_rate=0.2,
        use_spec_norm=False,
        padding_size=(1, 1, 1),
        use_layer_noise=False,
        noise_std=0.1,
        dim=3  # Use 3 for 3D, 2 for 2D
):
    """
    Downsamples an input tensor using either 2D or 3D convolutional layers.

    Args:
        x (Tensor): Input tensor.
        filters (int): Number of output filters in the convolutional layer.
        activation (callable): Activation function to use after convolution.
        kernel_initializer (str, optional): Kernel initializer. Defaults to None.
        kernel_size (tuple of ints, optional): Kernel size for the convolutional layer. Defaults to (3, 3).
        strides (tuple of ints, optional): Strides for the convolutional layer. Defaults to (2, 2).
        padding (str, optional): Padding mode. Defaults to "valid".
        gamma_initializer (str, optional): Gamma initializer for InstanceNormalization. Defaults to None.
        use_bias (bool, optional): Whether to use bias in the convolutional layer. Defaults to False.
        use_dropout (bool, optional): Whether to use dropout after activation. Defaults to True.
        padding_size (tuple of ints, optional): Padding size for ReflectionPadding. Defaults to (1, 1).
        use_layer_noise (bool, optional): Whether to add Gaussian noise after ReflectionPadding. Defaults to False.
        noise_std (float, optional): Standard deviation of Gaussian noise. Defaults to 0.1.
        dim (int, optional): Whether the input is 2D or 3D. Use `dim=3` for 3D and `dim=2` for 2D.

    Returns:
        Tensor: The downsampled tensor.
    """

    PaddingLayer = ReflectionPadding3D if dim == 3 else ReflectionPadding2D
    ConvLayer = layers.Conv3D if dim == 3 else layers.Conv2D
    DropoutLayer = layers.SpatialDropout3D if dim == 3 else layers.SpatialDropout2D

    if padding == 'valid':
        x = PaddingLayer(padding_size)(x)

    if use_layer_noise:
        x = layers.GaussianNoise(noise_std)(x)

    x = ConvLayer(
        filters,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias
    )(x)

    x = tfa.layers.InstanceNormalization()(x)

    if activation:
        x = activation(x)
        if use_dropout:
            x = DropoutLayer(dropout_rate)(x)

    return x


def deconv(
        x,
        filters,
        activation,
        kernel_size=(4, 4),
        strides=(2, 2),
        padding="same",
        kernel_initializer='he_normal',
        gamma_initializer=None,
        use_bias=False,
        dim=3  # Use 3 for 3D, 2 for 2D
):
    """
    Performs 2D or 3D deconvolution on the input tensor using transpose convolutional layers.

    Args:
        x (tf.Tensor): Input tensor.
        filters (int): Number of output filters in the convolutional layer.
        activation (Callable, optional): Activation function to use. If `None`, no activation is applied.
        kernel_size (tuple, optional): Size of the convolutional kernel.
        strides (tuple, optional): The strides of the deconvolution.
        padding (str, optional): Padding type. Defaults to 'same'.
        dim (int, optional): Whether the input is 2D or 3D. Use `dim=3` for 3D and `dim=2` for 2D.

    Returns:
        tf.Tensor: Output tensor.
    """

    ConvTransposeLayer = layers.Conv3DTranspose if dim == 3 else layers.Conv2DTranspose

    x = ConvTransposeLayer(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=kernel_initializer,
        use_bias=use_bias,
    )(x)

    x = tfa.layers.InstanceNormalization()(x)

    if activation:
        x = activation(x)

    return x


def upsample(
        x,
        filters,
        activation,
        kernel_size=(4, 4),
        strides=(1, 1),
        padding="same",
        kernel_initializer='he_normal',
        gamma_initializer=None,
        use_bias=False,
        dim=3  # Use 3 for 3D, 2 for 2D
):
    """
    Upsamples the input tensor using either 2D or 3D transpose convolution.

    Args:
        x (tf.Tensor): The input tensor.
        filters (int): The dimensionality of the output space.
        activation (Optional[Callable]): The activation function to use. Defaults to None.
        kernel_size (Tuple[int, int]): The size of the transposed convolution window.
        strides (Tuple[int, int]): The strides of the transposed convolution.
        padding (str): The type of padding to use. Defaults to 'same'.
        dim (int, optional): Whether the input is 2D or 3D. Use `dim=3` for 3D and `dim=2` for 2D.

    Returns:
        tf.Tensor: The upsampled tensor.
    """

    UpSamplingLayer = layers.UpSampling3D if dim == 3 else layers.UpSampling2D
    ConvLayer = layers.Conv3D if dim == 3 else layers.Conv2D

    x = UpSamplingLayer(size=2)(x)

    x = ConvLayer(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=kernel_initializer,
        use_bias=use_bias,
    )(x)

    x = tfa.layers.InstanceNormalization()(x)

    if activation:
        x = activation(x)

    return x


class StandardisationLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(StandardisationLayer, self).__init__(**kwargs)
    def call(self, inputs):
        # Normalize the output tensor to have mean 0 and std 1
        return ((inputs - tf.reduce_mean(inputs, axis=(1, 2, 3, 4), keepdims=True))
                / (tf.math.reduce_std(inputs, axis=(1, 2, 3, 4), keepdims=True) + tf.keras.backend.epsilon()))
