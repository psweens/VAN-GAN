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

    def call(self, input_tensor, mask=None):
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
    kernel_size=(3, 3, 3),
    strides=(1, 1, 1),
    padding="valid",
    gamma_initializer=None,
    use_bias=False
):
    """
    Defines a residual block for use in a 3D convolutional neural network.
    
    Args:
        x (tf.Tensor): The input tensor.
        activation (Union[Callable, str]): The activation function to be used.
        kernel_initializer (Optional[Callable], optional): The initializer for the kernel. Defaults to None.
        kernel_size (Tuple[int, int, int], optional): The kernel size. Defaults to (3, 3, 3).
        strides (Tuple[int, int, int], optional): The stride size. Defaults to (1, 1, 1).
        padding (str, optional): The padding type. Defaults to "valid".
        gamma_initializer (Optional[Callable], optional): The initializer for the gamma value. Defaults to None.
        use_bias (bool, optional): Whether to use a bias. Defaults to False.
    
    Returns:
        tf.Tensor: The output tensor.
    """
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
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = activation(x)
    # x = layers.SpatialDropout3D(0.5)(x)

    x = ReflectionPadding3D()(x)

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
    padding="valid",
    gamma_initializer=None,
    use_bias=False,
    use_dropout=True,
    use_SN=False,
    padding_size=(1, 1, 1),
    use_layer_noise=False,
    noise_std=0.1
):
    """
    Downsamples an input tensor using a 3D convolutional layer.
    
    Args:
        x (Tensor): Input tensor.
        filters (int): Number of output filters in the convolutional layer.
        activation (callable): Activation function to use after convolution.
        kernel_initializer (str, optional): Kernel initializer. Defaults to None.
        kernel_size (tuple of ints, optional): Kernel size for the convolutional layer. Defaults to (3, 3, 3).
        strides (tuple of ints, optional): Strides for the convolutional layer. Defaults to (2, 2, 2).
        padding (str, optional): Padding mode. Defaults to "valid".
        gamma_initializer (str, optional): Gamma initializer for InstanceNormalization. Defaults to None.
        use_bias (bool, optional): Whether to use bias in the convolutional layer. Defaults to False.
        use_dropout (bool, optional): Whether to use dropout after activation. Defaults to True.
        use_SN (bool, optional): Whether to use Spectral Normalization. Defaults to False.
        padding_size (tuple of ints, optional): Padding size for ReflectionPadding3D. Defaults to (1, 1, 1).
        use_layer_noise (bool, optional): Whether to add Gaussian noise after ReflectionPadding3D. Defaults to False.
        noise_std (float, optional): Standard deviation of Gaussian noise. Defaults to 0.1.
    
    Returns:
        Tensor: The downsampled tensor.
    """
    
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
        x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
        
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
    """
    3D deconvolution on the input tensor `x` using transpose convolutional layers.
    
    Args:
        x (tf.Tensor): Input tensor of shape [batch_size, height, width, depth, channels]
        filters (int): Number of output filters in the convolutional layer.
        activation (Callable, optional): Activation function to use. If `None`, no activation is applied.
        kernel_size (tuple, optional): Size of the 3D convolutional kernel. Defaults to (4, 4, 4).
        strides (tuple, optional): The strides of the deconvolution. Defaults to (2, 2, 2).
        padding (str, optional): The type of padding to apply. Defaults to 'same'.
        kernel_initializer (tf.keras.initializers.Initializer, optional): Initializer for the kernel weights. Defaults to None.
        gamma_initializer (tf.keras.initializers.Initializer, optional): Initializer for the gamma weights of instance normalization layer. Defaults to None.
        use_bias (bool, optional): Whether to include a bias term in the convolutional layer. Defaults to False.
    
    Returns:
        tf.Tensor: Output tensor of shape [batch_size, height, width, depth, filters].
    """
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
    """
    Upsamples the input tensor using 3D transposed convolution and applies instance normalization.
    
    Args:
        x (tf.Tensor): The input tensor.
        filters (int): The dimensionality of the output space.
        activation (Optional[Callable]): The activation function to use. Defaults to None.
        kernel_size (Tuple[int, int, int]): The size of the 3D transposed convolution window. Defaults to (4, 4, 4).
        strides (Tuple[int, int, int]): The strides of the 3D transposed convolution. Defaults to (2, 2, 2).
        padding (str): The type of padding to use. Defaults to 'same'.
        kernel_initializer (Optional[Callable]): The initializer for the kernel weights. Defaults to None.
        gamma_initializer (Optional[Callable]): The initializer for the gamma weights of the instance normalization layer. Defaults to None.
        use_bias (bool): Whether to include a bias vector in the convolution layer. Defaults to False.
    
    Returns:
        tf.Tensor: The upsampled tensor with instance normalization applied.
    """
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
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x