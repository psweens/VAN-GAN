import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from building_blocks import downsample, ReflectionPadding2D, ReflectionPadding3D
from tensorflow.keras.layers import (
    Activation,
    Add,
    concatenate,
    Conv2D,
    Conv3D,
    Conv2DTranspose,
    Conv3DTranspose,
    Dropout,
    GaussianNoise,
    Input,
    LeakyReLU,
    Multiply,
    SpatialDropout2D,
    SpatialDropout3D,
    UpSampling2D,
    UpSampling3D
)


class ConvolutionalDiscriminator(Model):
    def __init__(self, input_img_size=(64, 64, 512, 1), batch_size=None,
                 filters=64, kernel_initializer='he_normal',
                 num_downsampling=3, use_dropout=False, dropout_rate=0.2,
                 wasserstein=False, use_SN=False, use_input_noise=False,
                 use_layer_noise=False, noise_std=0.1, dim=3, name=None):
        super(ConvolutionalDiscriminator, self).__init__()
        self.input_img_size = input_img_size
        self.batch_size = batch_size
        self.filters = filters
        self.kernel_initializer = kernel_initializer
        self.num_downsampling = num_downsampling
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.wasserstein = wasserstein
        self.use_SN = use_SN
        self.use_input_noise = use_input_noise
        self.use_layer_noise = use_layer_noise
        self.noise_std = noise_std
        self.dim = dim
        self.discriminator_name = name

        # Define the layer types based on the dimension
        if self.dim == 3:
            self.Conv = Conv3D
            self.ConvTranspose = Conv3DTranspose
            self.UpSampling = UpSampling3D
            self.SpatialDropout = SpatialDropout3D
            self.ReflectionPadding = ReflectionPadding3D
        else:
            self.Conv = Conv2D
            self.ConvTranspose = Conv2DTranspose
            self.UpSampling = UpSampling2D
            self.SpatialDropout = SpatialDropout2D
            self.ReflectionPadding = ReflectionPadding2D

        # Build the model
        self.model = self.build_model()

    def call(self, inputs, **kwargs):
        return self.model(inputs)

    def build_model(self):

        img_input = Input(shape=self.input_img_size,
                          batch_size=self.batch_size,
                          name=self.discriminator_name + "_img_input" if self.name else None)

        x = self.ReflectionPadding()(img_input)

        if self.use_input_noise:
            x = GaussianNoise(self.noise_std)(x)

        x = self.Conv(filters=self.filters,
                      kernel_size=4,
                      strides=2,
                      padding="valid",
                      kernel_initializer=self.kernel_initializer)(x)
        x = tfa.layers.InstanceNormalization()(x)

        x = LeakyReLU(0.2)(x)

        for i in range(self.num_downsampling):
            filters_multiplier = 2 ** (i + 1)
            strides = 4 if i < 2 else 1
            padding = 'same' if i >= 2 else 'valid'

            x = downsample(
                x,
                filters=self.filters * filters_multiplier,
                activation=LeakyReLU(0.2),
                kernel_size=4,
                strides=strides,
                use_dropout=self.use_dropout if i < 2 else False,
                dropout_rate=self.dropout_rate,
                use_spec_norm=self.use_SN,
                use_layer_noise=self.use_layer_noise,
                noise_std=self.noise_std,
                dim=self.dim,
                padding=padding
            )

        if self.use_layer_noise:
            x = GaussianNoise(self.noise_std)(x)

        x = self.Conv(filters=1,
                      kernel_size=3,
                      strides=1,
                      padding="same",
                      kernel_initializer=self.kernel_initializer)(x)

        if self.wasserstein:
            x = layers.Flatten()(x)
            x = layers.Dropout(0.2)(x)
            x = layers.Dense(1)(x)

        model = keras.models.Model(inputs=img_input, outputs=x, name=self.name)

        model.summary()
        return model

    def downsample(self,
                   x,
                   filters,
                   activation,
                   kernel_size=3,
                   strides=2,
                   padding="valid",
                   gamma_initializer=None,
                   use_bias=False,
                   use_dropout=True,
                   use_spec_norm=False,
                   padding_size=1,
                   use_layer_noise=False,
                   noise_std=0.1):
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
            use_spec_norm (bool, optional): Whether to use Spectral Normalization. Defaults to False.
            padding_size (tuple of ints, optional): Padding size for ReflectionPadding3D. Defaults to (1, 1, 1).
            use_layer_noise (bool, optional): Whether to add Gaussian noise after ReflectionPadding3D. Defaults to False.
            noise_std (float, optional): Standard deviation of Gaussian noise. Defaults to 0.1.

        Returns:
            Tensor: The downsampled tensor.
        """
        padding_size = (padding_size, padding_size, padding_size) if self.dim == 3 else (padding_size, padding_size)

        if padding == 'valid':
            x = self.ReflectionPadding(padding_size)(x)

        if use_layer_noise:
            x = GaussianNoise(noise_std)(x)

        # if use_spec_norm:
        #     x = tfa.layers.SpectralNormalization(layers.Conv3D(
        #         filters,
        #         kernel_size,
        #         strides=strides,
        #         kernel_initializer=kernel_initializer,
        #         padding=padding,
        #         use_bias=use_bias
        #     ))(x)
        # else:
        x = self.Conv(filters=filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      kernel_initializer=self.kernel_initializer,
                      padding=padding,
                      use_bias=use_bias)(x)
        # x = layers.GroupNormalization(groups=1, axis=-1, gamma_initializer=gamma_initializer)(x)
        x = tfa.layers.InstanceNormalization()(x)

        if activation:
            x = activation(x)
            if use_dropout:
                x = self.SpatialDropout(self.dropout_rate)(x)
        return x
