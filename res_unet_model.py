import tensorflow_addons as tfa
from tensorflow.keras import Model
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
    UpSampling3D,
    #GroupNormalization,
)
from building_blocks import ReflectionPadding3D, ReflectionPadding2D


class ResUNet(Model):
    def __init__(self, input_shape, dim=3, upsample_mode='deconv', dropout=0.2,
                 dropout_change_per_layer=0.0, dropout_type='none', kernel_initializer='he_normal',
                 use_attention_gate=False, filters=16, num_layers=4, output_activation='tanh',
                 use_input_noise=False):
        super(ResUNet, self).__init__()
        self.dim = dim
        self.upsample_mode = upsample_mode
        self.dropout = dropout
        self.dropout_change_per_layer = dropout_change_per_layer
        self.dropout_type = dropout_type
        self.kernel_initializer = kernel_initializer
        self.use_attention_gate = use_attention_gate
        self.filters = filters
        self.num_layers = num_layers
        self.output_activation = output_activation
        self.use_input_noise = use_input_noise
        self.image_shape = input_shape

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
        f = [self.filters, self.filters * 2, self.filters * 4, self.filters * 8, self.filters * 16]
        inputs = Input(self.image_shape)
        skip_layers = []

        x = inputs

        if self.use_input_noise:
            x = GaussianNoise(0.2)(x)

        x = self.stem(x, f[0], dim=self.dim)
        skip_layers.append(x)

        # Encoder
        for e in range(1, self.num_layers + 1):
            x = self.residual_block(x,
                                    filters=f[e],
                                    strides=2,
                                    kernel_initializer=self.kernel_initializer,
                                    dropout_type=self.dropout_type,
                                    dropout=self.dropout + (e - 1) * self.dropout_change_per_layer, dim=self.dim)
            skip_layers.append(x)

        # Bridge
        x = self.conv_block(x, filters=f[-1], strides=1)
        x = self.conv_block(x, filters=f[-1], strides=1)

        # Decoder
        for d in reversed(range(self.num_layers)):
            x = self.upsample_concat_block(x, skip_layers[d], filters=f[d + 1])
            x = self.residual_block(x, filters=f[d])

        # Output Layer
        x = self.final_layer(x)

        model = Model(inputs, x)
        model.summary()
        return model

    def conv_block(self,
                   x,
                   filters,
                   kernel_size=3,
                   strides=1,
                   padding='valid'):
        """
        A convolutional block that supports both 2D and 3D data by adjusting kernel and convolution type.
        """

        x = self.norm_act(x)
        if padding == 'valid':
            x = self.ReflectionPadding()(x)

        x = self.Conv(filters=filters,
                      kernel_size=kernel_size,
                      padding=padding,
                      strides=strides,
                      kernel_initializer=self.kernel_initializer)(x)
        return x

    def stem(self,
             x,
             filters,
             kernel_size=3,
             padding="valid",
             strides=1,
             dim=3):
        """
        The stem operation, generalized to handle both 2D and 3D input.
        """
        y = self.ReflectionPadding()(x)
        y = self.Conv(filters=filters,
                      kernel_size=kernel_size,
                      padding=padding,
                      strides=strides)(y)

        y = self.conv_block(y,
                            filters=filters,
                            kernel_size=kernel_size,
                            padding=padding,
                            strides=strides)

        # Identity mapping
        shortcut = self.Conv(filters=filters,
                             kernel_size=1,
                             padding="same",
                             strides=strides)(x)

        shortcut = self.norm_act(shortcut, act=False)
        output = Add()([y, shortcut])
        return output

    def residual_block(self,
                       x,
                       filters,
                       kernel_size=3,
                       padding="valid",
                       strides=1,
                       kernel_initializer='he_normal',
                       dropout_type=None,
                       dropout=None,
                       dim=3):
        """
        Residual block that supports both 2D and 3D convolutions.
        """

        res = self.conv_block(x,
                              filters=filters,
                              kernel_size=kernel_size,
                              padding=padding,
                              strides=strides)
        res = self.conv_block(res,
                              filters=filters,
                              kernel_size=kernel_size,
                              padding=padding,
                              strides=1)

        # Identity mapping
        shortcut = self.Conv(filters,
                             kernel_size=1,
                             padding="same",
                             strides=strides,
                             kernel_initializer=kernel_initializer)(x)

        shortcut = self.norm_act(shortcut, act=False)
        output = Add()([shortcut, res])

        if dropout_type == 'spatial':
            output = self.SpatialDropout(dropout)(output)
        elif dropout_type == 'standard':
            output = Dropout(dropout)(output)

        return output

    def upsample_concat_block(self,
                              x,
                              xskip,
                              filters,
                              use_attention_gate=False):
        """
        Upsample and concatenate block for U-Net with optional attention gates, supports both 2D and 3D.
        """
        if self.upsample_mode == 'deconv':
            x = self.ReflectionPadding()(x)
            x = self.ConvTranspose(filters=filters,
                                   kernel_size=2,
                                   strides=2,
                                   padding='valid',
                                   kernel_initializer=self.kernel_initializer)(x)
        else:
            x = self.UpSampling(size=2)(x)

        # Apply attention gate if specified
        if use_attention_gate:
            xskip = self.attention_gate(xskip, x, filters // 2)

        # Concatenate with skip connection
        x = concatenate([x, xskip])
        return x

    def norm_act(self, x, act=True):
        """
        Apply instance normalization and activation function (ReLU by default) to input tensor.
        """
        x = tfa.layers.InstanceNormalization()(x)
        if act:
            x = Activation("relu")(x)
        return x

    def final_layer(self, x):
        # Output layer logic here
        if self.output_activation == 'norm':
            x = self.Conv(filters=1,
                          kernel_size=1,
                          padding="same")(x)
        else:
            x = self.Conv(filters=1,
                          kernel_size=1,
                          padding="same",
                          activation=self.output_activation)(x)
        return x

    def attention_gate(self,
                       x,
                       g,
                       inter_shape):
        """
        Attention gate for U-Net-like architectures suited for segmentation.
        Parameters:
        - x: skip connection (input from encoder)
        - g: gating signal (input from decoder)
        - inter_shape: number of filters for intermediate convolutions
        - dim: 2 for 2D, 3 for 3D
        """
        # Theta_x -> convolution of the skip connection
        theta_x = self.Conv(filters=inter_shape,
                            kernel_size=1,
                            strides=1,
                            padding='same')(x)

        # Phi_g -> convolution of the gating signal
        phi_g = self.Conv(filters=inter_shape,
                          kernel_size=1,
                          strides=1,
                          padding='same')(g)

        # Add both convolutions
        attn = Add()([theta_x, phi_g])
        attn = LeakyReLU(0.2)(attn)

        # Psi -> Convolution for generating the attention weights
        psi = self.Conv(filters=1,
                        kernel_size=1,
                        padding='same',
                        activation='tanh')(attn)

        # Multiply the attention map with the skip connection
        x_attn = Multiply()([x, psi])

        return x_attn

