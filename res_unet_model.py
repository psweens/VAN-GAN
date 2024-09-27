import tensorflow_addons as tfa
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Conv3D,
    Conv2DTranspose,
    Conv3DTranspose,
    MaxPooling3D,
    Dropout,
    SpatialDropout2D,
    SpatialDropout3D,
    UpSampling2D,
    UpSampling3D,
    Input,
    concatenate,
    multiply,
    add,
    Activation,
    Add,
    GaussianNoise
)
from building_blocks import ReflectionPadding3D, ReflectionPadding2D, StandardisationLayer
from vnet_model import attention_concat

import tensorflow as tf


def norm_act(x, act=True):
    """
    Apply instance normalization and activation function (ReLU by default) to input tensor.
    """
    x = tfa.layers.InstanceNormalization()(x)
    if act:
        x = Activation("relu")(x)
    return x


def conv_block(x,
               filters,
               kernel_size=(3, 3, 3),
               padding="valid",
               strides=1,
               kernel_initializer='he_normal',
               dim=3):
    """
    A convolutional block that supports both 2D and 3D data by adjusting kernel and convolution type.
    """
    conv = norm_act(x)
    if padding == 'valid':
        if dim == 3:
            conv = ReflectionPadding3D()(conv)
        else:
            conv = ReflectionPadding2D()(conv)

    if dim == 3:
        conv = Conv3D(filters, kernel_size, padding=padding, strides=strides, kernel_initializer=kernel_initializer)(
            conv)
    else:
        conv = Conv2D(filters, kernel_size, padding=padding, strides=strides, kernel_initializer=kernel_initializer)(
            conv)

    return conv


def stem(x,
         filters,
         kernel_size=(3, 3, 3),
         padding="valid",
         strides=1,
         dim=3):
    """
    The stem operation, generalized to handle both 2D and 3D input.
    """
    if padding == 'valid':
        if dim == 3:
            conv = ReflectionPadding3D()(x)
            conv = Conv3D(filters, kernel_size, padding=padding, strides=strides)(conv)
        else:
            conv = ReflectionPadding2D()(x)
            conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
    else:
        if dim == 3:
            conv = Conv3D(filters, kernel_size, padding=padding, strides=strides)(x)
        else:
            conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)

    conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides, dim=dim)

    # Identity mapping
    if dim == 3:
        shortcut = Conv3D(filters, kernel_size=(1, 1, 1), padding="same", strides=strides)(x)
    else:
        shortcut = Conv2D(filters, kernel_size=(1, 1), padding="same", strides=strides)(x)

    shortcut = norm_act(shortcut, act=False)
    output = Add()([conv, shortcut])
    return output


def residual_block(x,
                   filters,
                   kernel_size=(3, 3, 3),
                   padding="valid",
                   strides=1,
                   kernel_initializer='he_normal',
                   dropout_type=None,
                   dropout=None,
                   dim=3):
    """
    Residual block that supports both 2D and 3D convolutions.
    """
    res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides,
                     kernel_initializer=kernel_initializer, dim=dim)
    res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1,
                     kernel_initializer=kernel_initializer, dim=dim)

    # Identity mapping
    if dim == 3:
        shortcut = Conv3D(filters, kernel_size=(1, 1, 1), padding="same", strides=strides,
                          kernel_initializer=kernel_initializer)(x)
    else:
        shortcut = Conv2D(filters, kernel_size=(1, 1), padding="same", strides=strides,
                          kernel_initializer=kernel_initializer)(x)

    shortcut = norm_act(shortcut, act=False)
    output = Add()([shortcut, res])

    if dropout_type == 'spatial':
        if dim == 3:
            output = SpatialDropout3D(dropout)(output)
        else:
            output = SpatialDropout2D(dropout)(output)
    elif dropout_type == 'standard':
        output = Dropout(dropout)(output)

    return output


# Define an attention gate function for segmentation
def attention_gate(x, g, inter_shape, dim=2):
    """
    Attention gate for U-Net-like architectures suited for segmentation.
    Parameters:
    - x: skip connection (input from encoder)
    - g: gating signal (input from decoder)
    - inter_shape: number of filters for intermediate convolutions
    - dim: 2 for 2D, 3 for 3D
    """
    if dim == 3:
        conv = Conv3D
        add_layer = Add
        activation = Activation('sigmoid')
    else:
        conv = Conv2D
        add_layer = Add
        activation = Activation('sigmoid')

    # Theta_x -> convolution of the skip connection
    theta_x = conv(inter_shape, (1, 1), strides=(1, 1), padding='same')(x)

    # Phi_g -> convolution of the gating signal
    phi_g = conv(inter_shape, (1, 1), strides=(1, 1), padding='same')(g)

    # Add both convolutions
    attn = add_layer()([theta_x, phi_g])
    attn = Activation('relu')(attn)

    # Psi -> Convolution for generating the attention weights
    psi = conv(1, (1, 1), padding='same')(attn)
    psi = activation(psi)

    # Multiply the attention map with the skip connection
    x_attn = multiply([x, psi])

    return x_attn


# Update the upsample_concat_block function to include attention gates
def upsample_concat_block(x,
                          xskip,
                          filters,
                          kernel_initializer='he_normal',
                          upsample_mode='deconv',
                          padding='valid',
                          use_attention_gate=False,
                          dim=3):
    """
    Upsample and concatenate block for U-Net with optional attention gates, supports both 2D and 3D.
    """
    if upsample_mode == 'deconv':
        if padding == 'valid':
            if dim == 3:
                x = ReflectionPadding3D()(x)
            else:
                x = ReflectionPadding2D()(x)

        if dim == 3:
            x = Conv3DTranspose(filters, (2, 2, 2), strides=(2, 2, 2), padding='valid',
                                kernel_initializer=kernel_initializer)(x)
        else:
            x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='valid',
                                kernel_initializer=kernel_initializer)(x)
    else:
        if dim == 3:
            x = UpSampling3D(size=2)(x)
        else:
            x = UpSampling2D(size=2)(x)

    # Apply attention gate if specified
    if use_attention_gate:
        xskip = attention_gate(xskip, x, filters // 2, dim=dim)

    # Concatenate with skip connection
    x = concatenate([x, xskip])
    return x


def res_unet(
        input_shape,
        dim=3,  # '3' for 3D, '2' for 2D
        upsample_mode='deconv',  # 'deconv' or 'simple'
        dropout=0.2,
        dropout_change_per_layer=0.0,
        dropout_type='none',
        kernel_initializer='he_normal',
        use_attention_gate=False,
        filters=16,
        num_layers=4,
        output_activation='tanh',
        use_input_noise=False
):
    """
    Create a Residual U-Net model for either 2D or 3D input.
    """
    f = [filters, filters * 2, filters * 4, filters * 8, filters * 16]
    inputs = Input(input_shape)
    skip_layers = []

    x = inputs

    if use_input_noise:
        x = GaussianNoise(0.2)(x)

    x = stem(x, f[0], dim=dim)
    skip_layers.append(x)

    # Encoder
    for e in range(1, num_layers + 1):
        x = residual_block(x, f[e], strides=2, kernel_initializer=kernel_initializer, dropout_type=dropout_type,
                           dropout=dropout + (e - 1) * dropout_change_per_layer, dim=dim)
        skip_layers.append(x)

    # Bridge
    x = conv_block(x, f[-1], strides=1, kernel_initializer=kernel_initializer, dim=dim)
    x = conv_block(x, f[-1], strides=1, kernel_initializer=kernel_initializer, dim=dim)

    # Decoder
    for d in reversed(range(num_layers)):
        x = upsample_concat_block(x, skip_layers[d], f[d + 1], kernel_initializer=kernel_initializer,
                                  upsample_mode=upsample_mode, use_attention_gate=use_attention_gate, dim=dim)
        x = residual_block(x, f[d], kernel_initializer=kernel_initializer, dim=dim)

    # Output Layer
    if output_activation == 'norm':
        if dim == 3:
            x = Conv3D(1, (1, 1, 1), padding="same")(x)
        else:
            x = Conv2D(1, (1, 1), padding="same")(x)
    else:
        if dim == 3:
            x = Conv3D(1, (1, 1, 1), padding="same", activation=output_activation)(x)
        else:
            x = Conv2D(1, (1, 1), padding="same", activation=output_activation)(x)

    model = Model(inputs, x)
    model.summary()
    return model
