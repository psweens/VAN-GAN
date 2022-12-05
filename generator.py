from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_addons as tfa

from building_blocks import downsample, deconv, upsample, residual_block, ReflectionPadding3D

def get_resnet_generator(
    input_img_size=(64,64,512,1),
    batch_size=None,
    filters=32,
    num_downsampling_blocks=2,
    num_residual_blocks=6,
    num_upsample_blocks=2,
    gamma_initializer='he_normal',
    kernel_initializer='he_normal',
    name=None,
):
    
    ''' SWITCH 'SAME' PADDING TO REFLECTION PADDING? '''
    img_input = layers.Input(shape=input_img_size, batch_size=batch_size, name=name + "_img_input")
    x = ReflectionPadding3D(padding=(1, 1, 1))(img_input)

    for _ in range(1):
        x = layers.Conv3D(filters, (7, 7, 7), kernel_initializer=kernel_initializer,
                          use_bias=False)(x)
        x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
        x = layers.Activation("relu")(x)
        x = layers.SpatialDropout3D(0.5)(x)

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
    if num_downsampling_blocks == 2:
        x = ReflectionPadding3D(padding=(2, 2, 2))(x)
    x = layers.Conv3D(1, (7, 7, 7), padding="same")(x)
    x = layers.Activation("tanh")(x)

    model = keras.models.Model(img_input, x, name=name)

    model.summary()
    return model

