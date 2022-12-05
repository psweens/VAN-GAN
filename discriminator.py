from tensorflow import keras
from tensorflow.keras import layers
from building_blocks import downsample, ReflectionPadding3D
import tensorflow_addons as tfa
import tensorflow as tf

def get_discriminator(
    input_img_size=(64, 64, 512, 1),
    batch_size=None,
    filters=64,
    kernel_initializer=None,
    num_downsampling=3,
    use_dropout=False,
    wasserstein=False,
    use_SN=False,
    use_input_noise=False,
    use_layer_noise=False,
    name=None,
    noise_std=0.1
):

    img_input = layers.Input(
        shape=input_img_size, batch_size=batch_size, name=name + "_img_input"
    )
    if use_input_noise:
        x = ReflectionPadding3D()(img_input)
        x = layers.GaussianNoise(noise_std)(x)
    else:
        x = ReflectionPadding3D()(img_input)
        

    if use_SN:
        x = tfa.layers.SpectralNormalization(layers.Conv3D(
            filters,
            (4, 4, 4),
            strides=(2, 2, 2),
            padding="valid",
            kernel_initializer=kernel_initializer,
        ))(x)
    else:
        x = layers.Conv3D(
            filters,
            (4, 4, 4),
            strides=(2, 2, 2),
            padding="valid",
            kernel_initializer=kernel_initializer,
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        # x = tfa.layers.InstanceNormalization(gamma_initializer=None)(x)
    
    x = layers.LeakyReLU(0.2)(x)

    num_filters = filters
    for num_downsample_block in range(num_downsampling):
        num_filters *= 2
        if num_downsample_block < 2:
            x = downsample(
                x,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4, 4, 4),
                strides=(2, 2, 2),
                use_dropout=use_dropout,
                use_SN=use_SN,
                use_layer_noise=use_layer_noise,
                noise_std=noise_std
            )
        else:
            x = downsample(
                x,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4, 4, 4),
                strides=(1, 1, 1),
                use_dropout=use_dropout,
                padding='same',
                use_SN=use_SN,
                use_layer_noise=use_layer_noise,
                noise_std=noise_std
            )

    if use_layer_noise:
        x = layers.GaussianNoise(noise_std)(x)
        
    x = layers.Conv3D(
        1,
        (3, 3, 3),
        strides=(1, 1, 1),
        padding="same",
        kernel_initializer=kernel_initializer,
    )(x)
    
    if wasserstein:
        x = layers.Flatten()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(1)(x)

    model = keras.models.Model(inputs=img_input, outputs=x, name=name)

    model.summary()
    return model
