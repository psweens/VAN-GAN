import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from building_blocks import downsample, upsample, residual_block, ReflectionPadding3D


def resnet(
        input_img_size=(64, 64, 512, 1),
        batch_size=None,
        filters=32,
        num_downsampling_blocks=2,
        num_residual_blocks=6,
        num_upsample_blocks=2,
        gamma_initializer='he_normal',
        kernel_initializer='he_normal',
        name=None,
):
    """
    Returns a 3D ResNet generator model.

    Args: input_img_size (tuple): The size of the input image (height, width, depth, channels). batch_size (int,
    optional): The batch size to be used for the model. Defaults to None. filters (int, optional): The number of
    filters in the first convolutional layer. Defaults to 32. num_downsampling_blocks (int, optional): The number of
    downsampling blocks in the generator. Defaults to 2. num_residual_blocks (int, optional): The number of residual
    blocks in the generator. Defaults to 6. num_upsample_blocks (int, optional): The number of upsampling blocks in
    the generator. Defaults to 2. gamma_initializer (str, optional): The initializer to be used for the instance
    normalization gamma. Defaults to 'he_normal'. kernel_initializer (str, optional): The initializer to be used for
    the convolutional kernels. Defaults to 'he_normal'. name (str, optional): The name of the model. Defaults to None.

    Returns:
        tensorflow.keras.models.Model: The 3D ResNet generator model.
    """

    img_input = layers.Input(shape=input_img_size, batch_size=batch_size, name=name + "_img_input")
    x = ReflectionPadding3D(padding=(1, 1, 1))(img_input)

    for _ in range(1):
        x = layers.Conv3D(filters, (7, 7, 7), kernel_initializer=kernel_initializer,
                          use_bias=False)(x)
        #x = layers.GroupNormalization(groups=1, axis=-1, gamma_initializer=gamma_initializer)(x)
        x = tfa.layers.InstanceNormalization()(x)
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
