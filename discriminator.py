import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from building_blocks import downsample, ReflectionPadding3D, ReflectionPadding2D, StandardisationLayer
from utils import clip_images


def get_discriminator(
        input_img_size=(64, 64, 512, 1),
        batch_size=None,
        filters=64,
        kernel_initializer='he_normal',
        num_downsampling=3,
        use_dropout=False,
        dropout_rate=0.2,
        wasserstein=False,
        use_SN=False,
        use_input_noise=False,
        use_layer_noise=False,
        use_standardisation=False,
        name=None,
        noise_std=0.1,
        dim=3  # Use 3 for 3D, and 2 for 2D
):
    """
    Creates a discriminator model for both 2D and 3D images using convolutional layers.

    Args:
    - input_img_size: Tuple, the shape of the input image in the form (height, width, depth, channels) for 3D, or
                      (height, width, channels) for 2D.
                      Default is (64, 64, 512, 1) for 3D.
    - batch_size: Int, the batch size of the input images. Default is None.
    - filters: Int, the number of filters to use in the first layer of the model. Default is 64.
    - kernel_initializer: The initializer for the convolutional kernels. Default is None.
    - num_downsampling: Int, the number of times to downsample the input image with convolutional layers.
                        Default is 3.
    - use_dropout: Bool, whether to use dropout in the model. Default is False.
    - wasserstein: Bool, whether the model is a Wasserstein GAN. Default is False.
    - use_spec_norm: Bool, whether to use spectral normalization in the convolutional layers. Default is False.
    - use_input_noise: Bool, whether to add Gaussian noise to the input image. Default is False.
    - use_layer_noise: Bool, whether to add Gaussian noise to the convolutional layers. Default is False.
    - name: String, name for the model. Default is None.
    - noise_std: Float, the standard deviation of the Gaussian noise to add to the input and/or convolutional layers.
                 Default is 0.1.
    - dim: Int, whether to create a 2D or 3D model. Use `dim=3` for 3D and `dim=2` for 2D.

    Returns:
    - A tensorflow model representing the discriminator.
    """

    # Adjust padding and convolution layers based on the input dimension
    if dim == 3:
        ReflectionPadding = ReflectionPadding3D
        Conv = layers.Conv3D
        GaussianNoise = layers.GaussianNoise
    else:
        ReflectionPadding = ReflectionPadding2D
        Conv = layers.Conv2D
        GaussianNoise = layers.GaussianNoise

    img_input = layers.Input(
        shape=input_img_size, batch_size=batch_size, name=name + "_img_input"
    )

    if use_standardisation:
        x = StandardisationLayer()(img_input)
        x = ReflectionPadding()(x)
    else:
        x = ReflectionPadding()(img_input)

    if use_input_noise:
        x = GaussianNoise(noise_std)(x)

    x = Conv(
        filters,
        (4, 4, 4) if dim == 3 else (4, 4),
        strides=(2, 2, 2) if dim == 3 else (2, 2),
        padding="valid",
        kernel_initializer=kernel_initializer,
    )(x)

    x = tfa.layers.InstanceNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    num_filters = filters
    for num_downsample_block in range(num_downsampling):
        num_filters *= 2
        if num_downsample_block < 2:
            x = downsample(
                x,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4, 4, 4) if dim == 3 else (4, 4),
                strides=(2, 2, 2) if dim == 3 else (2, 2),
                use_dropout=use_dropout,
                dropout_rate=dropout_rate,
                use_spec_norm=use_SN,
                use_layer_noise=use_layer_noise,
                noise_std=noise_std,
                dim=dim  # Pass dimension to downsample
            )
        else:
            x = downsample(
                x,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4, 4, 4) if dim == 3 else (4, 4),
                strides=(1, 1, 1) if dim == 3 else (1, 1),
                use_dropout=use_dropout,
                dropout_rate=dropout_rate,
                padding='same',
                use_spec_norm=use_SN,
                use_layer_noise=use_layer_noise,
                noise_std=noise_std,
                dim=dim  # Pass dimension to downsample
            )

    if use_layer_noise:
        x = GaussianNoise(noise_std)(x)

    x = Conv(
        1,
        (3, 3, 3) if dim == 3 else (3, 3),
        strides=(1, 1, 1) if dim == 3 else (1, 1),
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
