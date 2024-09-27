from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv3D,
    Conv3DTranspose,
    MaxPooling3D,
    Dropout,
    SpatialDropout3D,
    UpSampling3D,
    Input,
    concatenate,
    multiply,
    add,
    Activation
)
import tensorflow as tf
import tensorflow_addons as tfa
from building_blocks import ReflectionPadding3D
from utils import min_max_norm_tf, rescale_arr_tf

'''https://github.com/karolzak/keras-unet'''


def attention_gate(inp_1, inp_2, n_intermediate_filters):
    """
    Attention gate. Compresses both inputs to `n_intermediate_filters` filters before processing.
    Implemented as proposed by Oktay et al. in their Attention U-Net, see: https://arxiv.org/abs/1804.03999.

    Args:
    inp_1 (tf.Tensor): First input tensor to the attention gate.
    inp_2 (tf.Tensor): Second input tensor to the attention gate (skip-connection).
    n_intermediate_filters (int): Number of intermediate filters to use in the attention gate.

    Returns:
    (tf.Tensor): Output tensor after applying the attention gate.
    """
    inp_1_conv = Conv3D(
        n_intermediate_filters,
        kernel_size=1,
        strides=1,
        padding='same',
        kernel_initializer='he_normal',
    )(inp_1)
    inp_2_conv = Conv3D(
        n_intermediate_filters,
        kernel_size=1,
        strides=1,
        padding='same',
        kernel_initializer='he_normal',
    )(inp_2)

    f = Activation('relu')(add([inp_1_conv, inp_2_conv]))
    g = Conv3D(
        filters=1,
        kernel_size=1,
        strides=1,
        padding='same',
        kernel_initializer='he_normal',
    )(f)
    h = Activation('sigmoid')(g)
    return multiply([inp_1, h])


def attention_concat(conv_below, skip_connection):
    """
    Concatenates the upsampled `conv_below` with the attention-gated version of `skip_connection`.

    Args:
    conv_below (tf.Tensor): The upsampled tensor that will be concatenated.
    skip_connection (tf.Tensor): The skip-connection tensor used for attention gating.

    Returns:
    (tf.Tensor): Output tensor after concatenation with attention gating.
    """
    below_filters = conv_below.get_shape().as_list()[-1]
    attention_across = attention_gate(skip_connection, conv_below, below_filters)
    return concatenate([conv_below, attention_across])


def conv3d_block(
        inputs,
        use_batch_norm=True,
        dropout=0.3,
        dropout_type='spatial',
        filters=16,
        kernel_size=(3, 3, 3),
        activation='relu',
        kernel_initializer='he_normal',
        padding='valid',
):
    """
    Create a 3D convolutional block consisting of two convolutional layers with optional batch normalization and
    dropout.

    Args:
    inputs (tf.Tensor): Input tensor to the convolutional block.
    use_batch_norm (bool): Whether to use batch normalization.
    dropout (float): Dropout rate for spatial or standard dropout (if enabled).
    dropout_type (str): Type of dropout, either 'spatial' or 'standard'.
    filters (int): Number of filters (output channels) in the convolutional layers.
    kernel_size (tuple): Size of the convolutional kernel in 3D (depth, height, width).
    activation (str): Activation function to be used after convolution.
    kernel_initializer (str): Initializer for the convolutional kernel weights.
    padding (str): Padding mode for the convolutional layers.

    Returns:
    (tf.Tensor): Output tensor after passing through the 3D convolutional block.
    """

    if dropout_type == 'spatial':
        do = SpatialDropout3D
    elif dropout_type == 'standard':
        do = Dropout
    else:
        raise ValueError(
            f"dropout_type must be one of ['spatial', 'standard'], got {dropout_type}"
        )
    c = ReflectionPadding3D()(inputs)
    c = Conv3D(
        filters,
        kernel_size,
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=not use_batch_norm,
    )(c)
    if use_batch_norm:
        c = BatchNormalization()(c)
    else:
        #c = GroupNormalization(groups=1, axis=-1)(c)
        c = tfa.layers.InstanceNormalization()(c)
    if dropout > 0.0:
        c = do(dropout)(c)
    c = ReflectionPadding3D()(c)
    c = Conv3D(
        filters,
        kernel_size,
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=not use_batch_norm,
    )(c)
    if use_batch_norm:
        c = BatchNormalization()(c)
    else:
        #c = GroupNormalization(groups=1, axis=-1)(c)
        c = tfa.layers.InstanceNormalization()(c)
    return c


def custom_vnet(
        input_shape,
        num_classes=1,
        activation='relu',
        use_batch_norm=True,
        upsample_mode='deconv',  # 'deconv' or 'simple'
        dropout=0.5,
        dropout_change_per_layer=0.0,
        dropout_type='spatial',
        use_dropout_on_upsampling=False,
        kernel_initializer='he_normal',
        use_attention_gate=False,
        filters=16,
        num_layers=4,
        output_activation='sigmoid',
        addnoise=False
):  # 'sigmoid' or 'softmax'

    """
    Customizable VNet architecture based on the work of
    Fausto Milletari, Nassir Navab, Seyed-Ahmad Ahmadi in
    V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation

    Arguments: input_shape: 4D Tensor of shape (x, y, z, num_channels) num_classes (int): Unique classes in the
    output mask. Should be set to 1 for binary segmentation activation (str): A keras.activations.Activation to use.
    ReLu by default. use_batch_norm (bool): Whether to use Batch Normalisation across the channel axis between
    convolutional layers upsample_mode (one of 'deconv' or 'simple'): Whether to use transposed convolutions or
    simple upsampling in the decoder part dropout (float between 0. and 1.): Amount of dropout after the initial
    convolutional block. Set to 0. to turn Dropout off dropout_change_per_layer (float between 0. and 1.): Factor to
    add to the Dropout after each convolutional block dropout_type (one of 'spatial' or 'standard'): Type of Dropout
    to apply. Spatial is recommended for CNNs [2] use_dropout_on_upsampling (bool): Whether to use dropout in the
    decoder part of the network use_attention_gate (bool): Whether to use an attention dynamic when concatenating
    with the skip-connection, implemented as proposed by Oktay et al. [3] filters (int): Convolutional filters in the
    initial convolutional block. Will be doubled every block num_layers (int): Number of total layers in the encoder
    not including the bottleneck layer output_activation (str): A keras.activations.Activation to use. Sigmoid by
    default for binary segmentation

    Returns:
    model (keras.models.Model): The built V-Net

    Raises:
    ValueError: If dropout_type is not one of 'spatial' or 'standard'


    [1]: https://arxiv.org/abs/1505.04597
    [2]: https://arxiv.org/pdf/1411.4280.pdf
    [3]: https://arxiv.org/abs/1804.03999

    """

    # Build model
    inputs = Input(input_shape)
    x = inputs

    if addnoise:
        x = min_max_norm_tf(x) + tf.random.normal(shape=tf.shape(x),
                                                  mean=-0.475,
                                                  stddev=0.06)
        x = tf.math.add(x, inputs)
        x = tf.clip_by_value(x, 0., 1.)
        x = rescale_arr_tf(x, -0.5, 0.5)

    down_layers = []
    for layer in range(num_layers):
        x = conv3d_block(
            inputs=x,
            filters=filters,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            dropout_type=dropout_type,
            kernel_initializer=kernel_initializer,
            activation=activation,
        )
        down_layers.append(x)
        x = MaxPooling3D((2, 2, 2))(x)
        dropout += dropout_change_per_layer
        filters = filters * 2  # double the number of filters with each layer

    x = conv3d_block(
        inputs=x,
        filters=filters,
        use_batch_norm=use_batch_norm,
        dropout=dropout,
        dropout_type=dropout_type,
        kernel_initializer=kernel_initializer,
        activation=activation,
    )

    if not use_dropout_on_upsampling:
        dropout = 0.0
        dropout_change_per_layer = 0.0

    for conv in reversed(down_layers):
        filters //= 2  # decreasing number of filters with each layer
        dropout -= dropout_change_per_layer
        if upsample_mode == 'deconv':
            x = Conv3DTranspose(filters, (2, 2, 2), strides=(2, 2, 2), padding='same')(x)
        else:
            x = UpSampling3D(size=2)(x)
            x = Conv3D(filters, (3, 3, 3), strides=(1, 1, 1), padding='same', kernel_initializer=kernel_initializer)(x)
        if use_attention_gate:
            x = attention_concat(conv_below=x, skip_connection=conv)
        else:
            x = concatenate([x, conv])

        x = conv3d_block(
            inputs=x,
            filters=filters,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            dropout_type=dropout_type,
            kernel_initializer=kernel_initializer,
            activation=activation,
        )

    outputs = Conv3D(num_classes, (1, 1, 1), activation=output_activation)(x)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.summary()
    return model
