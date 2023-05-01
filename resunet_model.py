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
    Activation,
    Add,
    GaussianNoise
    )
import tensorflow_addons as tfa
from building_blocks import ReflectionPadding3D
from vnet_model import attention_concat

def norm_act(x, 
             act=True):
    """
    Apply instance normalization and activation function (ReLU by default) to input tensor.
    
    Args:
        x (tensor): Input tensor.
        act (bool): Whether to apply activation function. Default is True.
    
    Returns:
        tensor: Output tensor after instance normalization and activation (if applicable).
    
    """
    x = tfa.layers.InstanceNormalization()(x)
    if act == True:
        x = Activation("relu")(x)
    return x

def conv_block(x, 
               filters, 
               kernel_size=(3, 3, 3), 
               padding="valid", 
               strides=1, 
               kernel_initializer=None, 
               dropout_type=None, 
               dropout=None):
    """
    A convolutional block that consists of a normalization and activation layer followed by a convolutional layer.
    
    Args:
    x (tensor): Input tensor.
    filters (int): The number of filters in the convolutional layer.
    kernel_size (tuple, optional): The size of the convolutional kernel. Defaults to (3, 3, 3).
    padding (str, optional): The type of padding to apply. Defaults to "valid".
    strides (int, optional): The stride of the convolution. Defaults to 1.
    kernel_initializer (str, optional): The name of the kernel initializer to use. Defaults to None.
    dropout_type (str, optional): The type of dropout to apply. Defaults to None.
    dropout (float, optional): The dropout rate to apply. Defaults to None.
    
    Returns:
    tensor: The output tensor after passing through the convolutional block.
    """
    conv = norm_act(x)
    if padding == 'valid':
        conv = ReflectionPadding3D()(conv)
    conv = Conv3D(filters, kernel_size, padding=padding, strides=strides, kernel_initializer=kernel_initializer)(conv)
    return conv

def stem(x, 
         filters, 
         kernel_size=(3, 3, 3), 
         padding="valid", 
         strides=1):
    """
    The stem operation for the start of the deep residual UNet.
    
    Args:
        x (tensor): A 5D input tensor.
        filters (int): The number of filters to use in the convolutional layers.
        kernel_size (tuple, optional): The size of the convolutional kernel. Defaults to (3, 3, 3).
        padding (str, optional): The padding mode to use in the convolutional layers. Defaults to "valid".
        strides (int, optional): The stride of the convolutional layers. Defaults to 1.
    
    Returns:
        tensor: A 5D tensor with the same spatial dimensions as the input tensor.
    
    """
    if padding == 'valid':
        conv = ReflectionPadding3D()(x)
        conv = Conv3D(filters, kernel_size, padding=padding, strides=strides)(conv)
    else: 
        conv = Conv3D(filters, kernel_size, padding=padding, strides=strides)(x)
    conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    
    # Identity mapping
    shortcut = Conv3D(filters, kernel_size=(1, 1, 1), padding="same", strides=strides)(x)
    shortcut = norm_act(shortcut, act=False)
    
    output = Add()([conv, shortcut])
    return output

def residual_block(x, 
                   filters, 
                   kernel_size=(3, 3, 3), 
                   padding="valid", 
                   strides=1, 
                   kernel_initializer=None, 
                   dropout_type=None, 
                   dropout=None):
    """
    Constructs a residual block of the 3D residual UNet architecture. 
    
    Args:
        x (tensor): The input tensor.
        filters (int): Number of filters for the convolutional layers.
        kernel_size (tuple, optional): The kernel size of the convolutional layers. Defaults to (3, 3, 3).
        padding (str, optional): The type of padding used in the convolutional layers. Defaults to "valid".
        strides (int, optional): The stride length of the convolutional layers. Defaults to 1.
        kernel_initializer (str, optional): The initialization function for the kernel weights. Defaults to None.
        dropout_type (str, optional): The type of dropout regularization to use. Defaults to None.
        dropout (float, optional): The dropout rate to use if `dropout_type` is specified. Defaults to None.
    
    Returns:
        tensor: The output tensor of the residual block.
    """
    res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides, kernel_initializer=kernel_initializer, dropout_type=dropout_type, dropout=dropout)
    res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1, kernel_initializer=kernel_initializer, dropout_type=dropout_type, dropout=dropout)
    
    # Identity mapping
    shortcut = Conv3D(filters, kernel_size=(1, 1, 1), padding="same", strides=strides, kernel_initializer=kernel_initializer)(x)
    shortcut = norm_act(shortcut, act=False)
    
    output = Add()([shortcut, res])    
    if dropout_type == 'spatial':
        output = SpatialDropout3D(dropout)(output)
    elif dropout_type == 'standard':
        output = Dropout(dropout)(output)
    
    return output

def upsample_concat_block(x, xskip, filters, kernel_initializer=None, upsample_mode='deconv', padding='valid', use_attention_gate=False):

    if upsample_mode == 'deconv':
        if padding == 'valid':
            x = ReflectionPadding3D()(x)
        x = Conv3DTranspose(filters, (2, 2, 2), strides=(2, 2, 2), padding='valid')(x)
    else:
        x = UpSampling3D(size=2)(x)
        #x = Conv3D(int(filters/2), (5, 5, 5), strides=(1, 1, 1), padding='valid')(x)
    if use_attention_gate:
        x = attention_concat(conv_below=x, skip_connection=xskip)
    else:
        x = concatenate([x, xskip])
    return x

def ResUNet(
        input_shape,
        num_classes=1,
        activation='relu',
        use_batch_norm=True,
        upsample_mode='deconv',  # 'deconv' or 'simple'
        dropout=0.2,
        dropout_change_per_layer=0.0,
        dropout_type='none',
        use_dropout_on_upsampling=False,
        kernel_initializer='he_normal',
        gamma_initializer='he_normal',
        use_attention_gate=False,
        filters=16,
        num_layers=4,
        output_activation='tanh',
        use_input_noise=False
        ):
    
    f = [filters, filters*2, filters*4, filters*8, filters*16]
    inputs = Input(input_shape)
    skip_layers = []
    
    x = inputs
    
    if use_input_noise:
        x = GaussianNoise(0.2)(x)
    
    e = stem(x, f[0])
    skip_layers.append(x)
    
    # Encoder
    for e in range(1,num_layers+1):
        x = residual_block(x, f[e], strides=2, kernel_initializer=kernel_initializer, dropout_type=dropout_type, dropout=dropout+(e-1)*dropout_change_per_layer)
        skip_layers.append(x)
        
    # Bridge
    x = conv_block(x, f[-1], strides=1, kernel_initializer=kernel_initializer)
    # #d = spatial_attention(d)
    x = conv_block(x, f[-1], strides=1, kernel_initializer=kernel_initializer)
    
    for d in reversed(range(num_layers)):
        x = upsample_concat_block(x, skip_layers[d], f[d+1], kernel_initializer=kernel_initializer, upsample_mode=upsample_mode, use_attention_gate=use_attention_gate)
        x = residual_block(x, f[d], kernel_initializer=kernel_initializer)
    
    outputs = Conv3D(1, (1, 1, 1), padding="same", activation=output_activation)(x)
    
    model = Model(inputs, outputs)
    model.summary()
    return model