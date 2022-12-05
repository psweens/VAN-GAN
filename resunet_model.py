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
    )
import tensorflow_addons as tfa
from building_blocks import ReflectionPadding3D
from vnet_model import attention_concat
from spatial_attention import spatial_attention

def bn_act(x, act=True):
    x = BatchNormalization()(x)
    # x = tfa.layers.InstanceNormalization()(x)
    if act == True:
        x = Activation("relu")(x)
    return x

def conv_block(x, filters, kernel_size=(3, 3, 3), padding="valid", strides=1, kernel_initializer=None, dropout_type=None, dropout=None):
    conv = bn_act(x)
    if padding == 'valid':
        conv = ReflectionPadding3D()(conv)
    conv = Conv3D(filters, kernel_size, padding=padding, strides=strides, kernel_initializer=kernel_initializer)(conv)
    # if dropout_type == 'spatial':
    #     conv = SpatialDropout3D(dropout)(conv)
    # elif dropout_type == 'standard':
    #     conv = Dropout(dropout)(conv)
    return conv

def stem(x, filters, kernel_size=(3, 3, 3), padding="valid", strides=1):
    if padding == 'valid':
        conv = ReflectionPadding3D()(x)
    conv = Conv3D(filters, kernel_size, padding=padding, strides=strides)(conv)
    conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    
    # Identity mapping
    shortcut = Conv3D(filters, kernel_size=(1, 1, 1), padding="same", strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    
    output = Add()([conv, shortcut])
    return output

def residual_block(x, filters, kernel_size=(3, 3, 3), padding="valid", strides=1, kernel_initializer=None, dropout_type=None, dropout=None):
    res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides, kernel_initializer=kernel_initializer, dropout_type=dropout_type, dropout=dropout)
    res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1, kernel_initializer=kernel_initializer, dropout_type=dropout_type, dropout=dropout)
    
    # Identity mapping
    shortcut = Conv3D(filters, kernel_size=(1, 1, 1), padding="same", strides=strides, kernel_initializer=kernel_initializer)(x)
    shortcut = bn_act(shortcut, act=False)
    
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
        residual_layers=4,
        filters=16,
        num_layers=4,
        output_activation='tanh'
        ):
    
    f = [filters, filters*2, filters*4, filters*8, filters*16]
    inputs = Input(input_shape)
    
    ## Encoder
    e0 = inputs
    e1 = stem(e0, f[0])
    e2 = residual_block(e1, f[1], strides=2, kernel_initializer=kernel_initializer, dropout_type=dropout_type, dropout=dropout)
    e3 = residual_block(e2, f[2], strides=2, kernel_initializer=kernel_initializer, dropout_type=dropout_type, dropout=dropout+dropout_change_per_layer)
    e4 = residual_block(e3, f[3], strides=2, kernel_initializer=kernel_initializer, dropout_type=dropout_type, dropout=dropout+dropout_change_per_layer)
    e5 = residual_block(e4, f[4], strides=2, kernel_initializer=kernel_initializer, dropout_type=dropout_type, dropout=dropout+dropout_change_per_layer)
    
    ## Bridge
    d = conv_block(e5, f[4], strides=1, kernel_initializer=kernel_initializer)
    #d = spatial_attention(d)
    d = conv_block(d, f[4], strides=1, kernel_initializer=kernel_initializer)
    
    ## Decoder
    d = upsample_concat_block(d, e4, f[4], kernel_initializer=kernel_initializer, upsample_mode=upsample_mode, use_attention_gate=use_attention_gate)
    d = residual_block(d, f[4], kernel_initializer=kernel_initializer)
    
    d = upsample_concat_block(d, e3, f[3], kernel_initializer=kernel_initializer, upsample_mode=upsample_mode, use_attention_gate=use_attention_gate)
    d = residual_block(d, f[3], kernel_initializer=kernel_initializer)
    
    d = upsample_concat_block(d, e2, f[2], kernel_initializer=kernel_initializer, upsample_mode=upsample_mode, use_attention_gate=use_attention_gate)
    d = residual_block(d, f[2], kernel_initializer=kernel_initializer)
    
    d = upsample_concat_block(d, e1, f[1], kernel_initializer=kernel_initializer, upsample_mode=upsample_mode, use_attention_gate=use_attention_gate)
    d = residual_block(d, f[1], kernel_initializer=kernel_initializer)
    
    outputs = Conv3D(1, (1, 1, 1), padding="same", activation=output_activation)(d)
    model = Model(inputs, outputs)
    model.summary()
    return model