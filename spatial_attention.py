''' https://github.com/clguo/SA-UNet/blob/master/Spatial_Attention.py '''
import tensorflow.keras.backend as K
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, \
    Conv2D, Add, Activation, Lambda,Conv1D
def spatial_attention(input_feature):
    kernel_size = 7

    channel = input_feature._keras_shape[-1]
    cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=4, keepdims=True))(cbam_feature)
    assert avg_pool._keras_shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=4, keepdims=True))(cbam_feature)
    assert max_pool._keras_shape[-1] == 1
    concat = Concatenate(axis=4)([avg_pool, max_pool])
    assert concat._keras_shape[-1] == 2
    cbam_feature = Conv3D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    assert cbam_feature._keras_shape[-1] == 1

    return multiply([input_feature, cbam_feature])