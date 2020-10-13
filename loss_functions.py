import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras as tfk
import tensorflow_addons as tfa

# Define the loss function for the generators
def generatorA_loss_fn(fake):
    # Loss function for evaluating adversarial loss
    adv_loss_fn = tfa.losses.SigmoidFocalCrossEntropy(from_logits = False,
                                                      reduction=tfk.losses.Reduction.AUTO)
    fake_loss = adv_loss_fn(tf.ones_like(fake), fake)
    return fake_loss

def generatorB_loss_fn(fake):
    # Loss function for evaluating adversarial loss
    adv_loss_fn = tfk.losses.BinaryCrossentropy(reduction=tfk.losses.Reduction.AUTO)
    fake_loss = adv_loss_fn(tf.ones_like(fake), fake)
    return fake_loss

# Define the loss function for the discriminators
def discriminatorA_loss_fn(real, fake):
    # Loss function for evaluating adversarial loss
    adv_loss_fn = tfa.losses.SigmoidFocalCrossEntropy(from_logits = False)
    real_loss = adv_loss_fn(tf.ones_like(real), real)
    fake_loss = adv_loss_fn(tf.zeros_like(fake), fake)
    return (real_loss + fake_loss) * 0.5

def discriminatorB_loss_fn(real, fake):
    # Loss function for evaluating adversarial loss
    adv_loss_fn = tfa.losses.SigmoidFocalCrossEntropy(from_logits = False)
    real_loss = adv_loss_fn(tf.ones_like(real), real)
    fake_loss = adv_loss_fn(tf.zeros_like(fake), fake)
    return (real_loss + fake_loss) * 0.5

def spatial_loss_fn(real, real_map):
    adv_loss_fn = tfk.losses.MeanAbsoluteError()
    return adv_loss_fn(real_map, real)

def gram_matrix(x, norm_by_channels=False):
    '''
    Returns the Gram matrix of the tensor x.
    '''

    if K.ndim(x) == 3:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
        shape = K.shape(x)
        C, H, W = shape[0], shape[1], shape[2]
        gram = K.dot(features, K.transpose(features))
    elif K.ndim(x) == 5:
        # Swap from (H, W, C) to (B, C, H, W)
        x = K.permute_dimensions(x, (0, 4, 1, 2, 3))
        shape = K.shape(x)
        B, C, H, W, D = shape[0], shape[1], shape[2], shape[3], shape[4]
        # Reshape as a batch of 2D matrices with vectorized channels
        features = K.reshape(x, K.stack([B, C, H*W*D]))
        # This is a batch of Gram matrices (B, C, C).
        gram = K.batch_dot(features, features, axes=2)
    else:
        raise ValueError('The input tensor should be either a 3d (H, W, C) or 5d (B, H, W, D, C) tensor.')
    # Normalize the Gram matrix
    if norm_by_channels:
        denominator = C * H * W * D # Normalization from Johnson
    else:
        denominator = H * W * D # Normalization from Google
    # gram = gram /  K.cast(denominator, x.dtype)

    return gram

def style_loss(style, combination, img_size):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = img_size[3]
    size = img_size[0] * img_size[1] * img_size[2]
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.batch_flatten(y_true)
    y_pred_f = K.batch_flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f, axis=1)
    return (2. * intersection + smooth) / (K.sum(y_true_f, axis=1) + K.sum(y_pred_f, axis=1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)
