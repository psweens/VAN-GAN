import tensorflow as tf
from keras import layers as KL
from keras import backend as K

''' Based on: https://github.com/jocpae/clDice'''


@tf.function(jit_compile=True)          # XLA → one fused GPU kernel
def soft_erode(x: tf.Tensor) -> tf.Tensor:
    if x.shape.rank == 4:               # N H W C
        k31 = -tf.nn.max_pool2d(-x, ksize=[1, 3], strides=1, padding="SAME")
        k13 = -tf.nn.max_pool2d(-x, ksize=[3, 1], strides=1, padding="SAME")
        return tf.minimum(k31, k13)     # ⊥-shaped structuring element
    else:                               # N D H W C
        k = -tf.nn.max_pool3d(-x, ksize=[1, 3, 3], strides=1, padding="SAME")
        k = tf.minimum(k, -tf.nn.max_pool3d(-x, ksize=[3, 1, 3],
                                            strides=1, padding="SAME"))
        return tf.minimum(k, -tf.nn.max_pool3d(-x, ksize=[3, 3, 1],
                                               strides=1, padding="SAME"))



def soft_dilate(img):
    """
    Perform soft dilation on a given image tensor.

    Args:
    img (tf.Tensor): Input image tensor on which soft dilation will be performed.

    Returns:
    (tf.Tensor): Image tensor after performing soft dilation.
    """
    if len(img.shape) == 4:
        return KL.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same', data_format=None)(img)
    else:
        return KL.MaxPool3D(pool_size=(3, 3, 3), strides=(1, 1, 1), padding='same', data_format=None)(img)


def soft_open(img):
    """
    Perform soft opening on a given image tensor.

    Args:
    img (tf.Tensor): Input image tensor on which soft opening will be performed.

    Returns:
    (tf.Tensor): Image tensor after performing soft opening.
    """
    img = soft_erode(img)
    img = soft_dilate(img)
    return img

@tf.function(jit_compile=True)          # XLA ⇒ single fused kernel
def soft_skel(img, max_iters=30):
    """Skeletonise `img` but break the loop once the skeleton stops growing."""
    img   = tf.cast(img, tf.float32)
    skel  = tf.zeros_like(img)
    it    = tf.constant(0, dtype=tf.int32)

    # ── upper bound for the number of iterations ───────────────────────
    if max_iters is None:
        stat_shape = img.shape.as_list()
        if None not in stat_shape[1:-1]:            # known statically
            max_iters = max(stat_shape[1:-1])
        else:                                       # dynamic shape
            max_iters = tf.reduce_max(tf.shape(img)[1:-1])

    # *** NEW: flag that tells us whether any new voxels appeared ***
    has_new = tf.constant(True)

    # ── loop guard ──
    def cond(i, cur_img, cur_skel, has_new):
        return tf.logical_and(tf.less(i, max_iters), has_new)

    # ── loop body ──
    def body(i, cur_img, cur_skel, _):
        opened  = soft_open(cur_img)
        delta   = tf.nn.relu(cur_img - opened)
        new_vox = tf.nn.relu(delta - cur_skel * delta)
        next_skel = cur_skel + new_vox
        next_img  = soft_erode(cur_img)

        # *** NEW: did we add anything at this step? ***
        new_flag = tf.reduce_any(new_vox > 0.0)

        return i + 1, next_img, next_skel, new_flag

    # ── run the loop ──
    _, _, skel, _ = tf.while_loop(
        cond,
        body,
        loop_vars=[it, img, skel, has_new],
        shape_invariants=[
            it.get_shape(),          # scalar
            img.get_shape(),         # same shape each iter
            img.get_shape(),         # … likewise
            has_new.get_shape()      # scalar bool
        ]
    )
    return skel

def soft_clDice_loss(y_true, y_pred, iter_=None):
    """
    clDice loss that no longer *requires* the `iter_` argument.
    Pass it only if you want to override the automatic behaviour.
    """
    smooth = 1.0
    skel_pred = soft_skel(y_pred, iter_)
    skel_true = soft_skel(y_true, iter_)
    pres = (K.sum(skel_pred * y_true) + smooth) / (K.sum(skel_pred) + smooth)
    rec  = (K.sum(skel_true * y_pred) + smooth) / (K.sum(skel_true) + smooth)
    return 1.0 - 2.0 * (pres * rec) / (pres + rec)


def soft_dice(y_true, y_pred):
    """
    Compute the soft Dice loss.

    Args:
    y_true (tf.Tensor): The ground truth segmentation mask tensor.
    y_pred (tf.Tensor): The predicted segmentation mask tensor.

    Returns:
    (tf.Tensor): The computed soft Dice loss.
    """
    smooth = 1
    intersection = K.sum((y_true * y_pred))
    coeff = (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)
    return 1. - coeff


def soft_dice_cldice_loss(iters=30, alpha=0.5):
    """
    Compute the combined soft Dice and clDice loss, a variant of the Dice loss used in segmentation tasks.

    Args:
    iters (int, optional): The number of iterations for skeletonisation. Defaults to 15.
    alpha (float, optional): The weight for the clDice component. Defaults to 0.5.

    Returns:
    (function): The loss function to be used in training.
    """

    def loss(y_true, y_pred):
        cl_dice = soft_clDice_loss(y_true, y_pred, iters)
        dice = soft_dice(y_true, y_pred)
        return (1.0 - alpha) * dice + alpha * cl_dice

    return loss