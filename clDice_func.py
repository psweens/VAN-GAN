import tensorflow as tf
from keras import layers as KL
from keras import backend as K

''' Based on: https://github.com/jocpae/clDice'''


def soft_erode(img):
    """
    Perform soft erosion on a given image tensor.

    Args:
    img (tf.Tensor): Input image tensor on which soft erosion will be performed.

    Returns:
    (tf.Tensor): Image tensor after performing soft erosion.
    """
    if len(img.shape) == 4:
        p2 = -KL.MaxPool2D(pool_size=(3, 1), strides=(1, 1), padding='same', data_format=None)(-img)
        p3 = -KL.MaxPool2D(pool_size=(1, 3), strides=(1, 1), padding='same', data_format=None)(-img)
        return tf.math.minimum(p2, p3)
    else:
        p1 = -KL.MaxPool3D(pool_size=(3, 3, 1), strides=(1, 1, 1), padding='same', data_format=None)(-img)
        p2 = -KL.MaxPool3D(pool_size=(3, 1, 3), strides=(1, 1, 1), padding='same', data_format=None)(-img)
        p3 = -KL.MaxPool3D(pool_size=(1, 3, 3), strides=(1, 1, 1), padding='same', data_format=None)(-img)
        return tf.math.minimum(tf.math.minimum(p1, p2), p3)


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


def soft_skel(img, iters):
    """
    Perform soft skeletonisation on a given image tensor.

    Args:
    img (tf.Tensor): Input image tensor on which skeletonisation will be performed.
    iters (int): Number of iterations for skeletonisation.

    Returns:
    (tf.Tensor): Skeletonised image tensor after performing soft skeletonisation.
    """
    img1 = soft_open(img)
    skel = tf.nn.relu(img - img1)

    for j in range(iters):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = tf.nn.relu(img - img1)
        intersect = tf.math.multiply(skel, delta)
        skel += tf.nn.relu(delta - intersect)
    return skel


def soft_clDice_loss(y_true, y_pred, iter_=50):
    """
    Compute the soft centre-line (clDice) loss, which is a variant of the Dice loss used in segmentation tasks.

    Args:
    y_true (tf.Tensor): The ground truth segmentation mask tensor.
    y_pred (tf.Tensor): The predicted segmentation mask tensor.
    iter_ (int, optional): The number of iterations for skeletonization. Defaults to 50.

    Returns:
    (tf.Tensor): The computed soft clDice loss.
    """
    smooth = 1.
    skel_pred = soft_skel(y_pred, iter_)
    skel_true = soft_skel(y_true, iter_)
    pres = (K.sum(tf.math.multiply(skel_pred, y_true)) + smooth) / (K.sum(skel_pred) + smooth)
    rec = (K.sum(tf.math.multiply(skel_true, y_pred)) + smooth) / (K.sum(skel_true) + smooth)
    cl_dice = 1. - 2.0 * (pres * rec) / (pres + rec)

    return cl_dice


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


def soft_dice_cldice_loss(iters=15, alpha=0.5):
    """
    Compute the combined soft Dice and clDice loss, a variant of the Dice loss used in segmentation tasks.

    Args:
    iters (int, optional): The number of iterations for skeletonisation. Defaults to 15.
    alpha (float, optional): The weight for the clDice component. Defaults to 0.5.

    Returns:
    (function): The loss function to be used in training.
    """

    def loss(y_true, y_pred):
        """
        Compute the combined soft Dice and clDice loss for a single batch of data.

        Args:
        y_true (tf.Tensor): The ground truth segmentation mask tensor.
        y_pred (tf.Tensor): The predicted segmentation mask tensor.

        Returns:
        (tf.Tensor): The computed combined loss value.
        """
        cl_dice = soft_clDice_loss(y_true, y_pred, iters)
        dice = soft_dice(y_true, y_pred)
        return (1.0 - alpha) * dice + alpha * cl_dice

    return loss
