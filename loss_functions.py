import tensorflow as tf
import numpy as np
from utils import min_max_norm_tf
from clDice_func import soft_dice_cldice_loss


@tf.function
def reduce_mean(self, inputs, axis=None, keepdims=False):
    """
    Compute the mean of the inputs tensor along the given axis and divide by the global batch size.
    
    Args:
    - inputs: A tensor of values to compute the mean on.
    - axis: The dimensions along which to compute the mean. If None (default), compute the mean over all dimensions.
    - keepdims: If True, retains the reduced dimensions with length 1. If False (default), the reduced dimensions are removed.
    
    Returns:
    - A tensor with the mean of the inputs tensor along the given axis divided by the global batch size.
    """
    return tf.reduce_mean(inputs, axis=axis, keepdims=keepdims) / self.global_batch_size


@tf.function
def MSLE(self, real, fake):
    """
    Compute the per-sample mean squared logarithmic error (MSLE) between the real and fake tensors.
    
    Args:
    - real: A tensor of real values.
    - fake: A tensor of fake values.
    
    Returns:
    - A scalar tensor representing the per-sample MSLE between the real and fake tensors.
    """
    return reduce_mean(self, tf.square(tf.math.log(real + 1.) - tf.math.log(fake + 1.)),
                       axis=list(range(1, len(real.shape))))


@tf.function
def MAE(self, y_true, y_pred):
    """
    Compute the per-sample mean absolute error (MAE) between the true and predicted tensors.

    Args:
    - y_true: A tensor of true values.
    - y_pred: A tensor of predicted values.

    Returns:
    - A scalar tensor representing the per-sample MAE between the true and predicted tensors.
    """
    return reduce_mean(self, tf.abs(y_true - y_pred), axis=list(range(1, len(y_true.shape))))


@tf.function
def MSE(self, y_true, y_pred):
    """
    Compute the per-sample mean squared error (MSE) between the true and predicted tensors.

    Args:
    - y_true: A tensor of true values.
    - y_pred: A tensor of predicted values.

    Returns:
    - A scalar tensor representing the per-sample MSE between the true and predicted tensors.
    """
    return reduce_mean(self, tf.square(y_true - y_pred), axis=list(range(1, len(y_true.shape))))


@tf.function
def L4(self, y_true, y_pred):
    """
    Compute the per-sample L4 loss between the true and predicted tensors.
    
    Args:
    - y_true: A tensor of true values.
    - y_pred: A tensor of predicted values.
    
    Returns:
    - A scalar tensor representing the per-sample L4 loss between the true and predicted tensors.
    """
    return reduce_mean(self, tf.math.pow(y_true - y_pred, 4), axis=list(range(1, len(y_true.shape))))

@tf.function
def ssim_loss_3d(y_true, y_pred, max_val=1.0, filter_size=3, filter_sigma=1.5, k1=0.01, k2=0.03):

    # Create Gaussian filter
    def gaussian_filter(size, sigma):
        grid = tf.range(-size // 2 + 1, size // 2 + 1, dtype=tf.float32)
        gaussian_filter = tf.exp(-0.5 * (grid / sigma)**2) / (sigma * tf.sqrt(2.0 * np.pi))
        return gaussian_filter / tf.reduce_sum(gaussian_filter)

    # Create 3D Gaussian filter
    filter_3d = gaussian_filter(filter_size, filter_sigma)
    filter_3d = tf.einsum('i,j,k->ijk', filter_3d, filter_3d, filter_3d)
    filter_3d = filter_3d[:, :, :, tf.newaxis, tf.newaxis]

    # Compute mean and variance
    mu_true = tf.nn.conv3d(y_true, filter_3d, strides=[1, 1, 1, 1, 1], padding='SAME')
    mu_pred = tf.nn.conv3d(y_pred, filter_3d, strides=[1, 1, 1, 1, 1], padding='SAME')
    mu_true_sq = mu_true**2
    mu_pred_sq = mu_pred**2
    mu_true_pred = mu_true * mu_pred

    sigma_true_sq = tf.nn.conv3d(y_true**2, filter_3d, strides=[1, 1, 1, 1, 1], padding='SAME') - mu_true_sq
    sigma_pred_sq = tf.nn.conv3d(y_pred**2, filter_3d, strides=[1, 1, 1, 1, 1], padding='SAME') - mu_pred_sq
    sigma_true_pred = tf.nn.conv3d(y_true * y_pred, filter_3d, strides=[1, 1, 1, 1, 1], padding='SAME') - mu_true_pred

    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2

    ssim_map = (2 * mu_true_pred + c1) * (2 * sigma_true_pred + c2) / ((mu_true_sq + mu_pred_sq + c1) * (sigma_true_sq + sigma_pred_sq + c2))

    # Compute the mean SSIM loss across the batch
    return 1.0 - tf.reduce_mean(ssim_map)


@tf.function
def wasserstein_loss(prob_real_is_real, prob_fake_is_real):
    """
    Compute the Wasserstein loss between the probabilities that the real inputs are real and the generated inputs are real.
    
    Args:
    - prob_real_is_real: A tensor representing the probability that the real inputs are real.
    - prob_fake_is_real: A tensor representing the probability that the generated inputs are real.
    
    Returns:
    - A scalar tensor representing the Wasserstein loss between the two input probability tensors.
    """
    return tf.reduce_mean(prob_real_is_real - prob_fake_is_real)


@tf.function
def matched_crop(self, stack, axis=None, rescale=False):
    """
    Randomly crop the input tensor `stack` (which is compose of two image stacks) along a specified axis and return the resulting cropped tensors.

    Args:
    - stack: A tensor to be cropped.
    - axis: The axis along which to crop the input tensor. If axis=1, the input tensor will be cropped horizontally; if axis=3, it will be cropped vertically. Defaults to None.
    - rescale: A Boolean value indicating whether to rescale the resulting tensor values between 0 and 1. Defaults to False.

    Returns:
    - A tuple containing two cropped tensors of the same shape as the input tensor.
    """
    if axis == 1:
        shape = (self.batch_size, 2 * self.img_size[1], self.img_size[2], 1, self.channels)
        raxis = 3
    elif axis == 3:
        shape = (self.batch_size, 1, self.img_size[2], 2 * self.img_size[3], self.channels)
        raxis = 1
        axis -= 1

    arr = tf.squeeze(tf.image.random_crop(stack, size=shape),
                     axis=raxis)
    if rescale:
        arr = min_max_norm_tf(arr)
    return tf.split(arr, num_or_size_splits=2, axis=axis)


@tf.function
def cycle_loss(self, real_image, cycled_image, typ=None):
    """
    Compute the cycle consistency loss between real and cycled images.

    Args:
        self (object): The instance of the class.
        real_image (tensor): The tensor of real images.
        cycled_image (tensor): The tensor of cycled images.
        typ (string): The type of loss to compute. It can be set to "mse", "L4", or None (default).

    Returns:
        tensor: The cycle consistency loss tensor.
    """
    if typ is None:
        return MAE(self, real_image, cycled_image) * self.lambda_cycle
    elif typ == "mse":
        return MSE(self, real_image, cycled_image) * self.lambda_cycle
    elif typ == "L4":
        return L4(self, real_image, cycled_image) * self.lambda_cycle
    else:
        real = min_max_norm_tf(real_image, axis=(1, 2, 3, 4))
        cycled = min_max_norm_tf(cycled_image, axis=(1, 2, 3, 4))
        loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
        return reduce_mean(self, loss_obj(real, cycled)) * self.lambda_cycle


@tf.function
def cycle_reconstruction(self, real_image, cycled_image):
    """
    Return the per sample cycle reconstruction loss using Structural Similarity Index (SSIM) loss

    Args:
    - real_image: Tensor, shape (batch_size, H, W, C), representing the real image
    - cycled_image: Tensor, shape (batch_size, H, W, C), representing the cycled image

    Returns:
    - loss: float Tensor, representing the per sample cycle reconstruction loss
    """
    real = min_max_norm_tf(real_image, axis=(1, 2, 3, 4))
    cycled = min_max_norm_tf(cycled_image, axis=(1, 2, 3, 4))
    return reduce_mean(self, ssim_loss_3d(real, cycled, max_val=1.0)) * self.lambda_cycle


@tf.function
def cycle_seg_loss(self, real_image, cycled_image):
    """
    Compute the segmentation loss between the real image and the cycled image
    
    Args:
    - real_image: a tensor of shape (batch_size, image_size, image_size, channels) representing the real image
    - cycled_image: a tensor of shape (batch_size, image_size, image_size, channels) representing the cycled image
    
    Returns:
    - a scalar tensor representing the segmentation loss
    """
    real = min_max_norm_tf(real_image, axis=(1, 2, 3, 4))
    cycled = min_max_norm_tf(cycled_image, axis=(1, 2, 3, 4))
    cl_loss_obj = soft_dice_cldice_loss()
    return reduce_mean(self, cl_loss_obj(real, cycled)) * self.lambda_cycle


@tf.function
def identity_loss(self, real_image, same_image, typ=None):
    """
    Compute the identity loss between the real image and the same image.

    Args:
        real_image: the real image
        same_image: the generated same image
        typ: the type of loss to use. Currently only supports "cldice", other MAE used.

    Returns:
        The identity loss between the real image and the same image.
    """
    if typ is None:
        return self.lambda_identity * MAE(self, real_image, same_image)
    else:
        if typ == "cldice":
            real = min_max_norm_tf(real_image)
            same = min_max_norm_tf(same_image)
            loss_obj = soft_dice_cldice_loss()
            # bf_loss_obj = tf.keras.losses.BinaryFocalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
            # id_loss = reduce_mean(self, bf_loss_obj(real, same_image)) * self.lambda_identity
            spat_loss = reduce_mean(self, loss_obj(real, same)) * self.lambda_identity
            return spat_loss


@tf.function
def generator_loss_fn(self, fake_image, typ=None, from_logits=True):
    """
    Calculates the loss for the generator.

    Args:
        self (object): Instance of the VANGAN class.
        fake_image (tf.Tensor): Generated fake image tensor.
        typ (str): Type of loss. If None, default MSE is used.
                   Else, the valid types are: "bce" - Binary cross-entropy,
                   "bfce" - Binary focal cross-entropy.
                   Default: None.
        from_logits (bool): Whether to use logits or probabilities.
                            Default: True.

    Returns:
        tf.Tensor: The generator loss.
    """
    if typ is None:
        return MSE(self, tf.ones_like(fake_image), fake_image)
    else:
        if typ == "bce":
            loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits,
                                                          reduction=tf.keras.losses.Reduction.NONE)
        elif typ == "bfce":
            loss_obj = tf.keras.losses.BinaryFocalCrossentropy(from_logits=from_logits,
                                                               reduction=tf.keras.losses.Reduction.NONE)
        fake = fake_image
        if not from_logits:
            fake = min_max_norm_tf(fake, axis=(1, 2, 3, 4))
        loss = loss_obj(tf.ones_like(fake_image), fake)
        return reduce_mean(self, loss)


@tf.function
def discriminator_loss_fn(self, real_image, fake_image, typ=None, from_logits=True):
    """
    Calculates the loss for the discriminator network.

    Args:
        self: The instance of the VANGAN model.
        real_image: A tensor representing the real image.
        fake_image: A tensor representing the fake image.
        typ (str, optional): The type of loss function to use. Defaults to None.
        from_logits (bool, optional): Whether to apply sigmoid activation function to the predictions. 
            Defaults to True.

    Returns:
        A tensor representing the discriminator loss.
    """

    if typ is None:
        return 0.5 * (
                MSE(self, tf.ones_like(real_image), real_image) + MSE(self, tf.zeros_like(fake_image), fake_image))
    else:
        if typ == "bce":
            loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits,
                                                          reduction=tf.keras.losses.Reduction.NONE)
        elif typ == "bfce":
            loss_obj = tf.keras.losses.BinaryFocalCrossentropy(from_logits=from_logits,
                                                               reduction=tf.keras.losses.Reduction.NONE)
        real = real_image
        fake = fake_image
        if from_logits == False:
            real = min_max_norm_tf(real)
            fake = min_max_norm_tf(fake)
        loss = (loss_obj(tf.ones_like(real), real) + loss_obj(tf.zeros_like(fake), fake)) * 0.5
        return reduce_mean(self, loss)


def wasserstein_discriminator_loss(self, prob_real_is_real, prob_fake_is_real):
    """Computes the Wassertein-GAN loss as minimized by the discriminator.
    From paper :
     WasserteinGAN : https://arxiv.org/pdf/1701.07875.pdf
     by Martin Arjovsky, Soumith Chintala and Léon Bottou
    Args:
        prob_real_is_real: The discriminator's estimate that images actually
            drawn from the real domain are in fact real.
        prob_fake_is_real: The discriminator's estimate that generated images
            made to look like real images are real.
    Returns:
        The total W-GAN loss.
    """
    return -reduce_mean(self, prob_real_is_real - prob_fake_is_real)


def wasserstein_generator_loss(self, prob_fake_is_real):
    """Computes the Wassertein-GAN loss as minimized by the generator.
    From paper :
     WasserteinGAN : https://arxiv.org/pdf/1701.07875.pdf
     by Martin Arjovsky, Soumith Chintala and Léon Bottou
    Args:
        prob_real_is_real: The discriminator's estimate that images actually
            drawn from the real domain are in fact real.
        prob_fake_is_real: The discriminator's estimate that generated images
            made to look like real images are real.
    Returns:
        The total W-GAN loss.
    """

    return -reduce_mean(self, prob_fake_is_real)
