import tensorflow as tf
import numpy as np
from utils import min_max_norm_tf, z_score_norm_tf
from clDice_func import soft_dice_cldice_loss
from cbDice_func import centreline_boundary_dice_loss_3d


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

    # Validate axes to ensure they're within the correct range of the input tensor's rank
    if axis is not None:
        # Ensure valid axis values
        input_rank = len(inputs.shape)
        valid_axes = [ax for ax in axis if ax < input_rank]
    else:
        valid_axes = None  # Reduce over all dimensions if no axis is specified

    # Compute the mean over the valid axes
    arr = tf.reduce_mean(inputs, axis=valid_axes, keepdims=keepdims)

    # Return the sum divided by the global batch size
    return tf.reduce_sum(arr) / self.global_batch_size


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
def custom_mse_loss(self, y_true, y_pred, focus='standard', focus_weight=10.0, mid_range_weight=0.5):
    """
    Custom MSE loss function where we can focus on pixel values in a certain region.

    Parameters:
    - y_true: The ground truth images (tensor).
    - y_pred: The predicted images (tensor).
    - focus: Can be 'standard', 'low', or 'high' to choose which part of the pixel values to focus on.
        - 'standard': regular MSE
        - 'low': focus more on values near -1
        - 'high': focus more on values near 1
    - focus_weight: A scalar that controls the weight given to the focused region (default is 5.0).

    Returns:
    - Tensor representing the computed loss.
    """

    if focus == 'standard':
        return self.MSE(y_true, y_pred, axis=list(range(1, len(y_true.shape))))  # Standard MSE without focusing on any specific region

    elif focus == 'low':
        # Focus on lower pixel values (near -1)
        low_focus_weight = tf.where(y_true < 0, focus_weight, 1.0)
        low_mse_loss = reduce_mean(self, low_focus_weight * tf.square(y_true - y_pred), axis=list(range(1, len(y_true.shape))))
        return low_mse_loss

    elif focus == 'high':
        # Focus on higher pixel values (near 1)
        high_focus_weight = tf.where(y_true > 0, focus_weight, 1.0)
        high_mse_loss = reduce_mean(self, high_focus_weight * tf.square(y_true - y_pred), axis=list(range(1, len(y_true.shape))))
        return high_mse_loss

    elif focus == 'low_high':
        # Assign higher weight to both low (negative) and high (positive) pixel values
        low_focus_weight = tf.where(y_true < -0.5, focus_weight, mid_range_weight)  # For values < -0.5
        high_focus_weight = tf.where(y_true > 0.5, focus_weight, mid_range_weight)  # For values > 0.5

        # Combine weights for all regions
        combined_weights = low_focus_weight + high_focus_weight

        return reduce_mean(self, combined_weights * tf.square(y_true - y_pred), axis=list(range(1, len(y_true.shape))))


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
def ssim_loss(y_true, y_pred, max_val=1.0, filter_size=3, filter_sigma=1.5, k1=0.01, k2=0.03, dim=2):
    """
    SSIM loss function that supports both 2D and 3D inputs.

    Args:
    - y_true: Ground truth tensor.
    - y_pred: Predicted tensor.
    - max_val: Maximum possible value of the image.
    - filter_size: Size of the Gaussian filter.
    - filter_sigma: Standard deviation of the Gaussian filter.
    - k1: Constant to stabilize the weak denominator (default: 0.01).
    - k2: Constant to stabilize the weak denominator (default: 0.03).
    - dim: Input dimensionality (2 for 2D inputs, 3 for 3D inputs).

    Returns:
    - SSIM loss value.
    """

    # Create Gaussian filter
    def gaussian_filter(size, sigma):
        grid = tf.range(-size // 2 + 1, size // 2 + 1, dtype=tf.float32)
        gaussian_filter = tf.exp(-0.5 * (grid / sigma) ** 2) / (sigma * tf.sqrt(2.0 * np.pi))
        return gaussian_filter / tf.reduce_sum(gaussian_filter)

    # Create the appropriate Gaussian filter for 2D or 3D
    filter_1d = gaussian_filter(filter_size, filter_sigma)

    if dim == 3:
        # Create 3D Gaussian filter
        filter_nd = tf.einsum('i,j,k->ijk', filter_1d, filter_1d, filter_1d)
        filter_nd = filter_nd[:, :, :, tf.newaxis, tf.newaxis]
        conv_fn = tf.nn.conv3d
        strides = [1, 1, 1, 1, 1]
    else:
        # Create 2D Gaussian filter
        filter_nd = tf.einsum('i,j->ij', filter_1d, filter_1d)
        filter_nd = filter_nd[:, :, tf.newaxis, tf.newaxis]
        conv_fn = tf.nn.conv2d
        strides = [1, 1, 1, 1]

    # Compute mean and variance using the appropriate convolution function
    mu_true = conv_fn(y_true, filter_nd, strides=strides, padding='SAME')
    mu_pred = conv_fn(y_pred, filter_nd, strides=strides, padding='SAME')
    mu_true_sq = mu_true ** 2
    mu_pred_sq = mu_pred ** 2
    mu_true_pred = mu_true * mu_pred

    sigma_true_sq = conv_fn(y_true ** 2, filter_nd, strides=strides, padding='SAME') - mu_true_sq
    sigma_pred_sq = conv_fn(y_pred ** 2, filter_nd, strides=strides, padding='SAME') - mu_pred_sq
    sigma_true_pred = conv_fn(y_true * y_pred, filter_nd, strides=strides, padding='SAME') - mu_true_pred

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    ssim_map = (2 * mu_true_pred + c1) * (2 * sigma_true_pred + c2) / (
            (mu_true_sq + mu_pred_sq + c1) * (sigma_true_sq + sigma_pred_sq + c2))

    # Compute the mean SSIM loss across the batch
    return 1.0 - ssim_map


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
        # return custom_mse_loss(self, real_image, cycled_image) * self.lambda_cycle
        return MSE(self, real_image, cycled_image) * self.lambda_cycle
    elif typ == "L4":
        return L4(self, real_image, cycled_image) * self.lambda_cycle
    else:
        # Dynamically determine axes based on tensor rank
        image_rank = len(real_image.shape)
        valid_axes = list(range(1, image_rank))  # All axes except batch dimension (axis 0)

        real = min_max_norm_tf(real_image, axis=valid_axes)
        cycled = min_max_norm_tf(cycled_image, axis=valid_axes)

        loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

        return reduce_mean(self, loss_obj(real, cycled), axis=valid_axes) * self.lambda_cycle


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
    # Dynamically determine axes based on tensor rank
    image_rank = len(real_image.shape)
    valid_axes = list(range(1, image_rank))  # All axes except batch dimension (axis 0)
    return reduce_mean(self,
                       ssim_loss(min_max_norm_tf(real_image, axis=valid_axes),
                                    min_max_norm_tf(cycled_image, axis=valid_axes), max_val=1.0),
                       axis=list(range(1, len(real_image.shape)))
                       ) * self.lambda_reconstruction


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
    image_rank = len(real_image.shape)
    valid_axes = list(range(1, image_rank))  # All axes except batch dimension (axis 0)
    real = min_max_norm_tf(real_image, axis=valid_axes)
    cycled = min_max_norm_tf(cycled_image, axis=valid_axes)
    cl_loss_obj = centreline_boundary_dice_loss_3d()
    return cl_loss_obj(real, cycled) * (self.lambda_topology / self.n_devices)


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
            # loss_obj = tf.keras.losses.BinaryFocalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
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
    image_rank = len(fake_image.shape)
    valid_axes = list(range(1, image_rank))  # All axes except batch dimension (axis 0)
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
            fake = min_max_norm_tf(fake, axis=valid_axes)
        loss = loss_obj(tf.ones_like(fake_image), fake)
        return reduce_mean(self, loss, axis=list(range(1, len(fake_image.shape))))


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
        return reduce_mean(self, loss, axis=list(range(1, len(real_image.shape))))
