import tensorflow as tf
import numpy as np
from utils import min_max_norm_tf, z_score_norm_tf
from clDice_func import soft_dice_cldice_loss
from cbDice_func import centreline_boundary_dice_loss_3d


# ------------------------------------------------------------------------------
# Helper function: reduce_mean
# ------------------------------------------------------------------------------

@tf.function
def reduce_mean(self, inputs, axis=None, keepdims=False):
    """
    Compute the mean of the inputs tensor along the given axis and then
    divides the result by the global batch size.

    Args:
        inputs: A tensor of values.
        axis: The dimensions along which to compute the mean.
              (Default: all dimensions except the batch dimension.)
        keepdims: Whether to retain reduced dimensions.

    Returns:
        A scalar tensor: the sum of the mean‐reduced tensor divided by the global batch size.

    Note:
        If your inputs have dynamic shape, you may replace the static axes with:

            axes = tf.range(1, tf.rank(inputs))

        instead of using: list(range(1, len(inputs.shape))).
    """
    if axis is None:
        # Using static shape if available.
        axes = list(range(1, len(inputs.shape)))
    else:
        axes = [ax for ax in axis if ax < len(inputs.shape)]
    arr = tf.reduce_mean(inputs, axis=axes, keepdims=keepdims)
    return tf.reduce_sum(arr) / self.global_batch_size


# ------------------------------------------------------------------------------
# Basic Loss Functions
# ------------------------------------------------------------------------------

@tf.function
def MSLE(self, real, fake):
    """
    Per-sample Mean Squared Logarithmic Error between real and fake tensors.
    """
    return reduce_mean(
        self,
        tf.square(tf.math.log(real + 1.) - tf.math.log(fake + 1.)),
        axis=list(range(1, len(real.shape)))
    )


@tf.function
def MAE(self, y_true, y_pred):
    """
    Per-sample Mean Absolute Error.
    """
    return reduce_mean(
        self,
        tf.abs(y_true - y_pred),
        axis=list(range(1, len(y_true.shape)))
    )


@tf.function
def MSE(self, y_true, y_pred):
    """
    Per-sample Mean Squared Error.
    """
    return reduce_mean(
        self,
        tf.square(y_true - y_pred),
        axis=list(range(1, len(y_true.shape)))
    )


@tf.function
def custom_mse_loss(self, y_true, y_pred, focus='standard', focus_weight=10.0, mid_range_weight=0.5):
    """
    Custom MSE loss where different pixel ranges can be weighted differently.

    Args:
        y_true: Ground truth tensor.
        y_pred: Predicted tensor.
        focus: Determines which region to emphasize.
            - 'standard': standard MSE,
            - 'low': higher weight for values near -1,
            - 'high': higher weight for values near 1,
            - 'low_high': high weight for both extremes.
        focus_weight: Weight for the focused region.
        mid_range_weight: Weight for mid-range pixel values (when focus == 'low_high').

    Returns:
        A scalar loss value.
    """
    if focus == 'standard':
        # Use the standard MSE (no extra axis argument is passed here)
        return self.MSE(y_true, y_pred)
    elif focus == 'low':
        low_focus_weight = tf.where(y_true < 0, focus_weight, 1.0)
        return reduce_mean(
            self,
            low_focus_weight * tf.square(y_true - y_pred),
            axis=list(range(1, len(y_true.shape)))
        )
    elif focus == 'high':
        high_focus_weight = tf.where(y_true > 0, focus_weight, 1.0)
        return reduce_mean(
            self,
            high_focus_weight * tf.square(y_true - y_pred),
            axis=list(range(1, len(y_true.shape)))
        )
    elif focus == 'low_high':
        low_focus_weight = tf.where(y_true < -0.5, focus_weight, mid_range_weight)
        high_focus_weight = tf.where(y_true > 0.5, focus_weight, mid_range_weight)
        combined_weights = low_focus_weight + high_focus_weight
        return reduce_mean(
            self,
            combined_weights * tf.square(y_true - y_pred),
            axis=list(range(1, len(y_true.shape)))
        )


@tf.function
def L4(self, y_true, y_pred):
    """
    Per-sample L4 loss.
    """
    return reduce_mean(
        self,
        tf.math.pow(y_true - y_pred, 4),
        axis=list(range(1, len(y_true.shape)))
    )


@tf.function
def ssim_loss(y_true, y_pred, max_val=1.0, filter_size=3, filter_sigma=1.5, k1=0.01, k2=0.03, dim=2):
    """
    Structural Similarity (SSIM) loss function for 2D or 3D inputs.

    Returns:
        A tensor representing 1.0 minus the SSIM map.
    """

    # Create a 1D Gaussian filter.
    def gaussian_filter(size, sigma):
        grid = tf.range(-size // 2 + 1, size // 2 + 1, dtype=tf.float32)
        gauss = tf.exp(-0.5 * (grid / sigma) ** 2) / (sigma * tf.sqrt(2.0 * np.pi))
        return gauss / tf.reduce_sum(gauss)

    filter_1d = gaussian_filter(filter_size, filter_sigma)

    if dim == 3:
        filter_nd = tf.einsum('i,j,k->ijk', filter_1d, filter_1d, filter_1d)
        filter_nd = filter_nd[:, :, :, tf.newaxis, tf.newaxis]
        conv_fn = tf.nn.conv3d
        strides = [1, 1, 1, 1, 1]
    else:
        filter_nd = tf.einsum('i,j->ij', filter_1d, filter_1d)
        filter_nd = filter_nd[:, :, tf.newaxis, tf.newaxis]
        conv_fn = tf.nn.conv2d
        strides = [1, 1, 1, 1]

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

    ssim_map = (2 * mu_true_pred + c1) * (2 * sigma_true_pred + c2) / \
               ((mu_true_sq + mu_pred_sq + c1) * (sigma_true_sq + sigma_pred_sq + c2))

    return 1.0 - ssim_map


@tf.function
def ms_ssim_loss(y_true, y_pred,
                 max_val=1.0,
                 filter_size=3,
                 filter_sigma=1.5,
                 k1=0.01,
                 k2=0.03,
                 levels=5):
    """
    Multi-Scale Structural Similarity (MS-SSIM) loss for single-channel 3D images.

    Args:
        y_true: Ground truth tensor of shape [batch, D, H, W, 1].
        y_pred: Predicted tensor of shape [batch, D, H, W, 1].
        max_val: The dynamic range of the images (default 1.0).
        filter_size: Size of the 1D Gaussian filter.
        filter_sigma: Standard deviation of the Gaussian filter.
        k1: Constant for luminance term.
        k2: Constant for contrast-structure term.
        levels: Number of scales (levels) to use.

    Returns:
        A tensor containing the MS-SSIM loss (1 - MS-SSIM) per batch element.
    """

    # Define a helper to create a 1D Gaussian filter.
    def gaussian_filter(size, sigma):
        grid = tf.range(-size // 2 + 1, size // 2 + 1, dtype=tf.float32)
        # The normalization constants cancel out in SSIM, so we omit them here.
        gauss = tf.exp(-0.5 * (grid / sigma) ** 2)
        return gauss / tf.reduce_sum(gauss)

    # Create a 3D Gaussian kernel by taking an outer product of the 1D filter.
    filter_1d = gaussian_filter(filter_size, filter_sigma)
    # Create a separable 3D filter.
    filter_nd = tf.einsum('i,j,k->ijk', filter_1d, filter_1d, filter_1d)
    # Reshape to [filter_depth, filter_height, filter_width, in_channels, out_channels]
    filter_nd = filter_nd[:, :, :, tf.newaxis, tf.newaxis]

    # Lists to store the contrast-structure (cs) values at each scale.
    mcs = []

    # Loop over scales.
    for level in range(levels):
        # Compute local means via convolution.
        mu_true = tf.nn.conv3d(y_true, filter_nd, strides=[1, 1, 1, 1, 1], padding='SAME')
        mu_pred = tf.nn.conv3d(y_pred, filter_nd, strides=[1, 1, 1, 1, 1], padding='SAME')
        mu_true_sq = mu_true ** 2
        mu_pred_sq = mu_pred ** 2
        mu_true_pred = mu_true * mu_pred

        # Compute local variances and covariance.
        sigma_true_sq = tf.nn.conv3d(y_true ** 2, filter_nd, strides=[1, 1, 1, 1, 1], padding='SAME') - mu_true_sq
        sigma_pred_sq = tf.nn.conv3d(y_pred ** 2, filter_nd, strides=[1, 1, 1, 1, 1], padding='SAME') - mu_pred_sq
        sigma_true_pred = tf.nn.conv3d(y_true * y_pred, filter_nd, strides=[1, 1, 1, 1, 1],
                                       padding='SAME') - mu_true_pred

        # Constants for stability.
        c1 = (k1 * max_val) ** 2
        c2 = (k2 * max_val) ** 2

        # Compute the luminance component.
        l = (2 * mu_true_pred + c1) / (mu_true_sq + mu_pred_sq + c1)
        # Compute the contrast-structure component.
        cs = (2 * sigma_true_pred + c2) / (sigma_true_sq + sigma_pred_sq + c2)

        if level < levels - 1:
            # For scales 1..M-1, store the mean contrast-structure index.
            mcs.append(tf.reduce_mean(cs, axis=[1, 2, 3, 4]))
            # Downsample the images by a factor of 2 using average pooling.
            y_true = tf.nn.avg_pool3d(y_true,
                                      ksize=[1, 2, 2, 2, 1],
                                      strides=[1, 2, 2, 2, 1],
                                      padding='SAME')
            y_pred = tf.nn.avg_pool3d(y_pred,
                                      ksize=[1, 2, 2, 2, 1],
                                      strides=[1, 2, 2, 2, 1],
                                      padding='SAME')
        else:
            # At the final scale, use only the luminance term.
            l_last = tf.reduce_mean(l, axis=[1, 2, 3, 4])

    # Define typical weights for 5 scales (from Wang et al.). If levels != 5, adjust.
    weights = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
    if levels != 5:
        weights = weights[:levels]
        weights = weights / tf.reduce_sum(weights)

    # Combine the measurements: multiply the cs values (from scales 1..M-1) and the luminance from the final scale.
    ms_ssim_val = tf.pow(l_last, weights[-1])
    for i in range(levels - 1):
        ms_ssim_val *= tf.pow(mcs[i], weights[i])

    # The loss is 1 minus the MS-SSIM index.
    return 1.0 - ms_ssim_val


# ------------------------------------------------------------------------------
# Cycle and Reconstruction Losses
# ------------------------------------------------------------------------------

@tf.function
def cycle_loss(self, real_image, cycled_image, typ=None):
    """
    Cycle consistency loss between real and cycled images.

    Args:
        typ: Can be None, "mse", "L4", or any other value (in which case binary crossentropy is used
             on normalized images).
    """
    if typ is None:
        return MAE(self, real_image, cycled_image) * self.lambda_cycle
    elif typ == "mse":
        return MSE(self, real_image, cycled_image) * self.lambda_cycle
    elif typ == "L4":
        return L4(self, real_image, cycled_image) * self.lambda_cycle
    else:
        image_rank = len(real_image.shape)
        valid_axes = list(range(1, image_rank))
        real = min_max_norm_tf(real_image, axis=valid_axes)
        cycled = min_max_norm_tf(cycled_image, axis=valid_axes)
        loss_obj = tf.keras.losses.BinaryCrossentropy(
            from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
        return reduce_mean(self, loss_obj(real, cycled), axis=valid_axes) * self.lambda_cycle


@tf.function
def cycle_reconstruction(self, real_image, cycled_image):
    """
    Cycle reconstruction loss based on SSIM.
    """
    image_rank = len(real_image.shape)
    valid_axes = list(range(1, image_rank))
    norm_real = min_max_norm_tf(real_image, axis=valid_axes)
    norm_cycled = min_max_norm_tf(cycled_image, axis=valid_axes)
    ssim_val = ms_ssim_loss(norm_real, norm_cycled, max_val=1.0)
    return reduce_mean(self, ssim_val, axis=list(range(1, len(real_image.shape)))) * self.lambda_reconstruction


@tf.function
def cycle_seg_loss(self, real_image, cycled_image):
    """
    Segmentation loss computed via a soft clDice loss.
    """
    image_rank = len(real_image.shape)
    valid_axes = list(range(1, image_rank))
    real = min_max_norm_tf(real_image, axis=valid_axes)
    cycled = min_max_norm_tf(cycled_image, axis=valid_axes)
    cl_loss_obj = soft_dice_cldice_loss()
    return cl_loss_obj(real, cycled) * (self.lambda_topology / self.n_devices)


# ------------------------------------------------------------------------------
# Identity Loss
# ------------------------------------------------------------------------------

@tf.function
def identity_loss(self, real_image, same_image, typ=None):
    """
    Identity loss between a real image and its generated counterpart.
    """
    if typ is None:
        return self.lambda_identity * MAE(self, real_image, same_image)
    else:
        if typ == "cldice":
            real = min_max_norm_tf(real_image)
            same = min_max_norm_tf(same_image)
            loss_obj = soft_dice_cldice_loss()
            spat_loss = reduce_mean(self, loss_obj(real, same)) * self.lambda_identity
            return spat_loss


# ------------------------------------------------------------------------------
# Generator and Discriminator Losses
# ------------------------------------------------------------------------------

@tf.function
def generator_loss_fn(self, fake_image, typ=None, from_logits=True):
    """
    Generator loss function.
    """
    image_rank = len(fake_image.shape)
    valid_axes = list(range(1, image_rank))
    if typ is None:
        return MSE(self, tf.ones_like(fake_image), fake_image)
    else:
        if typ == "bce":
            loss_obj = tf.keras.losses.BinaryCrossentropy(
                from_logits=from_logits, reduction=tf.keras.losses.Reduction.NONE)
        elif typ == "bfce":
            loss_obj = tf.keras.losses.BinaryFocalCrossentropy(
                from_logits=from_logits, reduction=tf.keras.losses.Reduction.NONE)
        fake = fake_image
        if not from_logits:
            fake = min_max_norm_tf(fake, axis=valid_axes)
        loss = loss_obj(tf.ones_like(fake_image), fake)
        return reduce_mean(self, loss, axis=list(range(1, len(fake_image.shape))))


@tf.function
def discriminator_loss_fn(self, real_image, fake_image, typ=None, from_logits=True):
    """
    Discriminator loss function.
    """
    if typ is None:
        return 0.5 * (MSE(self, tf.ones_like(real_image), real_image) +
                      MSE(self, tf.zeros_like(fake_image), fake_image))
    else:
        if typ == "bce":
            loss_obj = tf.keras.losses.BinaryCrossentropy(
                from_logits=from_logits, reduction=tf.keras.losses.Reduction.NONE)
        elif typ == "bfce":
            loss_obj = tf.keras.losses.BinaryFocalCrossentropy(
                from_logits=from_logits, reduction=tf.keras.losses.Reduction.NONE)
        real = real_image
        fake = fake_image
        if not from_logits:
            real = min_max_norm_tf(real)
            fake = min_max_norm_tf(fake)
        loss = (loss_obj(tf.ones_like(real), real) + loss_obj(tf.zeros_like(fake), fake)) * 0.5
        return reduce_mean(self, loss, axis=list(range(1, len(real_image.shape))))
