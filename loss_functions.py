import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from clDice_func import soft_dice_cldice_loss
import tensorflow_addons as tfa
import tensorflow_mri as tfmr

def get_shape(arr):
    res = []
    while isinstance((arr), list):
        res.append(len(arr))
        arr = arr[0]
    return res

def rescaleTensor(arr, alpha=None, beta=None):
    # alpha=-0.5, beta=0.5: [0,1] to [-1,1]
    # alpha=1.0, beta=2.0: [-1,1] to [0,1]
    arr2 = tf.math.divide_no_nan(
            tf.math.add(
                arr,
                tf.constant(
                    alpha,
                    dtype=tf.float32,
                    shape=get_shape(arr)
                    )
                ),
            tf.constant(
                beta,
                    dtype=tf.float32,
                    shape=get_shape(arr)
                )
            )
    
    return arr2

def normaliseTF(arr):
    tensor = tf.math.divide_no_nan(
               tf.math.subtract(
                  arr, 
                  tf.math.reduce_min(arr)
               ), 
               tf.math.subtract(
                  tf.math.reduce_max(arr), 
                  tf.math.reduce_min(arr)
               )
            )
    return tensor

def replacenan(t):
    return tf.where(tf.math.is_nan(t), tf.zeros_like(t), t)

def reduce_mean(self, inputs, axis=None, keepdims=False):
    """ return inputs mean with respect to the global_batch_size """
    return tf.reduce_mean(inputs, axis=axis, keepdims=keepdims) / self.global_batch_size

def MSLE(self, real, fake):
    """ return the per sample mean squared logarithmic error """
    return reduce_mean(self, tf.square(tf.math.log(real + 1.) - tf.math.log(fake + 1.)), axis=list(range(1, len(real.shape))))

def MAE(self, y_true, y_pred):
  """ return the per sample mean absolute error """
  return reduce_mean(self, tf.abs(y_true - y_pred), axis=list(range(1, len(y_true.shape))))

def MSE(self, y_true, y_pred):
  """ return the per sample mean squared error """
  return reduce_mean(self, tf.square(y_true - y_pred), axis=list(range(1, len(y_true.shape))))

def wasserstein_loss(prob_real_is_real, prob_fake_is_real):
    return tf.reduce_mean(prob_real_is_real - prob_fake_is_real)

@tf.function
def matched_crop(self, stack, axis=None, rescale=False):
    if axis==1:
        shape = (self.batch_size, 2*self.img_size[1], self.img_size[2], 1, self.channels)
        raxis = 3
    elif axis==3:
        shape = (self.batch_size, 1, self.img_size[2], 2*self.img_size[3], self.channels)
        raxis = 1
        axis -= 1

    arr = tf.squeeze(tf.image.random_crop(stack, size=shape), 
                     axis=raxis)
    if rescale:
        arr = normaliseTF(arr)
    return tf.split(arr, num_or_size_splits=2, axis=axis)

@tf.function
def ssim3D(self, real, fake, n=5):
    
    lossA = tf.constant(0.0)
    arr = tf.concat([real, fake], axis=1)      
    for _ in range(n):
        creal, cfake = matched_crop(self, arr, axis=1, rescale=True)
        lossA = tf.math.add(lossA,tf.clip_by_value(tf.image.ssim(creal, cfake, max_val=1.0, filter_size=6), 0., 1.))
                        
    lossB = tf.constant(0.0)
    arr = tf.concat([real, fake], axis=3)  
    for _ in range(n):
        creal, cfake = matched_crop(self, arr, axis=3, rescale=True)
        lossB = tf.math.add(lossB, tf.clip_by_value(tf.image.ssim(creal, cfake, max_val=1.0, filter_sigma=0.5, filter_size=6), 0., 1.))

    return reduce_mean(self, 1. - (0.5 / float(n)) * (lossA + lossA))

@tf.function
def cycle_loss(self, real_image, cycled_image, typ=None):
    if typ is None:
        return MAE(self, real_image, cycled_image) * self.lambda_cycle
    elif typ == "mse":
        return MSE(self, real_image, cycled_image) * self.lambda_cycle
    else:
        real = normaliseTF(real_image)
        cycled = normaliseTF(cycled_image)
        loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
        return reduce_mean(self, loss_obj(real, cycled)) * self.lambda_cycle
        
        
def cycle_perceptual(self, real_image, cycled_image):
    real = normaliseTF(real_image)
    cycled = normaliseTF(cycled_image)
    # loss_obj = tfmr.StructuralSimilarityLoss(max_val=1.0, reduction='none')
    # ssim = reduce_mean(self, 
    #                    tf.clip_by_value(1. - tfmr.ssim3d(real, cycled, max_val=1.), 0. , 1.)
                       # ) 
    return reduce_mean(self, tfmr.losses.ssim_loss(real, cycled, max_val=1.0)) * self.lambda_identity
        
@tf.function
def cycle_seg_loss(self, real_image, cycled_image):
    real = normaliseTF(real_image)
    cycled = normaliseTF(cycled_image)
    cl_loss_obj = soft_dice_cldice_loss()
    return reduce_mean(self, cl_loss_obj(real, cycled)) * self.lambda_identity

@tf.function
def identity_loss(self, real_image, same_image, typ=None):
    if typ is None:
        return self.lambda_identity * MAE(self, real_image, same_image)
    else:
        if typ == "cldice":
            real = normaliseTF(real_image)
            same = normaliseTF(same_image)
            loss_obj = soft_dice_cldice_loss()
            # bf_loss_obj = tf.keras.losses.BinaryFocalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
            # id_loss = reduce_mean(self, bf_loss_obj(real, same_image)) * self.lambda_identity
            spat_loss = reduce_mean(self, loss_obj(real, same)) * self.lambda_identity
            return spat_loss

@tf.function
def generator_loss_fn(self, fake_image, typ=None, from_logits=True):
    if typ == None:
        return MSE(self, tf.ones_like(fake_image), fake_image)
    else :
        if typ == "bce":
            loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits, reduction=tf.keras.losses.Reduction.NONE)
        elif typ == "bfce":
            loss_obj = tf.keras.losses.BinaryFocalCrossentropy(from_logits=from_logits, reduction=tf.keras.losses.Reduction.NONE)
        fake = fake_image
        if from_logits == False:
            fake = normaliseTF(fake)
        loss = loss_obj(tf.ones_like(fake_image), fake)
        return reduce_mean(self, loss)

@tf.function
def discriminator_loss_fn(self, real_image, fake_image, typ=None, from_logits=True):
    if typ == None:
        return 0.5 * (MSE(self, tf.ones_like(real_image), real_image) + MSE(self, tf.zeros_like(fake_image), fake_image))
    else :
        if typ == "bce":
            loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits, reduction=tf.keras.losses.Reduction.NONE)
        elif typ == "bfce":
            loss_obj = tf.keras.losses.BinaryFocalCrossentropy(from_logits=from_logits, reduction=tf.keras.losses.Reduction.NONE)
        real = real_image
        fake = fake_image
        if from_logits == False:
            real = normaliseTF(real)
            fake = normaliseTF(fake)
        loss = (loss_obj(tf.ones_like(real), real) + loss_obj(tf.zeros_like(fake), fake)) * 0.5
        return reduce_mean(self, loss)

def full_cycle_loss(self, real_A, real_B, cycled_A, cycled_B):
    return tf.math.add(tf.math.add(tf.math.add(
        self.cycle_loss_fn(self, real_A, cycled_A, typ='mse'),
        self.cycle_loss_fn(self, real_B, cycled_B, typ="bce")),
        self.perceptual_loss(self, real_A, cycled_A)),
        self.seg_loss_fn(self, real_B, cycled_B))
    
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

# From https://github.com/imransalam/style-transfer-tensorflow-2.0/blob/4af86e24d85000e30373e22bd20781cc653d06f2/synthesis.py#L20
def hist_match(source, template):
    shape = tf.shape(source)
    source = K.flatten(source)
    template = K.flatten(template)
    hist_bins = 255
    if tf.math.is_nan(tf.reduce_max(source)) or tf.math.is_nan(tf.reduce_min(source)) or tf.math.is_nan(tf.reduce_max(template)) or tf.math.is_nan(tf.reduce_min(template)):
        return tf.reshape(template, shape)
    else:
        max_value = tf.reduce_max([tf.reduce_max(source), tf.reduce_max(template)])
        min_value = tf.reduce_min([tf.reduce_min(source), tf.reduce_min(template)])
        hist_delta = (max_value - min_value)/hist_bins
        hist_range = tf.range(min_value, max_value, hist_delta)
        hist_range = tf.add(hist_range, tf.divide(hist_delta, 2))
        s_hist = tf.histogram_fixed_width(source, 
                                            [min_value, max_value],
                                             nbins = hist_bins, 
                                            dtype = tf.int64
                                            )
        t_hist = tf.histogram_fixed_width(template, 
                                             [min_value, max_value],
                                             nbins = hist_bins, 
                                            dtype = tf.int64
                                            )
        s_quantiles = tf.cumsum(s_hist)
        s_last_element = tf.subtract(tf.size(s_quantiles), tf.constant(1))
        s_quantiles = tf.divide(s_quantiles, tf.gather(s_quantiles, s_last_element)) # tf.gather prints 'unknown shape' warning
    
        t_quantiles = tf.cumsum(t_hist)
        t_last_element = tf.subtract(tf.size(t_quantiles), tf.constant(1))
        t_quantiles = tf.divide(t_quantiles, tf.gather(t_quantiles, t_last_element))
    
    
        nearest_indices = tf.map_fn(lambda x: tf.argmin(tf.abs(tf.subtract(t_quantiles, x))), 
                                      s_quantiles, fn_output_signature = tf.int64)
    
        s_bin_index = tf.cast(tf.divide(source, hist_delta), dtype=tf.int64)
    
        s_bin_index = tf.clip_by_value(s_bin_index, 0, 254)
        matched_to_t = tf.gather(hist_range, tf.gather(nearest_indices, s_bin_index))
        return tf.reshape(matched_to_t, shape)
#     shape = tf.shape(source)
#     sourceFlat = tf.reshape(source, [-1])
#     templateFlat = tf.reshape(template, [-1])
#     max_value = tf.reduce_max([tf.reduce_max(sourceFlat), tf.reduce_max(templateFlat)])
#     min_value = tf.reduce_min([tf.reduce_min(sourceFlat), tf.reduce_min(templateFlat)])
    
#     if tf.math.is_nan(max_value) or tf.math.is_nan(min_value):
#         return template
#     else:
    
#         s_hist, hist_delta, hist_range = compute_hist(sourceFlat, min_value=min_value, max_value=max_value)
#         t_hist, _, _ = compute_hist(sourceFlat, min_value=min_value, max_value=max_value)
        
#         s_quantiles = tf.cumsum(s_hist)
#         s_last_element = tf.subtract(tf.size(s_quantiles), tf.constant(1))
#         s_quantiles = tf.math.divide_no_nan(s_quantiles, tf.gather(s_quantiles, s_last_element))
    
#         t_quantiles = tf.cumsum(t_hist)
#         t_last_element = tf.subtract(tf.size(t_quantiles), tf.constant(1))
#         t_quantiles = tf.math.divide_no_nan(t_quantiles, tf.gather(t_quantiles, t_last_element))
    
    
#         nearest_indices = tf.map_fn(lambda x: tf.argmin(tf.abs(tf.subtract(t_quantiles, x))), 
#                                       s_quantiles, fn_output_signature = tf.int64)
    
#         s_bin_index = tf.cast(tf.math.divide_no_nan(sourceFlat, hist_delta), dtype=tf.int64)
#         s_bin_index = tf.clip_by_value(s_bin_index, 0, 254)
    
#         matched_to_t = tf.gather(hist_range, tf.gather(nearest_indices, s_bin_index))
#         #return rescaleTensor(tf.reshape(matched_to_t, shape), alpha=-0.5, beta=0.5)
#         return tf.reshape(matched_to_t, shape)

# def compute_hist(data, hist_bins=255, min_value=0., max_value=1.):
#     hist_delta = (max_value - min_value)/hist_bins
#     hist_range = tf.range(min_value, max_value, hist_delta)
#     hist_range = tf.add(hist_range, tf.math.divide_no_nan(hist_delta, 2))
#     hist = tf.histogram_fixed_width(data, 
#                                      [min_value, max_value],
#                                      nbins = hist_bins, 
#                                      dtype = tf.float32
#                                      )
#     return hist, hist_delta, hist_range

def get_content_loss(self, target, fake_image):
    # tf.reduce_mean(tf.square(content - target))
    real = normaliseTF(target)
    fake = normaliseTF(fake_image)
    content = hist_match(real, fake)
    return MAE(self, real, content)
