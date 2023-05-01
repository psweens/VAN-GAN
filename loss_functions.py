import tensorflow as tf
import tensorflow_mri as tfmr
from utils import min_max_norm_tf
from clDice_func import soft_dice_cldice_loss


@tf.function
def reduce_mean(self, inputs, axis=None, keepdims=False):
    """ return inputs mean with respect to the global_batch_size """
    return tf.reduce_mean(inputs, axis=axis, keepdims=keepdims) / self.global_batch_size

@tf.function
def MSLE(self, real, fake):
    """ return the per sample mean squared logarithmic error """
    return reduce_mean(self, tf.square(tf.math.log(real + 1.) - tf.math.log(fake + 1.)), axis=list(range(1, len(real.shape))))

@tf.function
def MAE(self, y_true, y_pred):
  """ return the per sample mean absolute error """
  return reduce_mean(self, tf.abs(y_true - y_pred), axis=list(range(1, len(y_true.shape))))

@tf.function
def MSE(self, y_true, y_pred):
  """ return the per sample mean squared error """
  return reduce_mean(self, tf.square(y_true - y_pred), axis=list(range(1, len(y_true.shape))))

@tf.function
def L4(self, y_true, y_pred):
    return reduce_mean(self, tf.math.pow(y_true - y_pred, 4), axis=list(range(1, len(y_true.shape))))

@tf.function
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
        arr = min_max_norm_tf(arr)
    return tf.split(arr, num_or_size_splits=2, axis=axis)

@tf.function
def cycle_loss(self, real_image, cycled_image, typ=None):
    if typ is None:
        return MAE(self, real_image, cycled_image) * self.lambda_cycle
    elif typ == "mse":
        return MSE(self, real_image, cycled_image) * self.lambda_cycle
    elif typ == "L4":
        return L4(self, real_image, cycled_image) * self.lambda_cycle
    else:
        real = min_max_norm_tf(real_image)
        cycled = min_max_norm_tf(cycled_image)
        loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
        return reduce_mean(self, loss_obj(real, cycled)) * self.lambda_cycle
        
@tf.function
def cycle_perceptual(self, real_image, cycled_image):
    real = min_max_norm_tf(real_image)
    cycled = min_max_norm_tf(cycled_image)
    return reduce_mean(self, tfmr.losses.ssim_loss(real, cycled, max_val=1.0)) * self.lambda_identity
        
@tf.function
def cycle_seg_loss(self, real_image, cycled_image):
    real = min_max_norm_tf(real_image)
    cycled = min_max_norm_tf(cycled_image)
    cl_loss_obj = soft_dice_cldice_loss()
    return reduce_mean(self, cl_loss_obj(real, cycled)) * self.lambda_identity

@tf.function
def identity_loss(self, real_image, same_image, typ=None):
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
    if typ == None:
        return MSE(self, tf.ones_like(fake_image), fake_image)
    else :
        if typ == "bce":
            loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits, reduction=tf.keras.losses.Reduction.NONE)
        elif typ == "bfce":
            loss_obj = tf.keras.losses.BinaryFocalCrossentropy(from_logits=from_logits, reduction=tf.keras.losses.Reduction.NONE)
        fake = fake_image
        if from_logits == False:
            fake = min_max_norm_tf(fake)
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
