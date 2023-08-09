import os
import utils
import tensorflow as tf
from tqdm import tqdm
from generator import get_resnet_generator
from discriminator import get_discriminator
from loss_functions import (generator_loss_fn,
                            discriminator_loss_fn,
                            cycle_loss,
                            identity_loss,
                            cycle_seg_loss,
                            wasserstein_generator_loss,
                            wasserstein_discriminator_loss,
                            reduce_mean,
                            cycle_reconstruction)
from vnet_model import custom_vnet
from resunet_model import ResUNet


class VanGan:
    def __init__(
            self,
            args,
            strategy,
            lambda_cycle=10.0,
            lambda_identity=5,
            gen_i2s='resnet',
            gen_s2i='resnet',
            wasserstein=False,
            ncritic=5,
            gp_weight=10.0
    ):
        self.strategy = strategy
        self.n_devices = args.N_DEVICES
        self.img_size = args.INPUT_IMG_SIZE
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity,
        self.channels = args.CHANNELS
        self.gen_i2s_typ = gen_i2s
        self.gen_s2i_typ = gen_s2i
        self.global_batch_size = args.GLOBAL_BATCH_SIZE
        self.dims = args.DIMENSIONS
        if self.dims == 2:
            self.subvol_patch_size = (args.SUBVOL_PATCH_SIZE[0], args.SUBVOL_PATCH_SIZE[1], self.channels)
            self.seg_subvol_patch_size = (args.SUBVOL_PATCH_SIZE[0], args.SUBVOL_PATCH_SIZE[1], 1)
        else:
            self.subvol_patch_size = (
                args.SUBVOL_PATCH_SIZE[0], args.SUBVOL_PATCH_SIZE[1], args.SUBVOL_PATCH_SIZE[2], self.channels)
            self.seg_subvol_patch_size = (
                args.SUBVOL_PATCH_SIZE[0], args.SUBVOL_PATCH_SIZE[1], args.SUBVOL_PATCH_SIZE[2], 1)
        self.train_steps = args.train_steps
        self.batch_size = args.BATCH_SIZE
        self.cycle_loss_fn = cycle_loss
        self.identity_loss_fn = identity_loss
        self.wasserstein = wasserstein
        self.ncritic = ncritic
        self.icritic = 1
        self.initModel = True
        self.updateGen = True
        self.gp_weight = gp_weight
        self.wasserstein_generator_loss = wasserstein_generator_loss
        self.wasserstein_discriminator_loss = wasserstein_discriminator_loss
        self.generator_loss_fn = generator_loss_fn
        self.discriminator_loss_fn = discriminator_loss_fn
        self.identity_loss_fn = identity_loss
        self.seg_loss_fn = cycle_seg_loss
        self.reconstruction_loss = cycle_reconstruction
        self.decayed_noise_rate = 0.5
        self.current_epoch = 0
        self.layer_noise = 0.1
        self.checkpoint_loaded = False

        # create checkpoint directory
        self.checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, 'checkpoint')

        # Initialize generator & discriminator
        with self.strategy.scope():

            if self.gen_i2s_typ == 'resnet':
                self.gen_IS = get_resnet_generator(
                    input_img_size=self.subvol_patch_size,
                    batch_size=self.global_batch_size,
                    name='generator_IS',
                    num_downsampling_blocks=3,
                    num_upsample_blocks=3
                )
            elif self.gen_i2s_typ == 'vnet':
                self.gen_IS = custom_vnet(
                    input_shape=self.subvol_patch_size,
                    activation='relu',
                    use_batch_norm=False,
                    upsample_mode='upsample',
                    dropout=0.5,
                    dropout_change_per_layer=0.0,
                    dropout_type='spatial',
                    use_dropout_on_upsampling=False,
                    use_attention_gate=False,
                    filters=32,
                    num_layers=4,
                    output_activation='tanh',
                )
            elif self.gen_i2s_typ == 'resUnet':
                self.gen_IS = ResUNet(
                    input_shape=self.subvol_patch_size,
                    upsample_mode='simple',
                    dropout=0.1,
                    dropout_change_per_layer=0.1,
                    dropout_type='none',
                    use_attention_gate=False,
                    filters=16,
                    num_layers=4,
                    # output_activation=None,
                )
            else:
                raise ValueError('AB Generator type not recognised')

            if self.gen_s2i_typ == 'resnet':
                self.gen_SI = get_resnet_generator(
                    input_img_size=self.subvol_patch_size,
                    batch_size=self.global_batch_size,
                    name='generator_SI',
                    num_downsampling_blocks=3,
                    num_upsample_blocks=3
                )
            elif self.gen_s2i_typ == 'vnet':
                self.gen_SI = custom_vnet(
                    input_shape=self.subvol_patch_size,
                    activation='relu',
                    use_batch_norm=True,
                    upsample_mode='deconv',
                    dropout=0.5,
                    dropout_change_per_layer=0.0,
                    dropout_type='spatial',
                    use_dropout_on_upsampling=False,
                    use_attention_gate=False,
                    filters=16,
                    num_layers=4,
                    output_activation='tanh',
                    addnoise=False
                )
            elif self.gen_s2i_typ == 'resUnet':
                self.gen_SI = ResUNet(
                    input_shape=self.seg_subvol_patch_size,
                    upsample_mode='simple',
                    dropout=0.1,
                    dropout_change_per_layer=0.1,
                    dropout_type='none',
                    use_attention_gate=False,
                    filters=16,
                    num_layers=4,
                    # output_activation=None,
                    use_input_noise=False
                )
            else:
                raise ValueError('BA Generator type not recognised')

            # Get the discriminators
            self.disc_I = get_discriminator(
                input_img_size=self.subvol_patch_size,
                batch_size=self.global_batch_size,
                name='discriminator_I',
                filters=64,
                use_dropout=False,
                wasserstein=self.wasserstein,
                use_SN=False,
                use_input_noise=True,
                use_layer_noise=True,
                noise_std=self.layer_noise
            )
            self.disc_S = get_discriminator(
                input_img_size=self.seg_subvol_patch_size,
                batch_size=self.global_batch_size,
                name='discriminator_S',
                filters=64,
                use_dropout=False,
                wasserstein=self.wasserstein,
                use_SN=False,
                use_input_noise=True,
                use_layer_noise=True,
                noise_std=self.layer_noise
            )

            # Initialise optimizers
            if self.wasserstein:

                self.gen_I_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.,
                                                                beta_2=0.9)  # , clipnorm=10.0)
                self.gen_S_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.,
                                                                beta_2=0.9)  # , clipnorm=10.0)
                self.disc_I_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.,
                                                                 beta_2=0.9)  # , clipnorm=10.0)
                self.disc_S_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.,
                                                                 beta_2=0.9)  # , clipnorm=10.0)

            else:
                # Initialise decay rates
                self.dI_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                    2e-4,
                    decay_steps=5 * self.train_steps,
                    decay_rate=0.98,
                    staircase=False)

                self.dS_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                    2e-4,
                    decay_steps=5 * self.train_steps,
                    decay_rate=0.98,
                    staircase=False)

                self.gen_I_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4,
                                                                beta_1=0.5,
                                                                beta_2=0.9,
                                                                clipnorm=100)
                self.gen_S_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4,
                                                                beta_1=0.5,
                                                                beta_2=0.9,
                                                                clipnorm=100)
                self.disc_I_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4,
                                                                 beta_1=0.5,
                                                                 beta_2=0.9,
                                                                 clipnorm=100)
                self.disc_S_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4,
                                                                 beta_1=0.5,
                                                                 beta_2=0.9,
                                                                 clipnorm=100)

            # Initialise checkpoint
            self.checkpoint = tf.train.Checkpoint(gen_AB=self.gen_IS,
                                                  gen_BAF=self.gen_SI,
                                                  disc_A=self.disc_I,
                                                  disc_B=self.disc_S,
                                                  gen_A_optimizer=self.gen_I_optimizer,
                                                  gen_B_optimizer=self.gen_S_optimizer,
                                                  disc_A_optimizer=self.disc_I_optimizer,
                                                  disc_B_optimizer=self.disc_S_optimizer)

    def save_checkpoint(self, epoch):
        """ save checkpoint to checkpoint_dir, overwrite if exists """
        self.checkpoint.write(self.checkpoint_prefix + "_e{epoch}".format(epoch=epoch + 1))
        print(f'\nSaved checkpoint to {self.checkpoint_prefix}\n')

    def load_checkpoint(self, epoch=None, expect_partial: bool = False, newpath=None):
        """ load checkpoint from checkpoint_dir if exists """
        if newpath is not None:
            self.checkpoint_prefix = os.path.join(newpath, 'checkpoint')
        checkpoint_path = self.checkpoint_prefix + "_e{epoch}".format(epoch=epoch)
        if os.path.exists(f'{os.path.join(checkpoint_path)}.index'):
            self.checkpoint_loaded = True
            with self.strategy.scope():
                if expect_partial:
                    self.checkpoint.read(checkpoint_path).expect_partial()
                else:
                    self.checkpoint.read(checkpoint_path)
            print(f'\nLoaded checkpoint from {checkpoint_path}\n')
        else:
            print('Error: Checkpoint not found!')

    def compute_losses(self, real_I, real_S, result, training=True):
        """
        Computes the losses for the VANGAN model using the given input images and model settings.

        Args:
            real_I (tf.Tensor): A tensor containing the real images from the imaging domain.
            real_S (tf.Tensor): A tensor containing the real images from the segmentation domain.
            result (dict): A dictionary to store the loss values.
            training (bool, optional): A flag indicating whether the model is being trained or not. Defaults to True.

        Returns:
            tuple: A tuple containing the updated result dictionary and the calculated losses.

        Raises:
            ValueError: If the `cycle_loss_fn`, `seg_loss_fn`, `reconstruction_loss`, `discriminator_loss_fn`,
            `generator_loss_fn`, `wasserstein_discriminator_loss` or `wasserstein_generator_loss` are not
            callable functions.

        """

        # Can be used to debug dataset numerics
        # tf.debugging.check_numerics(real_I, 'real_I failure')
        # tf.debugging.check_numerics(real_S, 'real_S failure')

        # A -> B
        fake_S = self.gen_IS(real_I, training=training)
        # B -> A
        fake_I = self.gen_SI(real_S, training=training)

        # Cycle loss
        cycled_S = self.gen_IS(fake_I, training=training)

        cycle_loss_I = self.cycle_loss_fn(self, real_S, cycled_S, typ="bce")

        seg_loss = self.seg_loss_fn(self, real_S, cycled_S)
        cycled_I = self.gen_SI(fake_S, training=training)
        cycle_loss_S = self.cycle_loss_fn(self, real_I, cycled_I, typ='L2')

        reconstruction_loss_I = self.reconstruction_loss(self, real_I, cycled_I)

        # Identity mapping
        # id_SI_loss = self.identity_loss_fn(self, real_I, self.gen_SI(real_I, training=True))
        # id_IS_loss = self.identity_loss_fn(self, real_S, self.gen_IS(real_S, training=True), typ='cldice')

        # Discriminator outputs         
        disc_real_S = self.disc_S(real_S, training=training)
        disc_fake_S = self.disc_S(fake_S, training=training)

        disc_real_I = self.disc_I(real_I, training=training)
        disc_fake_I = self.disc_I(fake_I, training=training)

        # Generator & discriminator loss
        if self.wasserstein:
            gen_IS_loss = self.wasserstein_generator_loss(self, disc_fake_S)
            gen_SI_loss = self.wasserstein_generator_loss(self, disc_fake_I)
            disc_I_loss = self.wasserstein_discriminator_loss(self, disc_real_I, disc_fake_I)
            disc_S_loss = self.wasserstein_discriminator_loss(self, disc_real_S, disc_fake_S)

        else:
            gen_IS_loss = self.generator_loss_fn(self, disc_fake_S, from_logits=True)
            gen_SI_loss = self.generator_loss_fn(self, disc_fake_I, from_logits=True)
            disc_I_loss = self.discriminator_loss_fn(self, disc_real_I, disc_fake_I, from_logits=True)
            disc_S_loss = self.discriminator_loss_fn(self, disc_real_S, disc_fake_S, from_logits=True)

        # Total generator loss
        total_loss_I = gen_IS_loss + cycle_loss_I + seg_loss  # + id_SI_loss
        total_loss_S = gen_SI_loss + cycle_loss_S + reconstruction_loss_I  # + id_IS_loss

        result.update({
            'total_IS_loss': total_loss_I,
            'total_SI_loss': total_loss_S,
            'D_I_loss': disc_I_loss,
            'D_S_loss': disc_S_loss,
            'gen_IS_loss': gen_IS_loss,
            'gen_SI_loss': gen_SI_loss,
            'cycle_gen_IS_loss': cycle_loss_I,
            'cycle_gen_SI_loss': cycle_loss_S,
            'seg_loss': seg_loss,
            'reconstruction_loss_I': reconstruction_loss_I,
            # 'identity_IS': id_IS_loss,
            # 'identity_SI': id_SI_loss
        })

        return result, total_loss_I, total_loss_S, disc_I_loss, disc_S_loss, fake_I, fake_S

    def gradient_penalty(self, real, fake, descrip='I'):
        """
        Computes the gradient penalty for the Wasserstein loss function. 

        Parameters:
        - real: the real input data (either A or B) with dimensions [batch_size, height, width, channels]
        - fake: the generated data (either A or B) with dimensions [batch_size, height, width, channels]
        - descrip: specifies which discriminator to use (either 'I' or 'S')

        Returns:
        - gp: the computed gradient penalty
        """
        alpha = tf.random.normal([self.batch_size, 1, 1, 1, 1], 0.0, 1.0)
        diff = fake - real
        interpolated = real + alpha * diff
        if descrip == 'I':
            pred = self.disc_I(interpolated, training=True)
        else:
            pred = self.disc_S(interpolated, training=True)
        grads = tf.gradients(pred, interpolated)[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads),
                                     axis=[1, 2, 3, 4]) + 1.e-12)  # small constant add to prevent division by zero
        gp = reduce_mean(self, (norm - 1.0) ** 2)
        return gp

    def train_step(self, real_I, real_S):
        """
        Trains the VANGAN model using a single batch of input images.

        Parameters:
        - `self`: the VANGAN object.
        - `real_I`: a batch of images from the imaging domain.
        - `real_S`: a batch of images from the segmentation domain.

        Returns:
        - `result`: a dictionary containing the losses and metrics computed during training.
        """

        result = {}
        with tf.GradientTape(persistent=True) as tape:
            result, total_loss_I, total_loss_S, disc_I_loss, disc_S_loss, fake_I, fake_S = self.compute_losses(real_I,
                                                                                                               real_S,
                                                                                                               result,
                                                                                                               training=True)

        if self.wasserstein:

            if self.updateGen:
                self.gen_I_optimizer.minimize(loss=total_loss_I,
                                              var_list=self.gen_IS.trainable_variables,
                                              tape=tape)
                self.gen_S_optimizer.minimize(loss=total_loss_S,
                                              var_list=self.gen_SI.trainable_variables,
                                              tape=tape)

            if not self.initModel:
                gp = self.gradient_penalty(real_I, fake_I, descrip='A')
                disc_I_loss = disc_I_loss + gp * self.gp_weight

                gp = self.gradient_penalty(real_S, fake_S, descrip='B')
                disc_S_loss = disc_S_loss + gp * self.gp_weight

            # clipping weights of discriminators as told in the
            # WasserteinGAN paper to enforce Lipschitz constraint.
            # clip_values = [-0.01, 0.01]
            # self.clip_discriminator_A_var_op = [var.assign(tf.clip_by_value(var, clip_values[0], clip_values[1])) for
            #    var in self.disc_I.trainable_variables]
            # self.clip_discriminator_B_var_op = [var.assign(tf.clip_by_value(var, clip_values[0], clip_values[1])) for
            #    var in self.disc_S.trainable_variables]

        else:
            self.gen_I_optimizer.minimize(loss=total_loss_I,
                                          var_list=self.gen_IS.trainable_variables,
                                          tape=tape)
            self.gen_S_optimizer.minimize(loss=total_loss_S,
                                          var_list=self.gen_SI.trainable_variables,
                                          tape=tape)

        self.disc_I_optimizer.minimize(loss=disc_I_loss,
                                       var_list=self.disc_I.trainable_variables,
                                       tape=tape)
        self.disc_S_optimizer.minimize(loss=disc_S_loss,
                                       var_list=self.disc_S.trainable_variables,
                                       tape=tape)

        return result

    def test_step(self, real_I, real_S):
        """
        Evaluates the VANGAN model on a single batch of input images.

        Parameters:
        - `self`: the VANGAN object.
        - `real_I`: a batch of images from the imaging domain.
        - `real_S`: a batch of images from the segmentation domain.

        Returns:
        - `result`: a dictionary containing the losses and metrics computed during evaluation.
        """

        result = {}
        result, _, _, _, _, _, _ = self.compute_losses(real_I, real_S, result, training=False)
        return result

    def reduce_dict(self, d: dict):
        """
        Reduces the values in a dictionary using the current distribution strategy.

        Parameters:
        - `self`: the VANGAN object.
        - `d`: a dictionary containing values to be reduced.

        Returns:
        - None
        """

        ''' reduce items in dictionary d '''
        for k, v in d.items():
            d[k] = self.strategy.reduce(tf.distribute.ReduceOp.SUM, v, axis=None)

    @tf.function
    def distributed_train_step(self, x, y):
        """
        Runs a training step using the current distribution strategy.

        Parameters:
        - `self`: the VANGAN object.
        - `x`: a batch of images from the imaging domain.
        - `y`: a batch of images from the segmentation domain.

        Returns:
        - `results`: a dictionary containing the losses and metrics computed during training.
        """
        results = self.strategy.run(self.train_step, args=(x, y))
        self.reduce_dict(results)
        return results

    @tf.function
    def distributed_test_step(self, x, y):
        """
        Runs a test step using the current distribution strategy.

        Parameters:
        - `self`: the VANGAN object.
        - `x`: a batch of images from the imaging domain.
        - `y`: a batch of images from the segmentation domain.

        Returns:
        - `results`: a dictionary containing the losses and metrics computed during testing.
        """
        results = self.strategy.run(self.test_step, args=(x, y))
        self.reduce_dict(results)
        return results


def train(ds, gan, summary, epoch: int, steps=None, desc=None, training=True):
    """
    Runs a training or testing loop for a given number of steps using the specified VANGAN object and dataset.

    Parameters:
    - `args`: command line arguments.
    - `ds`: a TensorFlow dataset containing the input data.
    - `gan`: a VANGAN object representing the model.
    - `summary`: a TensorFlow summary object for logging.
    - `epoch`: the current epoch number.
    - `steps`: the number of steps to run (default is None, meaning run until the end of the dataset).
    - `desc`: a string to use as the description for the tqdm progress bar (default is None).
    - `training`: a boolean indicating whether to run in training mode (default is True).

    Returns:
    - `results`: a dictionary containing the losses and metrics computed during training or testing.
    """
    results = {}
    cntr = 0
    for x, y in tqdm(ds, desc=desc, total=steps, disable=0):
        if cntr == steps:
            break
        else:
            cntr += 1
        if training:
            if gan.icritic % gan.ncritic == 0:
                gan.updateGen = True
                gan.icritic = 1
            else:
                gan.icritic += 1
            result = gan.distributed_train_step(x, y)
        else:
            result = gan.distributed_test_step(x, y)
        utils.append_dict(results, result)
        gan.updateGen = False
        gan.initModel = False

    for key, value in results.items():
        summary.scalar(key, tf.reduce_mean(value), epoch=epoch, training=training)

    return results
