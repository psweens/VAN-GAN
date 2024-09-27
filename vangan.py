import os
import utils
import tensorflow as tf
from tqdm import tqdm
from resnet_model import resnet
from discriminator import get_discriminator
from loss_functions import (generator_loss_fn,
                            discriminator_loss_fn,
                            cycle_loss,
                            identity_loss,
                            cycle_seg_loss,
                            cycle_reconstruction)
from vnet_model import custom_vnet
from res_unet_model import res_unet


class VanGan:
    def __init__(
            self,
            args,
            strategy,
            lambda_cycle=20.0,
            lambda_identity=10.,
            lambda_reconstruction=5.,
            lambda_topology=5.,
            gen_i2s='default',
            gen_s2i='default',
            semi_supervised=False,
            joint_cycle=tf.Variable(0., trainable=False, dtype=tf.float32, name="shared_cycle")
    ):
        self.strategy = strategy
        self.train_steps = args.train_steps
        self.n_devices = args.N_DEVICES
        self.img_size = args.INPUT_IMG_SIZE
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = tf.Variable(lambda_identity, trainable=False, dtype=tf.float32, name="lambda_identity")
        self.lambda_reconstruction = tf.Variable(lambda_reconstruction, trainable=False, dtype=tf.float32, name="lambda_reconstruction")
        self.lambda_topology = tf.Variable(lambda_topology, trainable=False, dtype=tf.float32, name="lambda_topology")
        self.channels = args.CHANNELS
        self.gen_i2s_typ = gen_i2s
        self.gen_s2i_typ = gen_s2i
        self.semi_supervised = semi_supervised
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
        self.generator_loss_fn = generator_loss_fn
        self.discriminator_loss_fn = discriminator_loss_fn
        self.identity_loss_fn = identity_loss
        self.seg_loss_fn = cycle_seg_loss
        self.reconstruction_loss = cycle_reconstruction
        self.decayed_noise_rate = 0.5
        self.current_epoch = tf.Variable(0, trainable=False, dtype=tf.int64, name="current_epoch")
        self.layer_noise = 0.1
        self.checkpoint_loaded = False
        self.shared_cycle = joint_cycle
        self.identity_off = tf.Variable(0., trainable=False, dtype=tf.float32, name="identity_off")

        # create checkpoint directory
        self.checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, 'checkpoint')

        # Initialize generator & discriminator
        with self.strategy.scope():

            # Default generator architecture = residual U-net
            if self.gen_i2s_typ == 'default':
                self.gen_IS = res_unet(
                    input_shape=self.subvol_patch_size,
                    upsample_mode='simple',
                    dropout=0.,
                    dropout_change_per_layer=0.,
                    dropout_type='none',
                    use_attention_gate=False,
                    filters=16,
                    num_layers=4,
                    dim=self.dims
                )

            if self.gen_s2i_typ == 'default':
                self.gen_SI = res_unet(
                    input_shape=self.seg_subvol_patch_size,
                    upsample_mode='simple',
                    dropout=0.,
                    dropout_change_per_layer=0.,
                    dropout_type='none',
                    use_attention_gate=False,
                    filters=16,
                    num_layers=4,
                    use_input_noise=False,
                    output_activation='tanh',
                    dim=self.dims
                )

            # Get the discriminators
            self.disc_I = get_discriminator(
                input_img_size=self.subvol_patch_size,
                batch_size=self.global_batch_size,
                name='discriminator_I',
                filters=64,
                use_dropout=True,
                dropout_rate=0.2,
                use_SN=False,
                use_input_noise=True,
                use_layer_noise=True,
                use_standardisation=False,
                noise_std=self.layer_noise,
                dim=self.dims
            )

            self.disc_S = get_discriminator(
                input_img_size=self.seg_subvol_patch_size,
                batch_size=self.global_batch_size,
                name='discriminator_S',
                filters=64,
                use_dropout=True,
                dropout_rate=0.2,
                use_SN=False,
                use_input_noise=True,
                use_layer_noise=True,
                noise_std=self.layer_noise,
                dim=self.dims
            )

            # Initialise optimizers
            self.gen_I_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4,
                                                            beta_1=0.5,
                                                            beta_2=0.9,
                                                            clipnorm=10)
            self.gen_S_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4,
                                                            beta_1=0.5,
                                                            beta_2=0.9,
                                                            clipnorm=1)
            self.disc_I_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4,
                                                             beta_1=0.5,
                                                             beta_2=0.9,
                                                             clipnorm=1)
            self.disc_S_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4,
                                                             beta_1=0.5,
                                                             beta_2=0.9,
                                                             clipnorm=1)

            # Initialise checkpoint
            self.checkpoint = tf.train.Checkpoint(gen_IS=self.gen_IS,
                                                  gen_SI=self.gen_SI,
                                                  disc_I=self.disc_I,
                                                  disc_S=self.disc_S,
                                                  gen_I_optimizer=self.gen_I_optimizer,
                                                  gen_S_optimizer=self.gen_S_optimizer,
                                                  disc_I_optimizer=self.disc_I_optimizer,
                                                  disc_S_optimizer=self.disc_S_optimizer)

    def save_checkpoint(self, epoch):
        """ save checkpoint to checkpoint_dir, overwrite if exists """
        self.checkpoint.write(self.checkpoint_prefix + "_e{epoch}".format(epoch=epoch + 1))
        print(f'\nSaved checkpoint to {self.checkpoint_prefix}\n')

    def load_checkpoint(self, epoch=None, expect_partial: bool = False, newpath=None):
        """ load checkpoint from checkpoint_dir if exists """
        if newpath is not None:
            self.checkpoint_prefix = os.path.join(newpath, 'checkpoint')
        checkpoint_path = self.checkpoint_prefix + "_e{epoch}".format(epoch=epoch)

        print(f"Trying to load checkpoint from path: {checkpoint_path}")
        checkpoint_files = [f'{checkpoint_path}.index', f'{checkpoint_path}.data-00000-of-00001']
        if all(os.path.exists(file) for file in checkpoint_files):
            if expect_partial:
                self.checkpoint.restore(checkpoint_path).expect_partial()
            else:
                self.checkpoint.restore(checkpoint_path)
            print(f'Loaded checkpoint from {checkpoint_path}\n')

        else:
            print('Error: Checkpoint not found!')

    def apply_seg_identity_loss(self, real_S, training):
        return tf.cond(tf.equal(self.identity_off, 0.0),
                       lambda: self.identity_loss_fn(self, real_S, self.gen_IS(real_S, training=training), typ='cldice'),
                       lambda: 0.0)

    def apply_imaging_identity_loss(self, real_I, training):
        return tf.cond(tf.equal(self.identity_off, 0.0),
                       lambda: self.identity_loss_fn(self, real_I, self.gen_SI(real_I, training=training)),
                       lambda: 0.0)

    def shared_cycle_loss(self, cycle_loss, vg_cycle_loss):
        return tf.cond(tf.equal(self.shared_cycle, 1.0), lambda: cycle_loss + vg_cycle_loss, lambda: 0.0)

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
            `generator_loss_fn` are not callable functions.

        """

        # Can be used to debug dataset numerics
        # tf.debugging.check_numerics(real_I, 'real_I failure')
        # tf.debugging.check_numerics(real_S, 'real_S failure')

        # if self.semi_supervised:
        #     real_S, real_sim = tf.split(real_S, num_or_size_splits=2, axis=3)
        #     tf.debugging.check_numerics(real_S, 'real_S failure')
        #     tf.debugging.check_numerics(real_sim, 'real_sim failure')
        #     fake_sim_seg = self.gen_IS(real_sim, training=training)
        #     tf.debugging.check_numerics(fake_sim_seg, 'fake_sim_seg failure')
        #     semi_seg_loss = self.seg_loss_fn(self, real_S, fake_sim_seg)

        # I -> S
        fake_S = self.gen_IS(real_I, training=training)

        # S -> I
        fake_I = self.gen_SI(real_S, training=training)

        # Cycle losses
        cycled_S = self.gen_IS(fake_I, training=training)
        cycled_I = self.gen_SI(fake_S, training=training)
        cycle_loss_ISI = self.cycle_loss_fn(self, real_I, cycled_I, typ='mse')
        cycle_loss_SIS = self.cycle_loss_fn(self, real_S, cycled_S, typ="bce")
        seg_loss = self.seg_loss_fn(self, real_S, cycled_S)
        reconstruction_loss = self.reconstruction_loss(self, real_I, cycled_I)

        # Discriminator outputs         
        disc_real_S = self.disc_S(real_S, training=training)
        disc_fake_S = self.disc_S(fake_S, training=training)

        disc_real_I = self.disc_I(real_I, training=training)
        disc_fake_I = self.disc_I(fake_I, training=training)

        # Generator & discriminator loss
        gen_IS_loss = self.generator_loss_fn(self, disc_fake_S, from_logits=True)
        gen_SI_loss = self.generator_loss_fn(self, disc_fake_I, from_logits=True)
        disc_I_loss = self.discriminator_loss_fn(self, disc_real_I, disc_fake_I, from_logits=True)
        disc_S_loss = self.discriminator_loss_fn(self, disc_real_S, disc_fake_S, from_logits=True)

        # Total generator loss
        # if self.semi_supervised:
        #     total_loss_I = gen_IS_loss + cycle_loss_SIS + seg_loss + semi_seg_loss
        #     result.update({'semi_supervised_loss_IS': semi_seg_loss})
        #     result.update({'combined_dice_score': 1. - (semi_seg_loss / 10.)})
        # else:
        total_loss_I = (gen_IS_loss + cycle_loss_SIS + seg_loss
                        + self.shared_cycle_loss(cycle_loss_ISI, reconstruction_loss)
                        + self.apply_seg_identity_loss(real_S, training))

        total_loss_S = (gen_SI_loss + cycle_loss_ISI + reconstruction_loss
                        + self.shared_cycle_loss(cycle_loss_SIS, seg_loss)
                        + self.apply_imaging_identity_loss(real_I, training))

        result.update({
            'total_I_loss': total_loss_I,
            'total_S_loss': total_loss_S,
            'D_I_loss': disc_I_loss,
            'D_S_loss': disc_S_loss,
            'gen_IS_loss': gen_IS_loss,
            'gen_SI_loss': gen_SI_loss,
            'cycle_gen_ISI_loss': cycle_loss_ISI,
            'cycle_gen_SIS_loss': cycle_loss_SIS,
            'seg_loss': seg_loss,
            'reconstruction_loss': reconstruction_loss
        })

        return result, total_loss_I, total_loss_S, disc_I_loss, disc_S_loss, fake_I, fake_S

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
            result = gan.distributed_train_step(x, y)
        else:
            result = gan.distributed_test_step(x, y)
        utils.append_dict(results, result)

    for key, value in results.items():
        summary.scalar(key, tf.reduce_mean(value), epoch=epoch, training=training)

    return results
