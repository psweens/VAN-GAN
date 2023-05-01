import os
import utils
import tensorflow as tf
from tqdm import tqdm
from generator import get_resnet_generator
from discriminator import get_discriminator
from discriminator2D import get_2D_discriminator
from loss_functions import (generator_loss_fn, 
                            discriminator_loss_fn, 
                            cycle_loss, 
                            identity_loss, 
                            cycle_seg_loss, 
                            wasserstein_generator_loss, 
                            wasserstein_discriminator_loss, 
                            reduce_mean,
                            cycle_perceptual)
from vnet_model import custom_vnet
from resunet_model import ResUNet
from resunet_model_2D import TwoDResUNet

class VanGan():
    def __init__(
        self,
        args,
        strategy,
        lambda_cycle=10.0,
        lambda_identity=5,
        genAB_typ = 'resnet',
        genBA_typ = 'resnet',
        wasserstein = False,
        ncritic = 5,
        gp_weight=10.0
    ):
        self.strategy = strategy
        self.n_devices = args.N_DEVICES
        self.img_size = args.INPUT_IMG_SIZE
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity,
        self.channels = args.CHANNELS
        self.genAB_typ = genAB_typ
        self.genBA_typ = genBA_typ
        self.global_batch_size = args.GLOBAL_BATCH_SIZE
        self.dims = args.DIMENSIONS
        if self.dims == 2:
            self.subvol_patch_size = (args.SUBVOL_PATCH_SIZE[0], args.SUBVOL_PATCH_SIZE[1], self.channels)
            self.seg_subvol_patch_size = (args.SUBVOL_PATCH_SIZE[0], args.SUBVOL_PATCH_SIZE[1], 1)
        else:
            self.subvol_patch_size = (args.SUBVOL_PATCH_SIZE[0], args.SUBVOL_PATCH_SIZE[1], args.SUBVOL_PATCH_SIZE[2], self.channels)
            self.seg_subvol_patch_size = (args.SUBVOL_PATCH_SIZE[0], args.SUBVOL_PATCH_SIZE[1], args.SUBVOL_PATCH_SIZE[2], 1)
        self.gamma_init = args.GAMMA_INIT
        self.kernel_init = args.KERNEL_INIT
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
        self.perceptual_loss = cycle_perceptual
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
            
            if self.genAB_typ == 'resnet':
                self.gen_AB = get_resnet_generator(
                                    input_img_size=self.subvol_patch_size,
                                    batch_size=self.global_batch_size,
                                    # gamma_initializer=self.gamma_init,
                                    # kernel_initializer=self.kernel_init,
                                    name='generator_AB',
                                    num_downsampling_blocks=3,
                                    num_upsample_blocks=3
                                    )
            elif self.genAB_typ == 'vnet':
                self.gen_AB = custom_vnet(
                                input_shape = self.subvol_patch_size,
                                num_classes=1,
                                activation='relu',
                                use_batch_norm=False,
                                upsample_mode='upsample', 
                                dropout=0.5,
                                dropout_change_per_layer=0.0,
                                dropout_type='spatial',
                                use_dropout_on_upsampling=False,
                                kernel_initializer=self.kernel_init,
                                use_attention_gate=False,
                                filters=32,
                                num_layers=4,
                                output_activation='tanh',
                                )
            elif self.genAB_typ == 'resUnet':
                if self.dims == 2:
                    self.gen_AB = TwoDResUNet(
                                    input_shape = self.subvol_patch_size,
                                    num_classes=1,
                                    activation='relu',
                                    use_batch_norm=False,
                                    upsample_mode='simple', 
                                    dropout=0.1,
                                    dropout_change_per_layer=0.1,
                                    dropout_type='none',
                                    use_dropout_on_upsampling=False,
                                    kernel_initializer=self.kernel_init,
                                    use_attention_gate=False,
                                    filters=16,
                                    num_layers=4,
                                    output_activation='tanh',
                                    )
                else:
                    self.gen_AB = ResUNet(
                                    input_shape = self.subvol_patch_size,
                                    num_classes=1,
                                    activation='relu',
                                    use_batch_norm=False,
                                    upsample_mode='simple', 
                                    dropout=0.1,
                                    dropout_change_per_layer=0.1,
                                    dropout_type='none',
                                    use_dropout_on_upsampling=False,
                                    kernel_initializer=self.kernel_init,
                                    use_attention_gate=False,
                                    filters=16,
                                    num_layers=4,
                                    # output_activation=None,
                                    )
            else:
                raise ValueError('AB Generator type not recognised')
                
            if self.genBA_typ == 'resnet':
                self.gen_BA = get_resnet_generator(
                                    input_img_size=self.subvol_patch_size,
                                    batch_size=self.global_batch_size,
                                    # gamma_initializer=self.gamma_init,
                                    #kernel_initializer=self.kernel_init,
                                    name='generator_BA',
                                    num_downsampling_blocks=3,
                                    num_upsample_blocks=3
                                    )
            elif self.genBA_typ == 'vnet':
                self.gen_BA = custom_vnet(
                                input_shape = self.subvol_patch_size,
                                num_classes=1,
                                activation='relu',
                                use_batch_norm=True,
                                upsample_mode='deconv', 
                                dropout=0.5,
                                dropout_change_per_layer=0.0,
                                dropout_type='spatial',
                                use_dropout_on_upsampling=False,
                                kernel_initializer=self.kernel_init,
                                use_attention_gate=False,
                                filters=16,
                                num_layers=4,
                                output_activation='tanh',
                                addnoise=False
                                )
            elif self.genBA_typ == 'resUnet':
                if self.dims == 2:
                    self.gen_BA = TwoDResUNet(
                                    input_shape = self.seg_subvol_patch_size,
                                    num_classes=1,
                                    activation='relu',
                                    use_batch_norm=False,
                                    upsample_mode='simple', 
                                    dropout=0.1,
                                    dropout_change_per_layer=0.1,
                                    dropout_type='none',
                                    use_dropout_on_upsampling=False,
                                    kernel_initializer=self.kernel_init,
                                    use_attention_gate=False,
                                    filters=16,
                                    num_layers=4,
                                    output_activation='tanh',
                                    channels=self.channels
                                    )
                else:
                    self.gen_BA = ResUNet(
                                    input_shape = self.seg_subvol_patch_size,
                                    num_classes=1,
                                    activation='relu',
                                    use_batch_norm=False,
                                    upsample_mode='simple', 
                                    dropout=0.1,
                                    dropout_change_per_layer=0.1,
                                    dropout_type='none',
                                    use_dropout_on_upsampling=False,
                                    kernel_initializer=self.kernel_init,
                                    use_attention_gate=False,
                                    filters=16,
                                    num_layers=4,
                                    # output_activation=None,
                                    use_input_noise=False
                                    )
            else:
                raise ValueError('BA Generator type not recognised')
        
        
            # Get the discriminators
            if self.dims == 2:
                self.disc_A = get_2D_discriminator(
                                    input_img_size=self.subvol_patch_size,
                                    batch_size=self.global_batch_size,
                                    kernel_initializer=self.kernel_init,
                                    name='discriminator_A',
                                    filters=64,
                                    use_dropout=False,
                                    wasserstein=self.wasserstein,
                                    use_SN=False,
                                    use_input_noise=True,
                                    use_layer_noise=True,
                                    noise_std=self.layer_noise
                                    )
                self.disc_B = get_2D_discriminator(
                                    input_img_size=self.seg_subvol_patch_size,
                                    batch_size=self.global_batch_size,
                                    kernel_initializer=self.kernel_init,
                                    name='discriminator_B',
                                    filters=64,
                                    use_dropout=False,
                                    wasserstein=self.wasserstein,
                                    use_SN=False,
                                    use_input_noise=True,
                                    use_layer_noise=True,
                                    noise_std=self.layer_noise
                                    )
            else:
                self.disc_A = get_discriminator(
                                    input_img_size=self.subvol_patch_size,
                                    batch_size=self.global_batch_size,
                                    kernel_initializer=self.kernel_init,
                                    name='discriminator_A',
                                    filters=64,
                                    use_dropout=False,
                                    wasserstein=self.wasserstein,
                                    use_SN=False,
                                    use_input_noise=True,
                                    use_layer_noise=True,
                                    noise_std=self.layer_noise
                                    )
                self.disc_B = get_discriminator(
                                    input_img_size=self.seg_subvol_patch_size,
                                    batch_size=self.global_batch_size,
                                    kernel_initializer=self.kernel_init,
                                    name='discriminator_B',
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
                
                self.gen_A_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0., beta_2=0.9)#, clipnorm=10.0)
                self.gen_B_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0., beta_2=0.9)#, clipnorm=10.0)
                self.disc_A_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0., beta_2=0.9)#, clipnorm=10.0)
                self.disc_B_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0., beta_2=0.9)#, clipnorm=10.0)
                
            else:
                # Initialise decay rates
                self.dA_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                                            2e-4,
                                            decay_steps=5*self.train_steps,
                                            decay_rate=0.98,
                                            staircase=False)
                
                self.dB_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                                            2e-4,
                                            decay_steps=5*self.train_steps,
                                            decay_rate=0.98,
                                            staircase=False)
                
                self.gen_A_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4,
                                                            beta_1=0.5,
                                                            beta_2=0.9,
                                                            clipnorm=100)
                self.gen_B_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4,
                                                            beta_1=0.5,
                                                            beta_2=0.9,
                                                            clipnorm=100)
                self.disc_A_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4,
                                                            beta_1=0.5,
                                                            beta_2=0.9,
                                                            clipnorm=100)
                self.disc_B_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4,
                                                            beta_1=0.5,
                                                            beta_2=0.9,
                                                            clipnorm=100)
            
            # Initialise checkpoint
            self.checkpoint = tf.train.Checkpoint(gen_AB=self.gen_AB,
                                            gen_BAF=self.gen_BA,
                                            disc_A=self.disc_A,
                                            disc_B=self.disc_B,
                                            gen_A_optimizer=self.gen_A_optimizer,
                                            gen_B_optimizer=self.gen_B_optimizer,
                                            disc_A_optimizer=self.disc_A_optimizer,
                                            disc_B_optimizer=self.disc_B_optimizer)
     
    def save_checkpoint(self, epoch):
        """ save checkpoint to checkpoint_dir, overwrite if exists """
        self.checkpoint.write(self.checkpoint_prefix+"_e{epoch}".format(epoch=epoch+1))
        print(f'\nSaved checkpoint to {self.checkpoint_prefix}\n')

    def load_checkpoint(self, epoch=None, expect_partial: bool = False, newpath=None):
        """ load checkpoint from checkpoint_dir if exists """
        if newpath is not None:
            self.checkpoint_prefix = os.path.join(newpath, 'checkpoint')
        checkpoint_path = self.checkpoint_prefix +"_e{epoch}".format(epoch=epoch)
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
            
    def addNoise(self, img, rate):
        return tf.clip_by_value(img + tf.random.normal(tf.shape(img), 0.0, rate), -1., 1.)
    
    def binarise_tensor(self, arr):
        return tf.where(tf.math.greater_equal(arr, tf.zeros(tf.shape(arr))),
                            tf.ones(tf.shape(arr)),
                            tf.math.negative(tf.ones(tf.shape(arr))))
            
    def computeLosses(self, real_A, real_B, result, training=True):
        
        # Can be used to debug dataset numerics
        #tf.debugging.check_numerics(real_A, 'real_A failure')
        #tf.debugging.check_numerics(real_B, 'real_B failure')

        # For CycleGan, we need to calculate different
        # kinds of losses for the generators and discriminators.
        # We will perform the following steps here:
        #
        # 1. Pass real images through the generators and get the generated images
        # 2. Pass the generated images back to the generators to check if we
        #    we can predict the original image from the generated image.
        # 3. Do an identitB mapping of the real images using the generators.
        # 4. Pass the generated images in 1) to the corresponding discriminators.
        # 5. Calculate the generators total loss (adverserial + cycle + identity)
        # 6. Calculate the discriminators loss
        # 7. Update the weights of the generators
        # 8. Update the weights of the discriminators
        # 9. Return the losses in a dictionary
        
        # A -> B
        fake_B = self.gen_AB(real_A, training=training)
        # B -> A
        fake_A = self.gen_BA(real_B, training=training)
        
        # Cycle loss
        cycled_B = self.gen_AB(fake_A, training=training)
    
        cycle_loss_A = self.cycle_loss_fn(self, real_B, cycled_B, typ="bce")
        
        
        seg_loss = self.seg_loss_fn(self, real_B, cycled_B)
        cycled_A = self.gen_BA(fake_B, training=training)
        cycle_loss_B = self.cycle_loss_fn(self, real_A, cycled_A, typ='L2')
        
        perceptualA_loss = self.perceptual_loss(self, real_A, cycled_A)
        
        # Identity mapping
        # id_BA_loss = self.identity_loss_fn(self, real_A, self.gen_BA(real_A, training=True))
        # id_AB_loss = self.identity_loss_fn(self, real_B, self.gen_AB(real_B, training=True), typ='cldice')
        
                 
        # Discriminator outputs         
        disc_real_B = self.disc_B(real_B, training=training)
        disc_fake_B = self.disc_B(fake_B, training=training)
        
        disc_real_A = self.disc_A(real_A, training=training)
        disc_fake_A = self.disc_A(fake_A, training=training)
        

        # Generator & discriminator loss
        if self.wasserstein:
            gen_AB_loss = self.wasserstein_generator_loss(self, disc_fake_B)
            gen_BA_loss = self.wasserstein_generator_loss(self, disc_fake_A)      
            disc_A_loss = self.wasserstein_discriminator_loss(self, disc_real_A, disc_fake_A)
            disc_B_loss = self.wasserstein_discriminator_loss(self, disc_real_B, disc_fake_B)
           
        else:
            gen_AB_loss = self.generator_loss_fn(self, disc_fake_B, from_logits=True)
            gen_BA_loss = self.generator_loss_fn(self, disc_fake_A, from_logits=True)
            disc_A_loss = self.discriminator_loss_fn(self, disc_real_A, disc_fake_A, from_logits=True)
            disc_B_loss = self.discriminator_loss_fn(self, disc_real_B, disc_fake_B, from_logits=True)
        
        # Total generator loss
        total_loss_A = gen_AB_loss + cycle_loss_A + seg_loss #+ id_BA_loss
        total_loss_B = gen_BA_loss + cycle_loss_B + perceptualA_loss #+ id_AB_loss
        
        result.update({
            'total_AB_loss': total_loss_A,
            'total_BA_loss': total_loss_B,
            'D_A_loss': disc_A_loss,
            'D_B_loss': disc_B_loss,
            'gen_AB_loss': gen_AB_loss,
            'gen_BA_loss': gen_BA_loss,
            'cycle_gen_AB_loss': cycle_loss_A,
            'cycle_gen_BA_loss': cycle_loss_B,
            'seg_loss': seg_loss,
            'perceptualA_loss': perceptualA_loss,
            # 'identity_AB': id_AB_loss,
            # 'identity_BA': id_BA_loss
        })
        
        return result, total_loss_A, total_loss_B, disc_A_loss, disc_B_loss, fake_A, fake_B
    
    def gradient_penalty(self, real, fake, descrip='A'):
        alpha = tf.random.normal([self.batch_size, 1, 1, 1, 1], 0.0, 1.0)
        diff = fake - real
        interpolated = real + alpha * diff
        if descrip == 'A':
            pred = self.disc_A(interpolated, training=True)
        else:
            pred = self.disc_B(interpolated, training=True)
        grads = tf.gradients(pred, interpolated)[0]  
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3, 4]) + 1.e-12) # small constant add to prevent division by zero
        gp = reduce_mean(self, (norm - 1.0) ** 2)
        return gp
    
    def train_step(self, real_A, real_B):
        result = {}
        with tf.GradientTape(persistent=True) as tape:  
            result, total_loss_A, total_loss_B, disc_A_loss, disc_B_loss, fake_A, fake_B = self.computeLosses(real_A, real_B, result, training=True)   

        if self.wasserstein:
            
            if self.updateGen:
                self.gen_A_optimizer.minimize(loss=total_loss_A,
                                              var_list=self.gen_AB.trainable_variables,
                                              tape=tape)
                self.gen_B_optimizer.minimize(loss=total_loss_B,
                                              var_list=self.gen_BA.trainable_variables,
                                              tape=tape)
            
            if self.initModel == False:
                gp = self.gradient_penalty(real_A, fake_A, descrip='A')
                disc_A_loss = disc_A_loss + gp * self.gp_weight
                
                gp = self.gradient_penalty(real_B, fake_B, descrip='B')
                disc_B_loss = disc_B_loss + gp * self.gp_weight

            
            # clipping weights of discriminators as told in the
            # WasserteinGAN paper to enforce Lipschitz constraint.
            # clip_values = [-0.01, 0.01]
            # self.clip_discriminator_A_var_op = [var.assign(tf.clip_by_value(var, clip_values[0], clip_values[1])) for
            #    var in self.disc_A.trainable_variables]
            # self.clip_discriminator_B_var_op = [var.assign(tf.clip_by_value(var, clip_values[0], clip_values[1])) for
            #    var in self.disc_B.trainable_variables]
            
            
        else:
            self.gen_A_optimizer.minimize(loss=total_loss_A,
                                          var_list=self.gen_AB.trainable_variables,
                                          tape=tape)
            self.gen_B_optimizer.minimize(loss=total_loss_B,
                                          var_list=self.gen_BA.trainable_variables,
                                          tape=tape)
                
                
        self.disc_A_optimizer.minimize(loss=disc_A_loss,
                                      var_list=self.disc_A.trainable_variables,
                                      tape=tape)
        self.disc_B_optimizer.minimize(loss=disc_B_loss,
                                      var_list=self.disc_B.trainable_variables,
                                      tape=tape)        
        
        return result
    
    def test_step(self, real_A, real_B):
        result = {}
        result, _, _, _, _, _, _ = self.computeLosses(real_A, real_B, result, training=False)
        return result
    
    def reduce_dict(self, d: dict):
        ''' reduce items in dictionary d '''
        for k, v in d.items():
          d[k] = self.strategy.reduce(tf.distribute.ReduceOp.SUM, v, axis=None)
    
    @tf.function
    def distributed_train_step(self, x, y):
        results = self.strategy.run(self.train_step, args=(x, y))
        self.reduce_dict(results)
        return results
  
    @tf.function
    def distributed_test_step(self, x, y):
        results = self.strategy.run(self.test_step, args=(x, y))
        self.reduce_dict(results)
        return results

def train(args, ds, gan, summary, epoch: int, steps=None, desc=None, training=True):
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
            
