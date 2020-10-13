import os
import numpy as np

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as tfk
import tensorflow_addons as tfa

from loss_functions import style_loss, dice_coef_loss

class CycleGan(keras.Model):
    def __init__(
        self,
        img_size,
        generator_AB,
        generator_BA,
        discriminator_A,
        discriminator_B,
        lambda_cycle=10.0,
        lambda_identity=0.5,

    ):
        super(CycleGan, self).__init__()
        self.img_size = img_size
        self.gen_AB = generator_AB
        self.gen_BA = generator_BA
        self.disc_A = discriminator_A
        self.disc_B = discriminator_B
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

    def compile(
        self,
        gen_AtoB_optimizer,
        gen_BtoA_optimizer,
        disc_A_optimizer,
        disc_B_optimizer,
        genA_loss_fn,
        genB_loss_fn,
        discA_loss_fn,
        discB_loss_fn,
        
    ):
        super(CycleGan, self).compile()
        self.gen_AtoB_optimizer = gen_AtoB_optimizer
        self.gen_BtoA_optimizer = gen_BtoA_optimizer
        self.disc_A_optimizer = disc_A_optimizer
        self.disc_B_optimizer = disc_B_optimizer
        self.generatorA_loss_fn = genA_loss_fn
        self.generatorB_loss_fn = genB_loss_fn
        self.discriminatorA_loss_fn = discA_loss_fn
        self.discriminatorB_loss_fn = discB_loss_fn
        # self.cycle_loss_fn = keras.losses.MeanAbsoluteError()
        # self.identity_loss_fn = keras.losses.MeanAbsoluteError()
        self.cycle_loss_fn_A = dice_coef_loss
        self.cycle_loss_fn_B = dice_coef_loss
        self.identity_loss_fn_A = dice_coef_loss
        self.identity_loss_fn_B = tfa.losses.SigmoidFocalCrossEntropy(from_logits = False)
        self.style_loss_fn = style_loss

        
    def train_step(self, batch_data):
        
        # A is RSOM data and B is synthetic data
        real_A, real_B = batch_data

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

        with tf.GradientTape(persistent=True) as tape:
            # Horse to fake zebra
            fake_B = self.gen_AB(real_A, training=True)
            # Zebra to fake horse -> B2A
            fake_A = self.gen_BA(real_B, training=True)

            # cycle: A -> B -> A
            cycled_A = self.gen_BA(fake_B, training=True)
            # cycle: B -> A -> B
            cycled_B = self.gen_AB(fake_A, training=True)

            # Identity mapping
            same_A = self.gen_BA(real_A, training=True)
            same_B = self.gen_AB(real_B, training=True)

            # Discriminator output
            disc_real_A = self.disc_A(real_A, training=True)
            disc_fake_A = self.disc_A(fake_A, training=True)

            disc_real_B = self.disc_B(real_B, training=True)
            disc_fake_B = self.disc_B(fake_B, training=True)

            # Generator adverserial loss
            gen_AB_loss = self.generatorA_loss_fn(disc_fake_B)
            gen_BA_loss = self.generatorB_loss_fn(disc_fake_A)

            # Generator cycle loss
            cycle_loss_A = self.cycle_loss_fn_A(real_B, cycled_B) * self.lambda_cycle
            cycle_loss_B = self.cycle_loss_fn_B(real_A, cycled_A) * self.lambda_cycle

            # Generator identity loss
            id_loss_A = (
                self.identity_loss_fn_A(real_B, same_B)
                * self.lambda_cycle
                * self.lambda_identity
            )
            id_loss_B = (
                self.identity_loss_fn_B(real_A, same_A)
                * self.lambda_cycle
                * self.lambda_identity
            )

            #  Generator cycle-style loss
            # style_loss_A = (
            #     self.style_loss_fn(real_A, cycled_A, self.img_size)
            #     * self.lambda_cycle
            #     * self.lambda_identity
            # )

            # style_loss_B = (
            #     self.style_loss_fn(real_B, cycled_B, self.img_size)
            #     * self.lambda_cycle
            #     * self.lambda_identity
            # )
            

            # Total generator loss
            total_loss_A = gen_AB_loss + cycle_loss_A + id_loss_A
            total_loss_B = gen_BA_loss + cycle_loss_B + id_loss_B

            # Discriminator loss
            disc_A_loss = self.discriminatorA_loss_fn(disc_real_A, disc_fake_A)
            disc_B_loss = self.discriminatorB_loss_fn(disc_real_B, disc_fake_B)


        # Get the gradients for the generators
        grads_A = tape.gradient(total_loss_A, self.gen_AB.trainable_variables)
        grads_B = tape.gradient(total_loss_B, self.gen_BA.trainable_variables)

        # Get the gradients for the discriminators
        disc_A_grads = tape.gradient(disc_A_loss, self.disc_A.trainable_variables)
        disc_B_grads = tape.gradient(disc_B_loss, self.disc_B.trainable_variables)

        # Update the weights of the generators
        self.gen_AtoB_optimizer.apply_gradients(
            zip(grads_A, self.gen_AB.trainable_variables)
        )
        self.gen_BtoA_optimizer.apply_gradients(
            zip(grads_B, self.gen_BA.trainable_variables)
        )

        # Update the weights of the discriminators
        self.disc_A_optimizer.apply_gradients(
            zip(disc_A_grads, self.disc_A.trainable_variables)
        )
        self.disc_B_optimizer.apply_gradients(
            zip(disc_B_grads, self.disc_B.trainable_variables)
        )
        
        
        return {
            "total_AB_loss": total_loss_A,
            "total_BA_loss": total_loss_B,
            "D_A_loss": disc_A_loss,
            "D_B_loss": disc_B_loss,
            "gen_AB_loss": gen_AB_loss,
            "gen_BA_loss": gen_BA_loss,
            "cycle_gen_AB_loss": cycle_loss_A,
            "cycle_gen_BA_loss": cycle_loss_B,
            "id_loss_A": id_loss_A,
            "id_loss_B": id_loss_B,
        }