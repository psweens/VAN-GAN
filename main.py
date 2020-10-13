import random
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
from tensorflow import keras

from cycleGAN import CycleGan
from generator import get_resnet_generator
from discriminator import get_discriminator
from loss_functions import generatorA_loss_fn, generatorB_loss_fn, discriminatorA_loss_fn, discriminatorB_loss_fn
from callback import GANMonitor

autotune = tf.data.experimental.AUTOTUNE

#  Allocate GPU memory as-needed
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     # Currently, memory growth needs to be the same across GPUs
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#     print(e)


# Size of original image stack
orig_size = (600,600,512,1)
# Size of the random crops to be used during training.
input_img_size = (64,64,512,1)
# Weights initializer for the layers.
kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
# Gamma initializer for instance normalization.
gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

buffer_size = 256
batch_size = 1
prefetch_size = 4
min_pixel_value = 0.0;

'''
PREPROCESSING
'''

#  Randomly sample image stack
pathA = np.load('rsom.npy', mmap_mode='r')
def data_gen_trainA():
    i = random.randint(0,pathA[0].shape[0]-1)
    yield np.load(pathA[0][i]) + 1.0
    
def data_gen_testA():
    i = random.randint(0,pathA[1].shape[0]-1)
    yield np.load(pathA[1][i]) + 1.0

pathB = np.load('synthetic.npy')
def data_gen_trainB():
    i = random.randint(0,pathB[0].shape[0]-1)    
    yield np.load(pathB[0][i]) + 1.0
    
def data_gen_testB():
    i = random.randint(0,pathB[1].shape[0]-1) 
    yield np.load(pathB[1][i]) + 1.0

# Process stack e.g. random crop & normalise
def signal_fn(image):
    arr = tf.image.random_crop(image, size=input_img_size)
    return tf.cond(tf.reduce_max(arr) == min_pixel_value, lambda: signal_fn(image), lambda: arr)

def load_filesA(image):
    return tf.image.random_crop(image, size=input_img_size)

def load_filesB(image):
    return tf.reshape(tf.py_function(signal_fn, inp=[image], 
                                     Tout=[tf.float32]), input_img_size)

# Create TF datasets
train_horses = tf.data.Dataset.from_generator(data_gen_trainA,output_types=tf.float32,
                                              output_shapes=orig_size)
train_horses = (
    train_horses.map(map_func=load_filesA, num_parallel_calls=autotune)
    .batch(batch_size)
    .prefetch(prefetch_size)
)

train_zebras = tf.data.Dataset.from_generator(data_gen_trainB,output_types=tf.float32,
                                              output_shapes=orig_size)
train_zebras = (
    train_zebras.map(map_func=load_filesB, num_parallel_calls=autotune)
    .batch(batch_size)
    .prefetch(prefetch_size)
)

test_horses = tf.data.Dataset.from_generator(data_gen_testA,output_types=tf.float32,
                                             output_shapes=orig_size)
# test_horses = (
#     test_horses.map(map_func=load_filesA, num_parallel_calls=autotune)
#     .batch(batch_size)
#     .prefetch(prefetch_size)
# )

test_zebras = tf.data.Dataset.from_generator(data_gen_testB,output_types=tf.float32,
                                              output_shapes=orig_size)
# test_zebras = (
#     test_zebras.map(map_func=load_filesB, num_parallel_calls=autotune)
#     .batch(batch_size)
#     .prefetch(prefetch_size)
# )



'''
TEST LOADED DATASETS
'''
#  Visualise some examples
nfig = 4
_, ax = plt.subplots(nfig, 2, figsize=(10, 15))
for i, samples in enumerate(zip(train_horses.take(1), train_zebras.take(1))):
    dA = samples[0][0] - 1.0
    dB = samples[1][0] - 1.0
    for j in range(0,nfig):
        showA = (((dA[:,:,j*int(input_img_size[2]/nfig),0] * 127.5) + 127.5).numpy()).astype(np.uint8)
        showB = (((dB[:,:,j*int(input_img_size[2]/nfig),0] * 127.5) + 127.5).numpy()).astype(np.uint8)
        ax[j, 0].imshow(showA, cmap='gray')
        ax[j, 1].imshow(showB, cmap='gray')
plt.show()

'''
DEFINE CYCLEGAN
'''

# Get the generators
gen_AtoB = get_resnet_generator(input_img_size=input_img_size,
                gamma_initializer=gamma_init,
                kernel_initializer=kernel_init,
                name="generator_AB")
gen_BtoA = get_resnet_generator(input_img_size=input_img_size,
                gamma_initializer=gamma_init,
                kernel_initializer=kernel_init,
                name="generator_BA")

# Get the discriminators
disc_A = get_discriminator(input_img_size=input_img_size,
                           kernel_initializer=kernel_init,
                           name="discriminator_X")
disc_B = get_discriminator(input_img_size=input_img_size,
                           kernel_initializer=kernel_init,
                           name="discriminator_Y")

# Create cycle gan model
cycle_gan_model = CycleGan(
    img_size=input_img_size, generator_AB=gen_AtoB, generator_BA=gen_BtoA, discriminator_A=disc_A, discriminator_B=disc_B
)

# Compile the model
cycle_gan_model.compile(
    gen_AtoB_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    gen_BtoA_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    disc_A_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    disc_B_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    genA_loss_fn=generatorA_loss_fn,
    genB_loss_fn=generatorB_loss_fn,
    discA_loss_fn=discriminatorA_loss_fn,
    discB_loss_fn=discriminatorB_loss_fn
)

# Callbacks
plotter = GANMonitor(test_AB=test_horses,test_BA=test_zebras, sub_img_size=input_img_size)
checkpoint_filepath = "./model_checkpoints/cyclegan_checkpoints.{epoch:03d}"
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath, savefreq=50
)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")

'''
TRAINING
'''
cycle_gan_model.fit(
    tf.data.Dataset.zip((train_horses, train_zebras)),
    epochs=5000,
    verbose=2,
    callbacks=[plotter, tensorboard_callback]
)