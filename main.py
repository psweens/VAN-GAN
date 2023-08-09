import os
import shutil
import glob
import argparse
import numpy as np
import scipy.stats as sp
import tensorflow as tf
from time import time
from vangan import VanGan, train
from custom_callback import GanMonitor
from dataset import DatasetGen
from preprocessing import DataPreprocessor
from tb_callback import TB_Summary
from utils import min_max_norm_tf, rescale_arr_tf, z_score_norm
from post_training import epoch_sweep

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

tf.keras.backend.clear_session()

print('*** Setting up GPU ***')
''' SET GPU MEMORY USAGE '''
physical_devices = tf.config.list_physical_devices('GPU')
# Prevent allocation of all memory
for i in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)

''' SET TF GPU STRATEGY '''
strategy = tf.distribute.MirroredStrategy(['GPU:0', 'GPU:1', 'GPU:2', 'GPU:3'])
# strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
# strategy = tf.distribute.ReductionToOneDevice
# strategy = tf.distribute.experimental.CentralStorageStrategy()
# strategy = tf.distribute.OneDeviceStrategy(device='GPU:0')


''' TENSORFLOW DEBUGGING '''
# tf.config.set_soft_device_placement(True)
# tf.debugging.enable_check_numerics()


''' ORGANISE TENSORBOARD OUTPUT FOLDERS '''
print('*** Organising tensorboard folders ***')
tensorboardDir = 'TB_Logs'
monitorDir = 'GANMonitor'
if os.path.isdir(tensorboardDir):
    shutil.rmtree(tensorboardDir)
else:
    os.makedirs(tensorboardDir)
if os.path.isdir(monitorDir):
    files = glob.glob(monitorDir + '/*')
    for f in files:
        os.remove(f)
else:
    os.makedirs(monitorDir)

summary = TB_Summary(tensorboardDir)  # Initialise TensorBoard summary helper

''' SET PARAMETERS '''
print('*** Setting VANGAN parameters ***')
args = argparse.ArgumentParser()
args.output_dir = '/mnt/sda/VG_Output'
args.N_DEVICES = len(physical_devices)
args.BUFFER_SIZE = 256
args.MIN_PIXEL_VALUE = -1.0
args.MAX_PIXEL_VALUE = 0.8

# Training parameters
args.EPOCHS = 200
args.BATCH_SIZE = 3
args.GLOBAL_BATCH_SIZE = args.N_DEVICES * args.BATCH_SIZE
args.PREFETCH_SIZE = 4
args.INITIAL_LR = 2e-4  # Learning rate
args.INITIATE_LR_DECAY = 0.5 * args.EPOCHS  # Set start of learning rate decay to 0
args.NO_NOISE = args.EPOCHS  # Set when discriminator noise decays to 0

# Image parameters
args.CHANNELS = 1
args.DIMENSIONS = 3
args.RAW_IMG_SIZE = (512, 512, 140, args.CHANNELS)  # Unprocessed imaging domain image dimensions
args.TARG_RAW_IMG_SIZE = (512, 512, 128, args.CHANNELS)  # Target size if downsampling
args.SYNTH_IMG_SIZE = (512, 512, 128)  # Unprocessed segmentation domain image dimensions
args.TARG_SYNTH_IMG_SIZE = (512, 512, 128)  # Target size if downsampling
args.SUBVOL_PATCH_SIZE = (128, 128, 128)  # Size of subvolume to be trained on
# Set model input image size for training (based on above)
if args.DIMENSIONS == 2:
    args.INPUT_IMG_SIZE = (
        args.GLOBAL_BATCH_SIZE,
        args.SUBVOL_PATCH_SIZE[0],
        args.SUBVOL_PATCH_SIZE[1],
        1,
    )
else:
    args.INPUT_IMG_SIZE = (
        args.GLOBAL_BATCH_SIZE,
        args.SUBVOL_PATCH_SIZE[0],
        args.SUBVOL_PATCH_SIZE[1],
        args.SUBVOL_PATCH_SIZE[2],
        1,
    )

# Set callback parameters
args.PERIOD_2D_CALLBACK = 2  # Period of epochs to output a 2D validation dataset example
args.PERIOD_3D_CALLBACK = 2  # Period of epochs to output a 3D validation dataset example

'''' PREPROCESSING '''
imaging_data = DataPreprocessor(args,
                                raw_path='/mnt/sdb/3DcycleGAN_simLNet_LNet/raw_data/simLNet',
                                main_dir='/mnt/sdb/3DcycleGAN_simLNet_LNet/',
                                partition_id='A',
                                partition_filename='dataA_partition.pkl',
                                tiff_size=args.RAW_IMG_SIZE,
                                target_size=args.TARG_RAW_IMG_SIZE)

synth_data = DataPreprocessor(args,
                              raw_path='/mnt/sdb/3DcycleGAN_simLNet_LNet/raw_data/LNet',
                              main_dir='/mnt/sdb/3DcycleGAN_simLNet_LNet/',
                              partition_id='B',
                              partition_filename='dataB_partition.pkl',
                              tiff_size=args.SYNTH_IMG_SIZE,
                              target_size=args.TARG_SYNTH_IMG_SIZE)


# Function used for preprocessing imaging domain images
# The following is used for preprocessing raster-scanning optoacoustic mesoscopic (RSOM) image volumes
def preprocess_rsom_images(img, lower_thresh=0.05, upper_thresh=99.95):
    """
    Preprocesses a 3D image array using slice-wise Z-score normalization and clipping of upper and lower percentiles.
    
    Args:
    - img (np.ndarray): A 3D numpy array representing the image to be preprocessed.
    - lower_thresh (float): The lower percentile value to clip the image at (default: 0.05).
    - upper_thresh (float): The upper percentile value to clip the image at (default: 99.95).
    
    Returns:
    - np.ndarray: The preprocessed 3D numpy array.
    """

    # Slice-wise Z-Score Normalisation
    for z in range(img.shape[2]):
        img[..., z] = z_score_norm(img[..., z])

    # Clipping of upper and lower percentiles
    lp = sp.scoreatpercentile(img, lower_thresh)
    up = sp.scoreatpercentile(img, upper_thresh)
    img[img < lp] = lp
    img[img > up] = up

    return img


# Perform any preprocessing of images if neccessary
# imaging_data.preprocess(preprocess_fn=preprocess_rsom_images,
#                         save_filtered=True,
#                         resize=True)
# synth_data.preprocess(resize=True)

# Load dataset partitions
imaging_data.load_partition('/mnt/sdb/3DcycleGAN_simLNet_LNet/dataA_partition.pkl')
synth_data.load_partition('/mnt/sdb/3DcycleGAN_simLNet_LNet/dataB_partition.pkl')

''' GENERATE TENSORFLOW DATASETS '''
print('*** Generating datasets for model ***')


# Define function to preprocess imaging domain image on the fly (otf)
# Min/max batch normalisation and rescaling to [-1,1] shown here
@tf.function
def process_imaging_otf(tensor):
    # Calculate the maximum and minimum values along the batch dimension
    max_vals = tf.reduce_max(tensor, axis=(1, 2, 3, 4), keepdims=True)
    min_vals = tf.reduce_min(tensor, axis=(1, 2, 3, 4), keepdims=True)

    # Normalize the tensor between -1 and 1
    return 2.0 * (tensor - min_vals) / (max_vals - min_vals) - 1.0


# Define dataset class
getDataset = DatasetGen(args=args,
                        imaging_domain_data=imaging_data.partition,
                        seg_domain_data=synth_data.partition,
                        strategy=strategy,
                        otf_imaging=process_imaging_otf  # Set to None if OTF processing not needed
                        )

''' CALCULATE NUMBER OF TRAINING / VALIDATION STEPS '''
args.train_steps = int(np.amax([len(imaging_data.partition['training']),
                                len(synth_data.partition['training'])]) / args.GLOBAL_BATCH_SIZE)

args.val_steps = int(np.amax([len(imaging_data.partition['validation']),
                              len(synth_data.partition['validation'])]) / args.GLOBAL_BATCH_SIZE)

''' DEFINE VANGAN '''
vangan_model = VanGan(args,
                      strategy=strategy,
                      gen_i2s='resUnet',
                      gen_s2i='resUnet'
                      )

''' DEFINE CUSTOM CALLBACK '''
plotter = GanMonitor(args,
                     dataset=getDataset,
                     imaging_val_data=imaging_data.partition['validation'],
                     segmentation_val_data=synth_data.partition['validation'],
                     process_imaging_domain=process_imaging_otf
                     )

''' TRAIN VAN-GAN MODEL '''
for epoch in range(args.EPOCHS):
    print(f'\nEpoch {epoch + 1:03d}/{args.EPOCHS:03d}')
    start = time()

    vangan_model.current_epoch = epoch
    plotter.on_epoch_start(vangan_model, epoch, args)

    'Training GAN for fixed no. of steps'
    results = train(getDataset.train_dataset, vangan_model, summary, epoch, args.train_steps, 'Train')
    summary.losses(results)

    'Run GAN for validation dataset'
    results = train(getDataset.val_dataset, vangan_model, summary, epoch, args.val_steps, 'Validate',
                    training=False)
    summary.losses(results)

    if epoch % args.PERIOD_2D_CALLBACK == 1 or epoch == args.EPOCHS - 1:
        plotter.on_epoch_end(vangan_model, epoch, args)
        vangan_model.save_checkpoint(epoch=epoch)

    end = time()
    summary.scalar('elapse', end - start, epoch=epoch, training=True)

''' CREATE VANGAN PREDICTIONS '''
# Predict segmentation probability maps for imaging test dataset
plotter.run_mapping(vangan_model, imaging_data.partition['testing'], args.INPUT_IMG_SIZE, filetext='VANGAN_',
                    filepath=args.output_dir, segmentation=True, stride=(25, 25, 25))
# Prediction fake imaging data using synthetic segmentation test dataset
plotter.run_mapping(vangan_model, synth_data.partition['testing'], args.INPUT_IMG_SIZE, filetext='VANGAN_',
                    filepath=args.output_dir, segmentation=False, stride=(25, 25, 25))

''' TESTING PREDICTIONS ACROSS EPOCHS '''
epoch_sweep(args,
            vangan_model,
            plotter,
            test_path='/PATH/TO/TEST/DATA/',  # Can use imaging_data.partition['testing']
            start=100,
            end=200,
            segmentation=True  # Set to False if fake imaging is wanted
            )

''' SEGMENTING NEW IMAGES '''
# Alternatively, to run VANGAN on a directory of images (saved as .npy) using the following example script
new_imaging_data = DataPreprocessor()  # Create data preprocessor
new_imaging_data.process_new_data(current_path='/PATH/TO/DATA/',
                                  new_path='/PATH/TO/SAVE/DATA/',
                                  preprocess_fn=preprocess_rsom_images,
                                  tiff_size=args.RAW_IMG_SIZE,
                                  target_size=args.TARG_RAW_IMG_SIZE,
                                  resize=True)

filepath = '/PATH/TO/SAVE/DATA/'
img_files = os.listdir(filepath)
for file in img_files:
    img_files[file] = os.path.join(filepath, file)
plotter.run_mapping(vangan_model, img_files, args.INPUT_IMG_SIZE, filetext='VANGAN_', filepath=args.output_dir,
                    segmentation=True, stride=(25, 25, 25))
