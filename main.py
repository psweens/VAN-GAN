import numpy as np
import shutil
import glob
import os
import argparse
import utils
from time import time
from scipy import ndimage
import scipy.stats as sp
from skimage import io
from preprocessing import load_volume
# time.sleep(7200)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = 0
import tensorflow as tf


from tensorflow import keras
from cycleGAN import CycleGan, train
from callback import GANMonitor
from preprocessing import load_dict
from dataset import getDataset
from preprocessing import split_dataset, process_tiff

tf.keras.backend.clear_session()

# tf.config.experimental.set_lms_enabled(True)

physical_devices = tf.config.list_physical_devices('GPU')
for i in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)

strategy = tf.distribute.MirroredStrategy(['GPU:0', 'GPU:1', 'GPU:2', 'GPU:3'])
# strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
# strategy = tf.distribute.ReductionToOneDevice
# strategy = tf.distribute.experimental.CentralStorageStrategy()
# strategy = tf.distribute.OneDeviceStrategy(device='GPU:0')

''' Tensorflow debugging '''
# tf.config.set_soft_device_placement(True)
# tf.debugging.enable_check_numerics()
    

''' Clean output folders '''
tensorboardDir = 'logs'
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
    os.makedirs(GANMonitor)


''' PARAMETERS '''
args = argparse.ArgumentParser()
args.output_dir = '/media/sweene01/5e3122f3-5d00-4bc0-b099-982762bf3999/cycleGAN_output'
args.N_DEVICES = len(physical_devices)
args.BUFFER_SIZE = 256
args.EPOCHS = 1000
args.BATCH_SIZE = 1
args.GLOBAL_BATCH_SIZE = args.N_DEVICES * args.BATCH_SIZE
args.PREFETCH_SIZE = 4
args.MIN_PIXEL_VALUE = -1.0
args.MAX_PIXEL_VALUE = 0.8
# args.KERNEL_INIT = keras.initializers.RandomNormal(mean=0.0, stddev=0.05)
# args.GAMMA_INIT = keras.initializers.RandomNormal(mean=0.0, stddev=0.05)

args.INITIAL_LR = 2e-4
args.INITIATE_LR_DECAY = args.EPOCHS#0.5 * args.EPOCHS
args.NO_NOISE = 0.1*args.EPOCHS

args.RAW_IMG_SIZE = (600, 600, 140)
args.TARG_IMG_SIZE = (600, 600, 128)
args.SYNTH_IMG_SIZE = (512, 512, 128)
args.SUBVOL_PATCH_SIZE = (128, 128, 128, 1)
# args.SUBVOL_STACK = (
#     args.SUBVOL_PATCH_SIZE[0],
#     args.SUBVOL_PATCH_SIZE[1],
#     args.SUBVOL_PATCH_SIZE[2] * 2,
#     1,
# )
args.INPUT_IMG_SIZE = (
    args.GLOBAL_BATCH_SIZE,
    args.SUBVOL_PATCH_SIZE[0],
    args.SUBVOL_PATCH_SIZE[1],
    args.SUBVOL_PATCH_SIZE[2],
    1,
)

# Weights initializer for the layers.
args.KERNEL_INIT = keras.initializers.HeNormal()
# Gamma initializer for instance normalization.
args.GAMMA_INIT = keras.initializers.HeNormal()

# initialize TensorBoard summary helper
summary = utils.Summary('logs/')

''''PREPROCESSING '''

# print('*** Preparing dataset A ***')
# split_dataset(raw_path='/media/sweene01/5e3122f3-5d00-4bc0-b099-982762bf3999/CH_TP_1_and_2_Large/raw_data/',
#                 new_dir='/media/sweene01/5e3122f3-5d00-4bc0-b099-982762bf3999/CH_TP_1_and_2_Large/',
#                 partition_id='A',
#                 partition_filename='dataA_partition.pkl',
#                 tiff_size=args.RAW_IMG_SIZE,
#                 target_size=args.TARG_IMG_SIZE,
#                 move_only=False,
#                 local_filter=True,
#                 global_filter=True,
#                 save_filtered=True)

# print('*** Preparing dataset B ***')
# split_dataset(raw_path='/media/sweene01/SSD/VS-GAN_LinaEars/raw_data/Lnet_aniso/',
#               new_dir='/media/sweene01/SSD/VS-GAN_LinaEars/',
#               partition_id='B',
#               partition_filename='dataB_partition.pkl',
#               tiff_size=(512,512,700),
#               target_size=args.SYNTH_IMG_SIZE,
#               move_only=False,
#               local_filter=False,
#               global_filter=False,
#               save_filtered=False)


# Load partitions
pathA = load_dict('/media/sweene01/5e3122f3-5d00-4bc0-b099-982762bf3999/VS-GAN_All_Data_plusCH/dataA_partition.pkl')
pathB = load_dict('/media/sweene01/SSD/3DcycleGAN_data/dataB_partition.pkl')
# pathA = load_dict('/media/sweene01/SSD/3DcycleGAN_simLNet_LNet/dataA_partition.pkl')
# pathB = load_dict('/media/sweene01/SSD/3DcycleGAN_simLNet_LNet/dataB_partition.pkl')

# pathB['training'].sort()
# pathB['testing'].sort()
# pathB['validation'].sort()

# pathB['training'] = np.delete(pathB['training'], np.arange(219,334))
# pathB['testing'] = np.delete(pathB['testing'], np.arange(20,36))
# pathB['validation'] = np.delete(pathB['validation'], np.arange(27,42))
    

''' Generate datasets '''
train_ds, val_ds, valFullDatasetA, valFullDatasetB = getDataset(args, 
    pathA=pathA, pathB=pathB, strategy=strategy
)

''' Calculate number of training / validation steps '''
args.train_steps = int(
    np.amin([len(pathA['training']), len(pathB['training'])]) / args.GLOBAL_BATCH_SIZE
)

args.val_steps = int(
    np.amin([len(pathA['validation']), len(pathB['validation'])]) / args.GLOBAL_BATCH_SIZE
)

''' DEFINE CYCLEGAN '''
cycle_gan_model = CycleGan(args, 
                           strategy=strategy,
                           genAB_typ='resUnet',
                           genBA_typ='resUnet',
                           wasserstein = False)

''' Define custom callback '''
args.period = 2
args.period3D = 2
plotter = GANMonitor(
    imgSize=args.INPUT_IMG_SIZE,
    test_AB=valFullDatasetA,
    test_BA=valFullDatasetB,
    Alist=pathA['validation'],
    Blist=pathB['validation'],
    period=args.period,
    period3D=args.period3D,
    model_path='/media/sweene01/5e3122f3-5d00-4bc0-b099-982762bf3999/cycleGAN_saved'
)

''' LOAD CHECKPOINT '''
# cycle_gan_model.load_checkpoint(epoch=192,
#                                 newpath='/media/sweene01/5e3122f3-5d00-4bc0-b099-982762bf3999/VAN-GAN Paper/VANGAN_Final/All_Data_plusCH_NoPretraining/cycleGAN_output/checkpoints/')

# cycle_gan_model.checkpoint_prefix = '/media/sweene01/5e3122f3-5d00-4bc0-b099-982762bf3999/cycleGAN_output'

# cycle_gan_model.checkpoint_loaded = False

''' Train CycleGAN '''
for epoch in range(args.EPOCHS):
    print(f'\nEpoch {epoch + 1:03d}/{args.EPOCHS:03d}')
    start = time()
    
    cycle_gan_model.current_epoch = epoch
    plotter.on_epoch_start(cycle_gan_model, epoch, args)
    
    'Training GAN for fixed no. of steps'
    results = train(args, train_ds, cycle_gan_model, summary, epoch, args.train_steps, 'Train')
    summary.losses(results)
    
    'Run GAN for validation dataset'
    results = train(args, val_ds, cycle_gan_model, summary, epoch, args.val_steps, 'Validate', training=False)
    summary.losses(results)
    
    
    if (epoch) % args.period == 1 or epoch == args.EPOCHS - 1:
        plotter.on_epoch_end(cycle_gan_model, epoch, args)  
        cycle_gan_model.save_checkpoint(epoch=epoch)
        
    end = time()
    summary.scalar('elapse', end - start, epoch=epoch, training=True)


# ''' Predictions '''
# filepath = '/media/sweene01/5e3122f3-5d00-4bc0-b099-982762bf3999/Ellie_Test/Downsampled'
# newfiles = os.listdir(filepath)
# for file in range(len(newfiles)):
#     if not newfiles[file] == 'filtered' and not newfiles[file] == 'valA':
#         process_tiff(file=newfiles[file], 
#                       raw_path=filepath, 
#                       new_dir=filepath, 
#                       label='val', 
#                       partition_id='A', 
#                       tiff_size=(600,600,140,1), 
#                       target_size=(600,600,128,1),
#                       local_filter=True,
#                       save_filtered=True)

filepath = '/media/sweene01/5e3122f3-5d00-4bc0-b099-982762bf3999/CH_TP_1_and_2_Large/all_data'
testfiles = os.listdir(filepath)
for file in range(len(testfiles)):
    testfiles[file] = os.path.join(filepath,testfiles[file])
plotter.run_mapping(cycle_gan_model, testfiles, args.INPUT_IMG_SIZE, filetext='VANGAN_', segmentation=True, stride=(25,25,1))

# Processing Predictions
# path = '/media/sweene01/5e3122f3-5d00-4bc0-b099-982762bf3999/Cycling_Hypoxia_Test/Predictions/'
# filteredpath = '/media/sweene01/5e3122f3-5d00-4bc0-b099-982762bf3999/Cycling_Hypoxia_Test/MedianFiltered_2x2x2_Predictions/'
# threshpath = '/media/sweene01/5e3122f3-5d00-4bc0-b099-982762bf3999/Cycling_Hypoxia_Test/MF_Thresh_Predictions/'
# files = os.listdir(path)
# for i in range(len(files)):
#     print(files[i])
#     file = os.path.join(path, files[i])
#     img = load_volume(file, datatype='float32', normalise=False)
#     img = ndimage.median_filter(img, size=2)

#     file = os.path.join(filteredpath, files[i])
#     io.imsave(file, img.astype('float32'), bigtiff=False)

#     thresh = 255. / 4.
#     img[img < thresh] = 0.
#     img[img >= thresh] = 255.
#     file = os.path.join(threshpath, files[i])
#     io.imsave(file, img.astype('uint8'), bigtiff=False)

# testpath = '/media/sweene01/5e3122f3-5d00-4bc0-b099-982762bf3999/VAN-GAN Paper/VANGAN_Final/All_Data_plusCH_NoPretraining/Sampling_Examples'
# args.INPUT_IMG_SIZE = (
#     1,
#     args.SUBVOL_PATCH_SIZE[0],
#     args.SUBVOL_PATCH_SIZE[1],
#     args.SUBVOL_PATCH_SIZE[2],
#     1,
# )

# for i in range(190,201,2):
#     print(i)
#     cycle_gan_model.load_checkpoint(epoch=i,
#                                     newpath='/media/sweene01/5e3122f3-5d00-4bc0-b099-982762bf3999/VAN-GAN Paper/VANGAN_Final/All_Data_plusCH_NoPretraining/cycleGAN_output/checkpoints')

#     # Make epoch folders
#     filepath = '/media/sweene01/5e3122f3-5d00-4bc0-b099-982762bf3999/VAN-GAN Paper/VANGAN_Final/All_Data_plusCH_NoPretraining/Sampling_Predictions_fakeRSOM'
#     folder = os.path.join(filepath, 'e{idx}'.format(idx=i))
#     if not os.path.isdir(folder):
#         os.makedirs(folder)
    
#     testfiles = os.listdir(testpath)
#     # filename = 'e{idx}_VANGAN_'.format(idx=i)
#     for file in range(len(testfiles)):
#         testfiles[file] = os.path.join(testpath,testfiles[file])
        
#     plotter.run_mapping(cycle_gan_model, testfiles, args.INPUT_IMG_SIZE, filetext='VANGAN_', segmentation=True, stride=(50,50,1), filepath=folder, padFactor=0.1)
