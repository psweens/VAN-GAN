import os
import shutil
import glob
import argparse
from types import SimpleNamespace

import numpy as np
import tensorflow as tf

tf.keras.backend.clear_session()

#tf.debugging.set_log_device_placement(True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

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

from time import time
from vangan import VanGan, train
from custom_callback import GanMonitor
from dataset import DatasetGen
from paired_dataset import PairedDatasetGen
from preprocessing import DataPreprocessor
from preprocess_modality import preprocess_rsom, preprocess_tplsm, preprocess_hrem, preprocess_lsm, \
    preprocess_retinal_image
from tb_callback import TB_Summary
from unet_training import train_unet, segment_volumes_with_unet
from utils import save_args
from post_training import epoch_sweep

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

summary = TB_Summary(tensorboardDir, len(physical_devices))  # Initialise TensorBoard summary helper
print(physical_devices)
''' SET PARAMETERS '''
print('*** Setting VANGAN parameters ***')
cli_parser = argparse.ArgumentParser(description='Training entry point')
cli_parser.add_argument('--architecture', choices=['vangan', 'unet'], default='vangan',
                        help='Select which architecture to train (unsupervised VAN-GAN or supervised 3D U-Net).')
cli_parser.add_argument('--unet_infer_dir', default=None,
                        help='Optional path to a directory or HDF5 file to segment after U-Net training. '
                             'Defaults to the imaging testing partition if not provided.')
cli_parser.add_argument('--unet_infer_stride', type=int, nargs='*', default=None,
                        help='Sliding-window stride for U-Net inference. Provide one value or one per spatial '
                             'dimension. Defaults to the training patch size when omitted.')
cli_parser.add_argument('--unet_threshold', type=float, default=0.5,
                        help='Probability threshold applied to the U-Net output when generating binary masks.')
cli_args = cli_parser.parse_args()

args = SimpleNamespace()
args.ARCHITECTURE = cli_args.architecture
args.output_dir = '/mnt/sda/VG_Output'
os.makedirs(args.output_dir, exist_ok=True)
args.N_DEVICES = len(physical_devices)
args.BUFFER_SIZE = 256
args.MIN_PIXEL_VALUE = -1.0
args.MAX_PIXEL_VALUE = 0.8

# Training parameters
args.EPOCHS = 250
args.BATCH_SIZE = 1
args.GLOBAL_BATCH_SIZE = args.N_DEVICES * args.BATCH_SIZE
args.PREFETCH_SIZE = 1
args.INITIAL_LR = 2e-4  # Learning rate
args.INITIATE_LR_DECAY = 200  #int(0.5 * args.EPOCHS)  # Set start of learning rate decay to 0
args.NO_NOISE = 110# args.EPOCHS  # Set when discriminator noise decays to 0

# Image parameters
args.SURFACE_ILLUMINATION = True
args.CHANNELS = 1
args.DIMENSIONS = 3
args.RAW_IMG_SIZE = (600, 600, 140, args.CHANNELS)  # Unprocessed imaging domain image dimensions
args.TARG_RAW_IMG_SIZE = (600, 600, 140, args.CHANNELS)  # Target size if downsampling
args.SYNTH_IMG_SIZE = (512, 512, 140)  # Unprocessed segmentation domain image dimensions
args.TARG_SYNTH_IMG_SIZE = (512, 512, 140)  # Target size if downsampling
args.SUBVOL_PATCH_SIZE = (128, 128, 128)  # Size of subvolume to be trained on
# Optional inference overrides for supervised U-Net runs
if cli_args.unet_infer_stride:
    stride_values = [int(val) for val in cli_args.unet_infer_stride]
    if len(stride_values) == 1:
        args.UNET_INFER_STRIDE = tuple([stride_values[0]] * args.DIMENSIONS)
    elif len(stride_values) == args.DIMENSIONS:
        args.UNET_INFER_STRIDE = tuple(stride_values)
    else:
        raise ValueError('`--unet_infer_stride` expects either one value or one value per spatial dimension.')
else:
    args.UNET_INFER_STRIDE = None
args.UNET_INFER_PATH = cli_args.unet_infer_dir
args.UNET_INFER_THRESHOLD = float(cli_args.unet_threshold)
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
args.RESOLUTION = 20 # 20 um resolution
args.MIN_VESSEL_DIAMETER = 20 # Minimum vessel diameter of imaged tissue

# Set callback parameters
args.PERIOD_2D_CALLBACK = 2  # Period of epochs to output a 2D validation dataset example
args.PERIOD_3D_CALLBACK = 2  # Period of epochs to output a 3D validation dataset example

'''' PREPROCESSING '''
imaging_data = DataPreprocessor(args,
                                raw_path='/mnt/sdb/HIPCT/raw_data/train/all_image_subvolumes/',
                                main_dir='/mnt/sdb/HIPCT/',
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

# Perform any preprocessing of images if neccessary
# imaging_data.preprocess(preprocess_fn=preprocess_hrem,
#                         save_filtered=True,
#                         resize=False)
# synth_data.preprocess(resize=True,
#                       save_filtered=False)

# Load dataset partitions
imaging_data.load_partition('/mnt/sdb/3DcycleGAN_simLNet_LNet/dataA_partition.pkl')
# imaging_data.load_partition('/mnt/sda/CH_training_dataset/dataA_partition.pkl')
# synth_data.load_partition('/mnt/sda/CH_training_dataset/dataB_partition.pkl')
# synth_data.partition['training'] = os.listdir('/mnt/sda/3DcycleGAN_simLNet_LNet/trainB')
# imaging_data.load_partition('/mnt/sdb/LD_Lightsheet/VG/dataA_partition.pkl')
# imaging_data.load_partition('/mnt/sda/VS-GAN_deepVess/dataA_partition.pkl')
# imaging_data.load_partition('/mnt/sda/VAN-GAN_HREM/dataA_partition.pkl')
# imaging_data.load_partition('/mnt/sdb/HIPCT/dataA_partition.pkl')
# imaging_data.load_partition('/mnt/sda/RSOM_EVB_Only/dataA_partition.pkl')
# imaging_data.load_partition('/mnt/sda/RSOM_VG_080125/dataA_partition.pkl')
# imaging_data.load_partition('/mnt/sda/RSOM_VG_220524/dataA_partition.pkl')
synth_data.load_partition('/mnt/sdb/3DcycleGAN_simLNet_LNet/dataB_partition.pkl')
# imaging_data.load_partition('/mnt/sdb/VG_Retinal_Dataset/dataA_partition.pkl')
# synth_data.load_partition('/mnt/sdb/VG_Retinal_Dataset/dataB_partition.pkl')

''' ENSURE DATASET SIZES ARE DIVISIBLE BY NUMBER OF GPUS '''
def adjust_dataset_partition(partition, n_devices):
    """
    Adjusts dataset partitions (train, val, test) so that their sizes are divisible by the number of GPUs.
    Removes excess elements from the lists.
    """
    for key in ['training', 'validation', 'testing']:
        if key in partition:
            size = len(partition[key])
            remainder = size % n_devices
            if remainder != 0:
                print(f"Adjusting {key} set from {size} to {size - remainder} to match {n_devices} GPUs.")
                partition[key] = partition[key][:size - remainder]


# Apply adjustment to training, validation, and testing sets
adjust_dataset_partition(imaging_data.partition, args.N_DEVICES)
adjust_dataset_partition(synth_data.partition, args.N_DEVICES)

''' GENERATE TENSORFLOW DATASETS '''
print('*** Generating datasets for model ***')
# Define function to preprocess imaging domain image on the fly (otf)
# Min/max batch normalisation and rescaling to [-1,1] shown here
@tf.function
def process_imaging_otf(tensor, axis=None, keepdims=True):

    # Calculate the maximum and minimum values along the batch dimension
    max_vals = tf.reduce_max(tensor, axis=axis, keepdims=keepdims)
    min_vals = tf.reduce_min(tensor, axis=axis, keepdims=keepdims)

    # Normalize the tensor between -1 and 1
    return 2.0 * (tensor - min_vals) / (max_vals - min_vals + 1.e-8) - 1.0


# OTF image-wise batch normalisation
# @tf.function
# def process_imaging_otf(tensor, axis=None, keepdims=True):
#
#     # Calculate the maximum and minimum values along the batch dimension
#     mean_vals = tf.reduce_mean(tensor, axis=axis, keepdims=keepdims)
#     std_vals = tf.math.reduce_std(tensor, axis=axis, keepdims=keepdims)
#
#     return (tensor - mean_vals) / std_vals

# process_imaging_otf = None

# Define dataset class
if args.ARCHITECTURE == 'unet':
    paired_dataset = PairedDatasetGen(
        args=args,
        imaging_paths=imaging_data.partition,
        segmentation_paths=synth_data.partition,
        strategy=strategy,
        otf_imaging=process_imaging_otf,
    )
else:
    getDataset = DatasetGen(args=args,
                            imaging_paths=imaging_data.partition,
                            segmentation_paths=synth_data.partition,
                            strategy=strategy,
                            otf_imaging=process_imaging_otf,  # Set to None if OTF processing not needed
                            surface_illumination=args.SURFACE_ILLUMINATION
                            # semi_supervised_dir='/mnt/sda/3DcycleGAN_simLNet_LNet/all_A_data'
                            )

''' CALCULATE NUMBER OF TRAINING / VALIDATION STEPS '''
if args.ARCHITECTURE == 'unet':
    args.train_steps = paired_dataset.train_steps
    args.val_steps = paired_dataset.val_steps
else:
    args.train_steps = int(np.amax([len(imaging_data.partition['training']),
                                    len(synth_data.partition['training'])]) / args.GLOBAL_BATCH_SIZE)

    args.val_steps = int(np.round(np.amax([len(imaging_data.partition['validation']),
                                           len(synth_data.partition['validation'])]) / args.GLOBAL_BATCH_SIZE))
    if args.val_steps < 1.:
        args.val_steps = int(1)

if args.ARCHITECTURE == 'unet':
    print('*** Training supervised 3D U-Net ***')
    save_args(args, os.path.join(args.output_dir, 'Args_Settings.txt'))
    unet_result = train_unet(args, strategy, paired_dataset)

    def _resolve_inference_targets(path_hint, default_list):
        if path_hint:
            if os.path.isdir(path_hint):
                entries = [
                    os.path.join(path_hint, entry)
                    for entry in sorted(os.listdir(path_hint))
                    if entry.lower().endswith(('.h5', '.hdf5'))
                ]
                return [p for p in entries if os.path.exists(p)]
            if os.path.isfile(path_hint):
                return [path_hint]
            print(f'Warning: `--unet_infer_dir` path not found: {path_hint}')
            return []
        if not default_list:
            return []
        valid_defaults = [p for p in default_list if os.path.exists(p)]
        missing = set(default_list) - set(valid_defaults)
        for missing_path in sorted(missing):
            print(f'Warning: testing partition file not found for inference: {missing_path}')
        return valid_defaults

    inference_targets = _resolve_inference_targets(
        args.UNET_INFER_PATH,
        imaging_data.partition.get('testing'),
    )

    if not inference_targets:
        print('No volumes were provided for U-Net inference; skipping post-training segmentation step.')
        raise SystemExit

    label_map = None
    if args.UNET_INFER_PATH is None:
        testing_segmentations = synth_data.partition.get('testing')
        if testing_segmentations:
            seg_lookup = {
                os.path.splitext(os.path.basename(seg_path))[0]: seg_path
                for seg_path in testing_segmentations
            }
            label_map = {}
            for img_path in inference_targets:
                base = os.path.splitext(os.path.basename(img_path))[0]
                match = seg_lookup.get(base)
                if match:
                    label_map[base] = match
            if not label_map:
                label_map = None

    prediction_dir = os.path.join(args.output_dir, 'unet_predictions')
    inference_results = segment_volumes_with_unet(
        args,
        unet_result.model,
        inference_targets,
        prediction_dir,
        stride=args.UNET_INFER_STRIDE,
        threshold=args.UNET_INFER_THRESHOLD,
        label_map=label_map,
    )

    print(f"Best U-Net checkpoint saved to: {unet_result.best_checkpoint}")
    for source_path, info in inference_results.items():
        dice_info = ''
        if info.get('dice') is not None:
            dice_info = f", Dice={info['dice']:.4f}"
        print(f"Segmented {source_path} -> {info['prediction_path']}{dice_info}")
    raise SystemExit

''' DEFINE VANGAN '''
vangan_model = VanGan(args, strategy=strategy, semi_supervised=False)

''' DEFINE CUSTOM CALLBACK '''
plotter = GanMonitor(args,
                     dataset=getDataset,
                     imaging_val_data=imaging_data.partition['validation'],
                     segmentation_val_data=synth_data.partition['validation'],
                     process_imaging_domain=process_imaging_otf,
                     surface_illumination=args.SURFACE_ILLUMINATION
                     )

# Save args to txt file
save_args(args, os.path.join(args.output_dir, 'Args_Settings.txt'))

''' TRAIN VAN-GAN MODEL '''
# vangan_model.load_checkpoint(epoch=240, newpath='/mnt/sda/VGp_Paper/PA_Synth_Ablation_Study/Cycle_MS_SSIM_cldice_ID_cldice/checkpoints/')
# vangan_model.load_checkpoint(epoch=150)
for epoch in range(args.EPOCHS):
    print(f'\nEpoch {epoch + 1:03d}/{args.EPOCHS:03d}')
    vangan_model.current_epoch.assign(epoch + 1)
    start = time()

    # Set shared_cycle to True after epoch 100
    if (epoch + 1) >= 100:
        vangan_model.shared_cycle.assign(True)  # Assigning Boolean value

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
        # if epoch > 100:
        vangan_model.save_checkpoint(epoch=epoch)

    end = time()
    summary.scalar('elapse', end - start, epoch=epoch, training=True)


''' CREATE VANGAN PREDICTIONS '''
# Predict segmentation probability maps for imaging test dataset
# plotter.run_mapping(vangan_model, imaging_data.partition['training'], args.INPUT_IMG_SIZE, filetext='VANGAN_',
#                     filepath=args.output_dir, segmentation=True)
# Prediction fake imaging data using synthetic segmentation test dataset
# plotter.run_mapping(vangan_model, synth_data.partition['testing'], args.INPUT_IMG_SIZE, filetext='VANGAN_',
#                     filepath=args.output_dir, segmentation=False, stride=(32, 32, 32))


''' SEGMENTING NEW IMAGES '''
# Alternatively, to run VANGAN on a directory of images (saved as .npy) using the following example script
# new_imaging_data = DataPreprocessor(args=args)  # Create data preprocessor
# new_imaging_data.data_type = 'float32'
# new_imaging_data.process_new_data(current_path='/mnt/sda/EVB/Phase 6 - AA RT_Isotropic/',
#                                   new_path='/mnt/sda/EVB/Phase 6 - AA RT_Isotropic_VG_Preprocessed/',
#                                   preprocess_fn=preprocess_rsom,
#                                   tiff_size=args.RAW_IMG_SIZE,
#                                   target_size=args.TARG_RAW_IMG_SIZE,
#                                   resize=False)

# ''' TESTING PREDICTIONS ACROSS EPOCHS '''
# epoch_sweep(args,
#             vangan_model,
#             plotter,
#             test_path='/mnt/sdb/3DcycleGAN_simLNet_LNet/epoch_sweep/',  # Can use imaging_data.partition['testing']
#             start=100,
#             end=250,
#             segmentation=True  # Set to False if fake imaging is wanted
#             )

# vangan_model.load_checkpoint(epoch=250)
filepath = '/mnt/sdb/3DcycleGAN_simLNet_LNet/all_data_A/'
img_files = os.listdir(filepath)
for file in range(len(img_files)):
    img_files[file] = os.path.join(filepath, img_files[file])
plotter.run_mapping(vangan_model, img_files, args.INPUT_IMG_SIZE, filetext='VANGAN_', filepath=args.output_dir,
                    segmentation=True)
