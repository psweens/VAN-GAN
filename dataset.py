import random
import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from utils import get_vacuum


class DatasetGen:
    def __init__(self, args, imaging_domain_data, seg_domain_data, strategy: tf.distribute.Strategy, otf_imaging=None):
        """ Setting shard policy for distributed dataset """
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

        ''' Setting parameters for below '''
        if args.DIMENSIONS == 2:
            self.imaging_output_shapes = (None, None, args.CHANNELS)
            self.segmentation_output_shapes = (None, None, 1)
            self.imaging_patch_shape = (args.GLOBAL_BATCH_SIZE,
                                        args.SUBVOL_PATCH_SIZE[0], args.SUBVOL_PATCH_SIZE[1], args.CHANNELS)
            self.segmentation_patch_shape = (args.SUBVOL_PATCH_SIZE[0], args.SUBVOL_PATCH_SIZE[1], 1)
        else:
            self.imaging_output_shapes = (None, None, None, args.CHANNELS)
            self.segmentation_output_shapes = (None, None, None, 1)
            self.imaging_patch_shape = (args.GLOBAL_BATCH_SIZE,
                                        args.SUBVOL_PATCH_SIZE[0], args.SUBVOL_PATCH_SIZE[1], args.SUBVOL_PATCH_SIZE[2],
                                        args.CHANNELS)
            self.segmentation_patch_shape = (
                args.SUBVOL_PATCH_SIZE[0], args.SUBVOL_PATCH_SIZE[1], args.SUBVOL_PATCH_SIZE[2], 1)

        self.strategy = strategy
        self.imaging_paths = imaging_domain_data
        self.segmentation_paths = seg_domain_data
        self.args = args
        self.otf_imaging = otf_imaging
        self.IMAGE_THRESH = 0.5
        self.SEG_THRESH = 0.8
        self.GLOBAL_BATCH_SIZE = args.GLOBAL_BATCH_SIZE
        self.SEGMENTATION_DIM = args.TARG_SYNTH_IMG_SIZE

        ''' Create datasets '''
        with self.strategy.scope():

            ''' Create imaging train dataset '''
            self.imaging_train_dataset = tf.data.Dataset.from_generator(lambda: self.imaging_datagen('training'),
                                                                        output_types=tf.float32,
                                                                        output_shapes=self.imaging_output_shapes)
            self.imaging_train_dataset = self.imaging_train_dataset.repeat()
            self.imaging_train_dataset = self.imaging_train_dataset.with_options(options)
            self.imaging_train_dataset = self.imaging_train_dataset.batch(self.GLOBAL_BATCH_SIZE, drop_remainder=True)
            self.imaging_train_dataset = self.imaging_train_dataset.map(self.process_imaging_domain,
                                                                        num_parallel_calls=tf.data.AUTOTUNE)

            ''' Create imaging validation dataset '''
            self.imaging_val_dataset = tf.data.Dataset.from_generator(lambda: self.imaging_datagen('validation'),
                                                                      output_types=tf.float32,
                                                                      output_shapes=self.imaging_output_shapes)
            self.imaging_val_dataset = self.imaging_val_dataset.repeat()
            self.imaging_val_dataset = self.imaging_val_dataset.with_options(options)
            self.imaging_val_dataset = self.imaging_val_dataset.batch(self.GLOBAL_BATCH_SIZE, drop_remainder=True)
            self.imaging_val_dataset = self.imaging_val_dataset.map(map_func=self.process_imaging_domain,
                                                                    num_parallel_calls=tf.data.AUTOTUNE)

            ''' Create segmentation train dataset '''
            self.segmentation_train_dataset = tf.data.Dataset.from_generator(
                lambda: self.segmentation_datagen('training'),
                output_types=tf.float32,
                output_shapes=self.segmentation_output_shapes)
            self.segmentation_train_dataset = self.segmentation_train_dataset.map(self.process_seg_domain,
                                                                                  num_parallel_calls=tf.data.AUTOTUNE)
            self.segmentation_train_dataset = self.segmentation_train_dataset.repeat()
            self.segmentation_train_dataset = self.segmentation_train_dataset.with_options(options)
            self.segmentation_train_dataset = self.segmentation_train_dataset.batch(self.GLOBAL_BATCH_SIZE,
                                                                                    drop_remainder=True)

            ''' Create segmentation validation dataset '''
            self.segmentation_val_dataset = tf.data.Dataset.from_generator(
                lambda: self.segmentation_datagen('validation'),
                output_types=tf.float32,
                output_shapes=self.segmentation_output_shapes)
            self.segmentation_val_dataset = self.segmentation_val_dataset.map(map_func=self.process_seg_domain,
                                                                              num_parallel_calls=tf.data.AUTOTUNE)
            self.segmentation_val_dataset = self.segmentation_val_dataset.repeat()
            self.segmentation_val_dataset = self.segmentation_val_dataset.with_options(options)
            self.segmentation_val_dataset = self.segmentation_val_dataset.batch(self.GLOBAL_BATCH_SIZE,
                                                                                drop_remainder=True)

            ''' Create validation dataset for full images (no sample cropping) '''
            self.imaging_val_full_vol_data = tf.data.Dataset.from_generator(self.imaging_val_datagen,
                                                                            output_types=(tf.float32, tf.int8))
            self.segmentation_val_full_vol_data = tf.data.Dataset.from_generator(self.segmentation_val_datagen,
                                                                                 output_types=(tf.float32, tf.int8))

            ''' Plot samples from training dataset '''
            self.plot_sample_dataset()

            ''' Zip training and validation datasets & setup to distribute across GPUs '''
            self.train_dataset = tf.data.Dataset.zip(
                (self.imaging_train_dataset,
                 self.segmentation_train_dataset)).prefetch(tf.data.AUTOTUNE)
            self.val_dataset = tf.data.Dataset.zip(
                (self.imaging_val_dataset,
                 self.segmentation_val_dataset)).prefetch(tf.data.AUTOTUNE)

            self.train_dataset = self.strategy.experimental_distribute_dataset(self.train_dataset)
            self.val_dataset = self.strategy.experimental_distribute_dataset(self.val_dataset)

    ''' Functions to gather imaging subvolumes '''

    def imaging_datagen(self, typ='training'):
        """
        Generates a batch of data from the imaging_paths directory.

        Args:
        - typ (str): The type of data to generate, either 'training' or 'validation'. Default is 'training'.

        Returns:
        - tensor: A tensor of shape [batch_size, height, width, channels] containing the batch of images.
        """
        iter_i = 0
        img_dataset = self.imaging_paths[typ]
        np.random.shuffle(img_dataset)
        while True:
            if iter_i >= math.floor(len(img_dataset) // self.args.GLOBAL_BATCH_SIZE):
                iter_i = 0
                np.random.shuffle(img_dataset)

            file = img_dataset[iter_i * self.args.GLOBAL_BATCH_SIZE:(iter_i + 1) * self.args.GLOBAL_BATCH_SIZE]

            # Load batch of full size images
            for idx, filename in enumerate(file):
                yield tf.convert_to_tensor(np.rot90(np.load(filename), np.random.choice([-1, 0, 1])), dtype=tf.float32)

            iter_i += 1

    def segmentation_datagen(self, typ='training'):
        """
        Generates a batch of data from the segmentation_paths directory.

        Args:
        - typ (str): The type of data to generate, either 'training' or 'validation'. Default is 'training'.

        Returns:
        - tensor: A tensor of shape [batch_size, height, width, channels] containing the batch of images.
        """
        iter_s = 0
        seg_dataset = self.segmentation_paths[typ]
        np.random.shuffle(seg_dataset)
        while True:
            if iter_s >= math.floor(len(seg_dataset) // self.args.GLOBAL_BATCH_SIZE):
                iter_s = 0
                np.random.shuffle(seg_dataset)

            file = seg_dataset[iter_s * self.args.GLOBAL_BATCH_SIZE:(iter_s + 1) * self.args.GLOBAL_BATCH_SIZE]

            # Load batch of full size images
            for idx, filename in enumerate(file):
                yield tf.convert_to_tensor(np.rot90(np.load(filename),
                                                    np.random.choice([-1, 0, 1])), dtype=tf.float32)

            iter_s += 1

    def imaging_val_datagen(self):
        while True:
            i = random.randint(0, self.imaging_paths['validation'].shape[0] - 1)
            yield tf.convert_to_tensor(np.load(self.imaging_paths['validation'][i]), dtype=tf.float32), i

    def segmentation_val_datagen(self):
        while True:
            i = random.randint(0, self.segmentation_paths['validation'].shape[0] - 1)
            yield tf.convert_to_tensor(np.load(self.segmentation_paths['validation'][i]), dtype=tf.float32), i

    ''' Functions for data preprocessing '''

    def process_imaging_domain(self, image):
        """ Standardizes image data and creates subvolumes """
        subvol = tf.image.random_crop(image, size=self.imaging_patch_shape)
        if self.otf_imaging is not None:
            subvol = self.otf_imaging(subvol)
        return subvol

    @tf.function
    def process_seg_domain(self, image):
        """
        Randomly crops the input_mask around a randomly selected feature voxel.

        Args:
            self:
            image (tf.Tensor): The 4D input segmentation mask (depth, width, length, channel).
                                   Features are labeled as 1, background as -1.

        Returns:
            cropped_mask (tf.Tensor): The randomly cropped segmentation mask.
        """

        # Get the indices of feature voxels
        feature_indices = tf.where(tf.equal(image, 1))

        # Randomly select a feature voxel
        random_feature_index = tf.cast(tf.random.shuffle(feature_indices)[0], tf.int32)

        # Calculate the cropping window based on the selected feature voxel
        crop_start_depth = random_feature_index[0] - self.segmentation_patch_shape[0] // 2
        crop_start_width = random_feature_index[1] - self.segmentation_patch_shape[1] // 2
        crop_start_length = random_feature_index[2] - self.segmentation_patch_shape[2] // 2

        # Calculate crop_end coordinates symmetrically based on image dimensions
        crop_end_depth = crop_start_depth + self.segmentation_patch_shape[0]
        crop_end_width = crop_start_width + self.segmentation_patch_shape[1]
        crop_end_length = crop_start_length + self.segmentation_patch_shape[2]

        image_shape = tf.shape(image)

        # Adjust cropping symmetrically if necessary
        if crop_start_depth < 0:
            crop_end_depth -= crop_start_depth
            crop_start_depth = 0
        elif crop_end_depth > image_shape[0]:
            crop_start_depth -= crop_end_depth - image_shape[0]
            crop_end_depth = image_shape[0]

        if crop_start_width < 0:
            crop_end_width -= crop_start_width
            crop_start_width = 0
        elif crop_end_width > image_shape[1]:
            crop_start_width -= crop_end_width - image_shape[1]
            crop_end_width = image_shape[1]

        if crop_start_length < 0:
            crop_end_length -= crop_start_length
            crop_start_length = 0
        elif crop_end_length > image_shape[2]:
            crop_start_length -= crop_end_length - image_shape[2]
            crop_end_length = image_shape[2]

        # Crop the tensor
        cropped_mask = image[crop_start_depth:crop_end_depth,
                       crop_start_width:crop_end_width,
                       crop_start_length:crop_end_length, :]

        return cropped_mask

    def plot_sample_dataset(self):
        """
        Plots a sample of the input datasets 'Imaging' and 'Segmentation' along with their histograms.
        The function saves a 3D TIFF file of the input data.

        Args:
        - self.imaging_train_dataset: Dataset A.
        - self.segmentation_train_dataset: Dataset B.
        - self.args.DIMENSIONS: Dimensionality of the input data.
        - self.args.SUBVOL_PATCH_SIZE: Size of the subvolume patch.

        Returns:
        - None
        """

        #  Visualise some examples
        if self.args.DIMENSIONS == 2:
            nfig = 1
        else:
            nfig = 6

        fig, axs = plt.subplots(nfig + 1, 2, figsize=(10, 15))
        fig.subplots_adjust(hspace=0.5)
        for i, samples in enumerate(zip(self.imaging_train_dataset.take(1), self.segmentation_train_dataset.take(1))):

            dI = samples[0][0].numpy()
            dS = samples[1][0].numpy()
            if self.args.DIMENSIONS == 3:
                ''' Save 3D images '''
                io.imsave("./GANMonitor/Imaging_Test_Input.tiff",
                          np.transpose(dI, (2, 0, 1, 3)),
                          bigtiff=False, check_contrast=False)

                io.imsave("./GANMonitor/Segmentation_Test_Input.tiff",
                          np.transpose(dS, (2, 0, 1, 3)),
                          bigtiff=False, check_contrast=False)

            if self.args.DIMENSIONS == 2:
                showI = (dI * 127.5 + 127.5).astype('uint8')
                showS = (dS * 127.5 + 127.5).astype('uint8')
                axs[0, 0].imshow(showI, cmap='gray')
                axs[0, 1].imshow(showS, cmap='gray')
            else:
                for j in range(0, nfig):
                    showI = (dI[:, :, j * int(self.args.SUBVOL_PATCH_SIZE[2] / nfig), ])
                    showS = (dS[:, :, j * int(self.args.SUBVOL_PATCH_SIZE[2] / nfig), ])
                    axs[j, 0].imshow(showI, cmap='gray')
                    axs[j, 1].imshow(showS, cmap='gray')

            ''' Include histograms '''
            axs[nfig, 0].hist(dI.ravel(), bins=256, range=(np.amin(dI), np.amax(dI)), fc='k', ec='k', density=True)
            axs[nfig, 1].hist(dS.ravel(), bins=256, range=(np.amin(dS), np.amax(dS)), fc='k', ec='k', density=True)

            # Set axis labels
            axs[0, 0].set_title('Imaging Dataset Example (XY Slices)')
            axs[0, 1].set_title('Segmentation Dataset Example (XY Slices)')
            axs[nfig, 0].set_ylabel('Voxel Frequency')
            plt.show(block=False)
            plt.close()

            if self.args.DIMENSIONS == 3:
                _, axs = plt.subplots(nfig, 2, figsize=(10, 15))
                for j in range(0, nfig):
                    showI = dI[:, j * int(self.args.SUBVOL_PATCH_SIZE[1] / nfig), :, 0]
                    showS = dS[:, j * int(self.args.SUBVOL_PATCH_SIZE[1] / nfig), :self.args.SUBVOL_PATCH_SIZE[2] - 1,
                            0]
                    axs[j, 0].imshow(showI, cmap='gray')
                    axs[j, 1].imshow(showS, cmap='gray')

            # Set axis labels
            axs[0, 0].set_title('Imaging Dataset Example (YZ Slices)')
            axs[0, 1].set_title('Segmentation Dataset Example (YZ Slices)')
            plt.show(block=False)
            plt.close()
