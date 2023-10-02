import random
import math
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from utils import get_vacuum, fast_clahe, clahe_3d


class DatasetGen:
    def __init__(self,
                 args,
                 imaging_domain_data,
                 seg_domain_data,
                 strategy: tf.distribute.Strategy,
                 otf_imaging=None,
                 semi_supervised_dir=None):
        """ Setting shard policy for distributed dataset """
        self.feature_indices = None
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

        ''' Setting parameters for below '''
        if args.DIMENSIONS == 2:
            self.imaging_output_shapes = (None, None, args.CHANNELS)
            self.segmentation_output_shapes = (None, None, 1)
            self.imaging_patch_shape = (args.SUBVOL_PATCH_SIZE[0], args.SUBVOL_PATCH_SIZE[1], args.CHANNELS)
            self.segmentation_patch_shape = (args.SUBVOL_PATCH_SIZE[0], args.SUBVOL_PATCH_SIZE[1], 1)
        else:
            self.imaging_output_shapes = (None, None, None, args.CHANNELS)
            self.segmentation_output_shapes = (None, None, None, 1)
            self.imaging_patch_shape = (args.SUBVOL_PATCH_SIZE[0], args.SUBVOL_PATCH_SIZE[1], args.SUBVOL_PATCH_SIZE[2],
                                        args.CHANNELS)
            self.segmentation_patch_shape = (
                args.SUBVOL_PATCH_SIZE[0], args.SUBVOL_PATCH_SIZE[1], args.SUBVOL_PATCH_SIZE[2], 1)

        self.strategy = strategy
        self.imaging_paths = imaging_domain_data
        self.segmentation_paths = seg_domain_data
        self.args = args
        self.otf_imaging = otf_imaging
        if semi_supervised_dir is not None:
            self.semi_supervised = True
            self.semi_supervised_dir = semi_supervised_dir
        else:
            self.semi_supervised = False
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
            self.imaging_train_dataset = self.imaging_train_dataset.map(self.process_imaging_domain,
                                                                        num_parallel_calls=tf.data.AUTOTUNE)
            self.imaging_train_dataset = self.imaging_train_dataset.batch(self.GLOBAL_BATCH_SIZE, drop_remainder=True)
            if self.otf_imaging is not None:
                self.imaging_train_dataset = self.imaging_train_dataset.map(self.otf_imaging,
                                                                            num_parallel_calls=tf.data.AUTOTUNE)

            ''' Create imaging validation dataset '''
            self.imaging_val_dataset = tf.data.Dataset.from_generator(lambda: self.imaging_datagen('validation'),
                                                                      output_types=tf.float32,
                                                                      output_shapes=self.imaging_output_shapes)
            self.imaging_val_dataset = self.imaging_val_dataset.repeat()
            self.imaging_val_dataset = self.imaging_val_dataset.with_options(options)
            self.imaging_val_dataset = self.imaging_val_dataset.map(self.process_imaging_domain,
                                                                    num_parallel_calls=tf.data.AUTOTUNE)
            self.imaging_val_dataset = self.imaging_val_dataset.batch(self.GLOBAL_BATCH_SIZE, drop_remainder=True)
            if self.otf_imaging is not None:
                self.imaging_val_dataset = self.imaging_val_dataset.map(self.otf_imaging,
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

            start_idx = iter_i * self.args.GLOBAL_BATCH_SIZE
            end_idx = (iter_i + 1) * self.args.GLOBAL_BATCH_SIZE

            if end_idx > len(img_dataset):
                end_idx = len(img_dataset)

            file = img_dataset[start_idx:end_idx]

            # Load batch of full size images
            for idx, filename in enumerate(file):
                yield tf.convert_to_tensor(np.load(filename), dtype=tf.float32)

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
                if self.semi_supervised:
                    ss_filename = os.path.join(self.semi_supervised_dir, os.path.basename(filename))
                    yield tf.convert_to_tensor(np.concatenate((np.load(filename),
                                                               np.load(ss_filename)),
                                                              axis=0),
                                               dtype=tf.float32)
                else:
                    yield tf.convert_to_tensor(np.load(filename), dtype=tf.float32)

            iter_s += 1

    def imaging_val_datagen(self):
        while True:
            i = random.randint(0, len(self.imaging_paths['validation']) - 1)
            yield tf.convert_to_tensor(np.load(self.imaging_paths['validation'][i]), dtype=tf.float32), i

    def segmentation_val_datagen(self):
        while True:
            i = random.randint(0, len(self.segmentation_paths['validation']) - 1)
            yield tf.convert_to_tensor(np.load(self.segmentation_paths['validation'][i]), dtype=tf.float32), i

    ''' Functions for data preprocessing '''

    @tf.function
    def random_spatial_augmentation(self, image, max_rotation_angle=180, preserve_depth_orientation=False):
        # Randomly flip horizontally
        image = tf.cond(tf.random.uniform(()) > 0.5, lambda: tf.image.flip_left_right(image), lambda: image)

        # Randomly flip vertically
        image = tf.cond(tf.random.uniform(()) > 0.5, lambda: tf.image.flip_up_down(image), lambda: image)

        if not preserve_depth_orientation:
            # Randomly rotate the image
            rotation_angle = tf.random.uniform((), minval=-max_rotation_angle, maxval=max_rotation_angle) * (
                        math.pi / 180.0)
            image = tf.image.rot90(image, k=tf.cast(rotation_angle // 90, dtype=tf.int32))

        return image

    def process_imaging_domain(self, image):
        """ Standardizes image data and creates subvolumes """
        # subvol = tf.image.random_crop(image, size=self.imaging_patch_shape)
        # if self.otf_imaging is not None:
        #     subvol = self.otf_imaging(subvol)
        arr = tf.image.random_crop(image, size=self.imaging_patch_shape)
        # arr = clahe_3d(arr)
        return self.random_spatial_augmentation(arr, preserve_depth_orientation=True)

    @tf.function
    def process_seg_domain(self, image):
        # Initialize a loop counter
        i = tf.constant(0)

        # Define the maximum number of iterations
        max_iterations = tf.constant(200)

        # Initialize arr
        arr = tf.image.random_crop(image, size=self.segmentation_patch_shape)

        # Start a while loop
        def condition(i, arr):
            return tf.math.logical_and(i < max_iterations, tf.math.reduce_max(arr) < self.SEG_THRESH)

        def body(i, _):
            # Generate a new random crop from the original image
            new_arr = tf.image.random_crop(image, size=self.segmentation_patch_shape)
            return i + 1, new_arr

        _, arr = tf.while_loop(condition, body, [i, arr])

        return self.random_spatial_augmentation(arr)

    # @tf.function
    # def process_imaging_domain(self, image):
    #     # Initialize a loop counter
    #     i = tf.constant(0)
    #
    #     # Define the maximum number of iterations
    #     max_iterations = tf.constant(10)
    #
    #     # Initialize arr
    #     arr = tf.image.random_crop(image, size=self.imaging_patch_shape)
    #
    #     # Start a while loop
    #     def condition(i, arr):
    #         return tf.math.logical_and(i < max_iterations, tf.math.reduce_max(arr) < 0.)
    #
    #     def body(i, _):
    #         # Generate a new random crop from the original image
    #         new_arr = tf.image.random_crop(image, size=self.imaging_patch_shape)
    #         return i + 1, new_arr
    #
    #     _, arr = tf.while_loop(condition, body, [i, arr])
    #
    #     return self.random_spatial_augmentation(arr)

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

        if self.semi_supervised:
            fig, axs = plt.subplots(nfig + 1, 3, figsize=(10, 15))
        else:
            fig, axs = plt.subplots(nfig + 1, 2, figsize=(10, 15))
        fig.subplots_adjust(hspace=0.5)
        for i, samples in enumerate(zip(self.imaging_train_dataset.take(1), self.segmentation_train_dataset.take(1))):

            dI = samples[0][0].numpy()
            dS = samples[1][0].numpy()
            if self.semi_supervised:
                dIS = dS[self.segmentation_patch_shape[0]:, ]
                dS = dS[:self.segmentation_patch_shape[0], ]
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
                    showI = (dI[:, :, j * int(self.segmentation_patch_shape[2] / nfig), ])
                    showS = (dS[:, :, j * int(self.segmentation_patch_shape[2] / nfig), ])
                    axs[j, 0].imshow(showI, cmap='gray')
                    axs[j, 1].imshow(showS, cmap='gray')
                    if self.semi_supervised:
                        showIS = (dIS[:, :, j * int(self.segmentation_patch_shape[2] / nfig), ])
                        axs[j, 2].imshow(showIS, cmap='gray')

            ''' Include histograms '''
            axs[nfig, 0].hist(dI.ravel(), bins=256, range=(np.amin(dI), np.amax(dI)), fc='k', ec='k', density=True)
            axs[nfig, 1].hist(dS.ravel(), bins=256, range=(np.amin(dS), np.amax(dS)), fc='k', ec='k', density=True)
            if self.semi_supervised:
                axs[nfig, 2].hist(dIS.ravel(), bins=256, range=(np.amin(dIS), np.amax(dIS)), fc='k', ec='k',
                                  density=True)

            # Set axis labels
            axs[0, 0].set_title('Imaging Dataset (XY)')
            axs[0, 1].set_title('Segmentation Dataset (XY)')
            if self.semi_supervised:
                axs[0, 2].set_title('Paired Imaging Dataset (XY)')
            axs[nfig, 0].set_ylabel('Voxel Frequency')
            plt.show(block=False)
            plt.close()

            if self.args.DIMENSIONS == 3:
                if self.semi_supervised:
                    _, axs = plt.subplots(nfig, 3, figsize=(10, 15))
                else:
                    _, axs = plt.subplots(nfig, 2, figsize=(10, 15))
                for j in range(0, nfig):
                    showI = dI[:, j * int(self.segmentation_patch_shape[1] / nfig), :, ]
                    showS = dS[:, j * int(self.segmentation_patch_shape[1] / nfig),
                            :self.args.SUBVOL_PATCH_SIZE[2] - 1, ]
                    axs[j, 0].imshow(showI, cmap='gray')
                    axs[j, 1].imshow(showS, cmap='gray')
                    if self.semi_supervised:
                        showIS = dIS[:, j * int(self.segmentation_patch_shape[1] / nfig),
                                 :self.args.SUBVOL_PATCH_SIZE[2] - 1, ]
                        axs[j, 2].imshow(showIS, cmap='gray')

            # Set axis labels
            axs[0, 0].set_title('Imaging Dataset (YZ)')
            axs[0, 1].set_title('Segmentation Dataset (YZ)')
            if self.semi_supervised:
                axs[0, 2].set_title('Paired Dataset (YZ)')
            plt.show(block=False)
            plt.close()
