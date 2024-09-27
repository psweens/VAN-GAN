import random
import math
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from utils import get_vacuum, fast_clahe, clahe_3d, min_max_norm_tf, sobel_edge_3d


class DatasetGen:
    def __init__(self,
                 args,
                 imaging_domain_data,
                 seg_domain_data,
                 strategy: tf.distribute.Strategy,
                 otf_imaging=None,
                 surface_illumination=False,
                 semi_supervised_dir=None):
        """ Setting shard policy for distributed dataset """
        self.feature_indices = None
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

        ''' Setting parameters for below '''
        if semi_supervised_dir is not None:
            self.semi_supervised = True
            self.semi_supervised_dir = semi_supervised_dir
            self.ss_patch_shape = (
                args.SUBVOL_PATCH_SIZE[0], args.SUBVOL_PATCH_SIZE[1], 2 * args.SUBVOL_PATCH_SIZE[2], 1)
        else:
            self.semi_supervised = False
        self.DIMENSIONS = args.DIMENSIONS
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
            if self.semi_supervised:
                self.segmentation_patch_shape = (
                    args.SUBVOL_PATCH_SIZE[0], args.SUBVOL_PATCH_SIZE[1], args.SUBVOL_PATCH_SIZE[2], 1)
            else:
                self.segmentation_patch_shape = (
                    args.SUBVOL_PATCH_SIZE[0], args.SUBVOL_PATCH_SIZE[1], args.SUBVOL_PATCH_SIZE[2], 1)

        self.strategy = strategy
        self.imaging_paths = imaging_domain_data
        self.segmentation_paths = seg_domain_data
        self.args = args
        self.otf_imaging = otf_imaging
        self.surface_illumination = surface_illumination
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
            # self.segmentation_train_dataset = self.segmentation_train_dataset.map(self.return_black,
            #                                                                    num_parallel_calls=tf.data.AUTOTUNE)
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
                # if np.random.rand() < 0.01:  # 5% chance to yield negative zeros
                #     # Generate array of negative zeros of predefined size
                #     zeros_array = np.zeros(self.segmentation_patch_shape) - 1.0  # Creating negative zeros
                #     yield tf.convert_to_tensor(zeros_array, dtype=tf.float32)
                # else:
                if self.semi_supervised:
                    ss_filename = os.path.join(self.semi_supervised_dir, os.path.basename(filename))
                    yield tf.convert_to_tensor(np.concatenate((np.load(filename),
                                                               np.load(ss_filename)),
                                                              axis=2),
                                               dtype=tf.float32)
                else:
                    yield tf.convert_to_tensor(np.load(filename), dtype=tf.float32)

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
    def random_intensity_augmentation(self, image, brightness_delta=0.2, contrast_range=(0.7, 1.3)):

        # Apply random brightness, contrast, and saturation adjustments
        image = tf.image.random_brightness(image, max_delta=brightness_delta)
        image = tf.image.random_contrast(image, lower=contrast_range[0], upper=contrast_range[1])

        return image

    def is_2d(self):
        return self.DIMENSIONS == 2

    @tf.function
    def random_rotate(self, image, preserve_z_axis=False):
        if self.is_2d():
            arr = tf.image.random_flip_left_right(image)
            return tf.image.rot90(arr, k=tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32))
        else:
            return self.random_rotate_3d(image, preserve_z_axis=preserve_z_axis)

    @tf.function
    def random_rotate_3d(self, image, preserve_z_axis):

        # Randomly decide whether to perform augmentation
        if not tf.random.uniform(()) > 0.5:
            return image

        if preserve_z_axis:
            # Rotate only around the z-axis
            num_rotations = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
            image = tf.transpose(image, perm=[2, 1, 0, 3])  # (z, y, x, c)
            image = tf.image.rot90(image, k=num_rotations)
            image = tf.transpose(image, perm=[2, 1, 0, 3])  # Back to (x, y, z, c)
        else:
            # Randomly select a rotation axis: 0 for x, 1 for y, 2 for z
            rotation_axis = tf.random.uniform((), minval=0, maxval=3, dtype=tf.int32)
            num_rotations = tf.random.uniform((), minval=1, maxval=4, dtype=tf.int32)

            if rotation_axis == 0:
                # Rotate around x-axis
                image = tf.transpose(image, perm=[0, 2, 1, 3])  # (x, z, y, c)
                image = tf.image.rot90(image, k=num_rotations)
                image = tf.transpose(image, perm=[0, 2, 1, 3])  # Back to (x, y, z, c)
            elif rotation_axis == 1:
                # Rotate around y-axis
                image = tf.transpose(image, perm=[1, 2, 0, 3])  # (y, z, x, c)
                image = tf.image.rot90(image, k=num_rotations)
                image = tf.transpose(image, perm=[2, 0, 1, 3])  # Back to (x, y, z, c)
            elif rotation_axis == 2:
                # Rotate around z-axis
                image = tf.transpose(image, perm=[2, 1, 0, 3])  # (z, y, x, c)
                image = tf.image.rot90(image, k=num_rotations)
                image = tf.transpose(image, perm=[2, 1, 0, 3])  # Back to (x, y, z, c)

        return image

    # def process_imaging_domain(self, image):
    #     """ Standardizes image data and creates subvolumes """
    #     arr = tf.image.random_crop(image, size=self.imaging_patch_shape)
    #     # return self.random_rotate_3d(arr)
    #     return self.random_rotate(arr, preserve_z_axis=self.surface_illumination)

    @tf.function
    def process_imaging_domain(self, image):

        # Set the cropping size dynamically based on whether the image is 2D or 3D
        if self.DIMENSIONS == 2:  # 2D image (batch, X, Y, C)
            crop_shape = self.imaging_patch_shape[:3]  # Ignore Z dimension for 2D
        else:  # 3D image (batch, X, Y, Z, C)
            crop_shape = self.imaging_patch_shape

        # Initialize a loop counter
        i = tf.constant(0)

        # Define the maximum number of iterations
        max_iterations = tf.constant(20)

        # Initialize arr with a random crop
        arr = tf.image.random_crop(image, size=crop_shape)

        # Define the condition for the while loop
        def condition(i, arr):
            return tf.math.logical_and(i < max_iterations, tf.math.reduce_mean(arr) < -0.8)

        # Define the body of the while loop
        def body(i, _):
            # Generate a new random crop from the original image
            new_arr = tf.image.random_crop(image, size=crop_shape)
            return i + 1, new_arr

        # Run the loop
        _, arr = tf.while_loop(condition, body, [i, arr])

        # If it's a 3D image, apply 3D-specific operations
        if self.DIMENSIONS == 3:  # 3D image (batch, X, Y, Z, C)
            arr = self.random_rotate(arr, preserve_z_axis=self.surface_illumination)

        # If it's a 2D image, skip the 3D-specific operation
        return self.random_rotate(arr, preserve_z_axis=self.surface_illumination)

    @tf.function
    def process_seg_domain(self, image):
        """
        Process segmentation domain for both 2D and 3D images.

        Args:
        - image: Input image tensor.
        - dim: Dimensionality of the input image (2 for 2D, 3 for 3D).

        Returns:
        - Cropped image tensor.
        """

        # Find the coordinates of non-zero elements in the binary mask
        positive_coords = tf.where(image > 0.)

        if tf.size(positive_coords) == 0:
            # If no non-zero elements, handle accordingly (e.g., return original image)
            return image

        # Randomly shuffle the non-zero coordinates and select one
        random_index = tf.random.shuffle(positive_coords)
        random_coord = tf.cast(random_index[0], tf.int32)

        # Calculate the cropping size based on self.segmentation_patch_shape
        crop_size = tf.cast(self.segmentation_patch_shape, tf.int32)

        if self.DIMENSIONS == 3:
            # For 3D images

            # Calculate valid crop boundaries to prevent overlap with outer dimensions
            width_max = tf.cast(tf.shape(image)[0] - crop_size[0], tf.int32)
            length_max = tf.cast(tf.shape(image)[1] - crop_size[1], tf.int32)
            depth_max = tf.cast(tf.shape(image)[2] - crop_size[2], tf.int32)

            # Ensure random_coord is within valid bounds
            width_start = tf.minimum(tf.maximum(0, random_coord[0] - crop_size[0] // 2), width_max)
            length_start = tf.minimum(tf.maximum(0, random_coord[1] - crop_size[1] // 2), length_max)
            depth_start = tf.minimum(tf.maximum(0, random_coord[2] - crop_size[2] // 2), depth_max)

            # Calculate crop boundaries based on start positions
            width_end = width_start + crop_size[0]
            length_end = length_start + crop_size[1]
            depth_end = depth_start + crop_size[2]

            # Perform cropping
            arr = image[width_start:width_end, length_start:length_end, depth_start:depth_end, :]
        else:
            # For 2D images

            # Calculate valid crop boundaries to prevent overlap with outer dimensions
            width_max = tf.cast(tf.shape(image)[0] - crop_size[0], tf.int32)
            length_max = tf.cast(tf.shape(image)[1] - crop_size[1], tf.int32)

            # Ensure random_coord is within valid bounds
            width_start = tf.minimum(tf.maximum(0, random_coord[0] - crop_size[0] // 2), width_max)
            length_start = tf.minimum(tf.maximum(0, random_coord[1] - crop_size[1] // 2), length_max)

            # Calculate crop boundaries based on start positions
            width_end = width_start + crop_size[0]
            length_end = length_start + crop_size[1]

            # Perform cropping
            arr = image[width_start:width_end, length_start:length_end, :]

        # Apply spatial augmentation if needed (e.g., self.random_rotate_3d for 3D or 2D equivalent)
        return self.random_rotate(arr)

    @tf.function
    def process_seg_domain_method2(self, image):

        if self.semi_supervised:
            shape = self.ss_patch_shape
        else:
            shape = self.segmentation_patch_shape

        def body(arr, image):
            return [tf.image.random_crop(image, size=shape), image]

        def segmentationCondition(arr, image):
            if self.semi_supervised:
                return tf.math.less(tf.math.reduce_max(tf.split(arr, num_or_size_splits=2, axis=2)), 0.8)
            else:
                return tf.math.less(tf.math.reduce_max(arr), 0.8)

        arr = tf.image.random_crop(value=image, size=shape)
        arr, _ = tf.while_loop(segmentationCondition, body, [arr, image], maximum_iterations=10)

        return self.random_rotate(arr)

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
                dIS = dS[:, :, self.segmentation_patch_shape[2]:, ]
                dS = dS[:, :, :self.segmentation_patch_shape[2], ]
            if self.args.DIMENSIONS == 3:
                ''' Save 3D images '''
                io.imsave("./GANMonitor/Imaging_Test_Input.tiff",
                          np.transpose(dI, (2, 0, 1, 3)),
                          bigtiff=False, check_contrast=False)

                io.imsave("./GANMonitor/Segmentation_Test_Input.tiff",
                          np.transpose(dS, (2, 0, 1, 3)),
                          bigtiff=False, check_contrast=False)

            if self.args.DIMENSIONS == 2:
                showI = dI
                showS = dS
                axs[0, 0].imshow(showI, cmap='gray', vmin=-1., vmax=1.)
                axs[0, 1].imshow(showS, cmap='gray', vmin=-1., vmax=1.)
            else:
                for j in range(0, nfig):
                    showI = (dI[:, :, j * int(self.segmentation_patch_shape[2] / nfig), ])
                    showS = (dS[:, :, j * int(self.segmentation_patch_shape[2] / nfig), ])
                    axs[j, 0].imshow(showI, cmap='gray', vmin=-1., vmax=1.)
                    axs[j, 1].imshow(showS, cmap='gray', vmin=-1., vmax=1.)
                    if self.semi_supervised:
                        showIS = (dIS[:, :, j * int(self.segmentation_patch_shape[2] / nfig), ])
                        axs[j, 2].imshow(showIS, cmap='gray', vmin=-1., vmax=1.)

            ''' Include histograms '''
            axs[nfig, 0].hist(dI.ravel(), bins=256, range=(-1., 1.), fc='k', ec='k', density=True)
            axs[nfig, 1].hist(dS.ravel(), bins=256, range=(-1., 1.), fc='k', ec='k', density=True)
            if self.semi_supervised:
                axs[nfig, 2].hist(dIS.ravel(), bins=256, range=(-1., 1.), fc='k', ec='k',
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
                    axs[j, 0].imshow(showI, cmap='gray', vmin=-1., vmax=1.)
                    axs[j, 1].imshow(showS, cmap='gray', vmin=-1., vmax=1.)
                    if self.semi_supervised:
                        showIS = dIS[:, j * int(self.segmentation_patch_shape[1] / nfig),
                                 :self.args.SUBVOL_PATCH_SIZE[2] - 1, ]
                        axs[j, 2].imshow(showIS, cmap='gray', vmin=-1., vmax=1.)

            # Set axis labels
            axs[0, 0].set_title('Imaging Dataset (YZ)')
            axs[0, 1].set_title('Segmentation Dataset (YZ)')
            if self.semi_supervised:
                axs[0, 2].set_title('Paired Dataset (YZ)')
            plt.show(block=False)
            plt.close()
