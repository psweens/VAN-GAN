import os
import shutil
import random
import numpy as np
import skimage.io as sk
from scipy import stats

from joblib import Parallel, delayed
import multiprocessing

from utils import min_max_norm, check_nan, save_dict, load_dict, resize_volume, load_volume


class DataPreprocessor:
    def __init__(self, args=None, raw_path=None, main_dir=None, partition_id='', partition_filename=None,
                 tiff_size=(600, 600, 700),
                 target_size=(600, 600, 700),
                 num_cores=multiprocessing.cpu_count() - 1):
        self.save_filtered = None
        self.resize = None
        self.preprocess_fn = None
        self.raw_path = raw_path
        self.main_dir = main_dir
        self.partition_id = partition_id
        self.partition_filename = partition_filename
        self.tiff_size = tiff_size
        self.target_size = target_size
        self.train_files = None
        self.validate_files = None
        self.test_files = None
        self.partition = {}
        self.data_type = 'float32'

        self.NUM_CORES = int(0.8 * num_cores)
        if args is not None:
            self.DIMENSIONS = args.DIMENSIONS
            self.CHANNELS = args.CHANNELS

    def save_partition(self, save_path=None):
        """
        Save the partition data into files in the specified directory.
        
        Args:
            save_path (str): The directory where the partition files will be saved.
        
        Returns:
            None
        """

        if save_path is None:
            raise ValueError("Partition save_path is not provided.")

        # Update partition directories
        new_partition = {}
        train_arr = np.empty(len(self.partition['training']), dtype=object)
        val_arr = np.empty(len(self.partition['validation']), dtype=object)
        test_arr = np.empty(len(self.partition['testing']), dtype=object)

        # Update the training partition directory
        for i in range(len(self.partition['training'])):
            file = self.partition['training'][i]
            file, _ = os.path.splitext(file)
            file = file + '.npy'
            file = os.path.join(save_path, 'train' + self.partition_id, file)
            train_arr[i] = file

        #  Update the validation partition directory
        for i in range(len(self.partition['validation'])):
            file = self.partition['validation'][i]
            file, _ = os.path.splitext(file)
            file = file + '.npy'
            file = os.path.join(save_path, 'val' + self.partition_id, file)
            val_arr[i] = file

        #  Update the testing partition directory
        for i in range(len(self.partition['testing'])):
            file = self.partition['testing'][i]
            file, _ = os.path.splitext(file)
            file = file + '.npy'
            file = os.path.join(save_path, 'test' + self.partition_id, file)
            test_arr[i] = file

        new_partition['training'] = train_arr
        new_partition['validation'] = val_arr
        new_partition['testing'] = test_arr

        save_dict(new_partition, os.path.join(save_path, self.partition_filename))

        self.partition = new_partition

    def load_partition(self, file_path):
        print('*** Loading Dataset %s Partition ***' % self.partition_id)
        self.partition = load_dict(file_path)

    def split_dataset(self):

        # Shuffle raw data list
        files = os.listdir(self.raw_path)
        random.shuffle(files)

        # Split data into train/validate/test
        print('Splitting dataset ...')
        self.train_files, self.test_files = np.split(files, [int(len(files) * 0.9)])
        self.train_files, self.validate_files = np.split(self.train_files,
                                                         [int(len(self.train_files,) * 0.8)])

        # Save partitioned dataset
        self.partition['training'] = self.train_files
        self.partition['validation'] = self.validate_files
        self.partition['testing'] = self.test_files

    def move_dataset(self):
        for file in range(len(self.partition['training'])):
            shutil.move(os.path.join(self.raw_path, self.partition['training'][file]),
                        os.path.join(self.main_dir, 'train' + self.partition_id))
        for file in range(len(self.partition['validation'])):
            shutil.move(os.path.join(self.raw_path, self.partition['validation'][file]),
                        os.path.join(self.main_dir, 'val' + self.partition_id))
        for file in range(len(self.partition['testing'])):
            shutil.move(os.path.join(self.raw_path, self.partition['testing'][file]),
                        os.path.join(self.main_dir, 'test' + self.partition_id))

    def preprocess(self, preprocess_fn=None, resize=False, save_filtered=False):

        print('*** Preprocessing partition %s images ***' % self.partition_id)
        self.split_dataset()

        self.preprocess_fn = preprocess_fn
        self.resize = resize
        self.save_filtered = save_filtered

        print('Processing training data ...')
        Parallel(n_jobs=self.NUM_CORES, verbose=50)(delayed(
            self.process_tiff)(file=self.partition['training'][file],
                               label='train') for file in range(len(self.partition['training'])))

        print('Processing validation data ...')
        Parallel(n_jobs=self.NUM_CORES, verbose=50)(delayed(
            self.process_tiff)(file=self.partition['validation'][file],
                               label='val') for file in range(len(self.partition['validation'])))

        print('Processing testing data ...')
        Parallel(n_jobs=self.NUM_CORES, verbose=50)(delayed(
            self.process_tiff)(file=self.partition['testing'][file],
                               label='test') for file in range(len(self.partition['testing'])))

        self.save_partition(self.main_dir)

    def process_tiff(self, file, label=''):

        """
        Process a TIFF image file.
        
        Args:
            file (str): The name of the file to be processed
            label (str): The label to be appended to the processed image
        
        Returns:
            None
        """

        stack = load_volume(os.path.join(self.raw_path, file), datatype=self.data_type, normalise=True)

        file, ext = os.path.splitext(file)
        # if partition_id == 'A':
        if self.DIMENSIONS == 3:
            stack = np.transpose(stack, (1, 2, 0))


        # if self.partition_id == 'B':
        #     stack = get_vacuum(stack, self.DIMENSIONS) # Reduce bounding box to tree size

        if self.preprocess_fn is not None:
            stack = self.preprocess_fn(stack)

        if not self.tiff_size == self.target_size and self.resize:
            stack = (resize_volume(stack, self.target_size)).astype(self.data_type)
            if self.partition_id == 'B':
                stack[stack < 0.] = 0.0
                stack[stack > 255.] = 255

        if self.partition_id == 'B':
            stack = min_max_norm(stack)
            mode, _ = stats.mode(stack, axis=None)
            if mode == 1:
                stack -= 1.
                stack = abs(stack)
            stack = (stack - 0.5) / 0.5

        if self.partition_id == 'B':
            stack[stack < 0.] = -1.0
            stack[stack >= 0.] = 1.0

        if not check_nan(stack):

            if self.save_filtered:
                arr_out = os.path.join(os.path.join(self.main_dir, 'filtered'),
                                       label + self.partition_id, file + '.tiff')
                if ext == '.npy':
                    sk.imsave(arr_out, (stack * 127.5 + 127.5).astype('uint8'), bigtiff=False, check_contrast=False)
                else:
                    if self.DIMENSIONS == 2:
                        sk.imsave(arr_out, (stack * 127.5 + 127.5).astype('uint8'), bigtiff=False, check_contrast=False)
                    else:
                        sk.imsave(arr_out, (np.transpose(stack, (2, 0, 1)) * 127.5 + 127.5).astype('float32'),
                                  bigtiff=False, check_contrast=False)

            if self.partition_id == 'B':
                np.save(os.path.join(self.main_dir, label + self.partition_id, file),
                        np.expand_dims(stack, axis=self.DIMENSIONS))
            else:
                if self.DIMENSIONS == 2 and self.CHANNELS == 3:
                    np.save(os.path.join(self.main_dir, label + self.partition_id, file), stack)
                else:
                    np.save(os.path.join(self.main_dir, label + self.partition_id, file),
                            np.expand_dims(stack, axis=self.DIMENSIONS))
        else:
            print('NaN detected ...')

    def process_new_data(self, current_path, new_path, tiff_size=None, target_size=None, preprocess_fn=None,
                         resize=None):

        self.raw_path = current_path
        self.main_dir = new_path
        self.tiff_size = tiff_size
        self.target_size = target_size
        self.preprocess_fn = preprocess_fn
        self.resize = resize
        self.save_filtered = False

        files = os.listdir(current_path)
        for file in files:
            self.process_tiff(file=file)
