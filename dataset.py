import random
import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

class dataset_gen:
    def __init__(self, args, imaging_domain_data, seg_domain_data, strategy: tf.distribute.Strategy, otf_imaging=None):
        ''' Setting shard policy for distributed dataset '''
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
            self.imaging_patch_shape = (args.SUBVOL_PATCH_SIZE[0], args.SUBVOL_PATCH_SIZE[1], args.SUBVOL_PATCH_SIZE[2], args.CHANNELS)
            self.segmentation_patch_shape = (args.SUBVOL_PATCH_SIZE[0], args.SUBVOL_PATCH_SIZE[1], args.SUBVOL_PATCH_SIZE[2], 1)
            
        self.strategy = strategy
        self.pathA = imaging_domain_data
        self.pathB = seg_domain_data
        self.args = args
        self.otf_imaging = otf_imaging
        self.IMAGE_THRESH = 0.5
        self.SEG_THRESH = 0.8
        
        ''' Create datasets '''
        with self.strategy.scope():
            self.trainDatasetA = tf.data.Dataset.from_generator(lambda: self.datagenA('training'), 
                                                                output_types=tf.float32,
                                                                output_shapes=self.imaging_output_shapes)
            self.trainDatasetA = self.trainDatasetA.map(self.processImagingDomain, num_parallel_calls=tf.data.AUTOTUNE)
            self.trainDatasetA = self.trainDatasetA.repeat()
            self.trainDatasetA = self.trainDatasetA.with_options(options)
            
            self.trainDatasetB = tf.data.Dataset.from_generator(lambda: self.datagenB('training'), 
                                                                output_types=tf.float32,
                                                                output_shapes=self.segmentation_output_shapes)
            self.trainDatasetB = self.trainDatasetB.map(self.processSegDomain, num_parallel_calls=tf.data.AUTOTUNE)
            self.trainDatasetB = self.trainDatasetB.repeat()
            self.trainDatasetB = self.trainDatasetB.with_options(options)
            
            self.valDatasetA = tf.data.Dataset.from_generator(lambda: self.datagenA('validation'), 
                                                              output_types=tf.float32,
                                                              output_shapes=self.imaging_output_shapes)
            self.valDatasetA  = self.valDatasetA.map(map_func = self.processImagingDomain, num_parallel_calls=tf.data.AUTOTUNE)
            self.valDatasetA = self.valDatasetA.repeat()
            self.valDatasetA = self.valDatasetA.with_options(options)
            
            self.valDatasetB = tf.data.Dataset.from_generator(lambda: self.datagenB('validation'), 
                                                              output_types=tf.float32,
                                                              output_shapes=self.segmentation_output_shapes)
            self.valDatasetB = self.valDatasetB.map(map_func = self.processSegDomain, num_parallel_calls=tf.data.AUTOTUNE)
            self.valDatasetB = self.valDatasetB.repeat()
            self.valDatasetB = self.valDatasetB.with_options(options)
            
            self.plotSampleDataset()
            
            self.valFullDatasetA = tf.data.Dataset.from_generator(self.valDatagenA, output_types=(tf.float32, tf.int8))
            self.valFullDatasetB = tf.data.Dataset.from_generator(self.valDatagenB, output_types=(tf.float32, tf.int8))
            
            self.train_ds = tf.data.Dataset.zip((self.trainDatasetA.batch(args.GLOBAL_BATCH_SIZE, drop_remainder=True),
                                            self.trainDatasetB.batch(args.GLOBAL_BATCH_SIZE, drop_remainder=True))).prefetch(tf.data.AUTOTUNE)
            self.val_ds = tf.data.Dataset.zip((self.valDatasetA.batch(args.GLOBAL_BATCH_SIZE, drop_remainder=True),
                                          self.valDatasetB.batch(args.GLOBAL_BATCH_SIZE, drop_remainder=True))).prefetch(tf.data.AUTOTUNE)

            self.train_ds = self.strategy.experimental_distribute_dataset(self.train_ds)
            self.val_ds = self.strategy.experimental_distribute_dataset(self.val_ds)
            

    
    ''' Functions to gather imaging subvolumes '''
    def datagenA(self, typ='training'):
        """
        Generates a batch of data from the pathA directory.

        Args:
        - typ (str): The type of data to generate, either 'training' or 'validation'. Default is 'training'.

        Returns:
        - tensor: A tensor of shape [batch_size, height, width, channels] containing the batch of images.
        """
        iterA = 0
        datasetA = self.pathA[typ]
        np.random.shuffle(datasetA)
        while True:
            if iterA >= math.floor(len(datasetA) // self.args.GLOBAL_BATCH_SIZE):
                iterA = 0
                np.random.shuffle(datasetA)
            
            file = datasetA[iterA*self.args.GLOBAL_BATCH_SIZE:(iterA+1)*self.args.GLOBAL_BATCH_SIZE]
            
            # Load batch of full size images
            for idx, filename in enumerate(file):
                yield tf.convert_to_tensor(np.rot90(np.load(filename), 
                                                    np.random.choice([-1, 0, 1])), dtype=tf.float32)
            
            iterA += 1
    
    def datagenB(self, typ='training'):
        """
        Generates a batch of data from the pathB directory.

        Args:
        - typ (str): The type of data to generate, either 'training' or 'validation'. Default is 'training'.

        Returns:
        - tensor: A tensor of shape [batch_size, height, width, channels] containing the batch of images.
        """
        iterB = 0
        datasetB = self.pathB[typ]
        np.random.shuffle(datasetB)
        while True:
            if iterB >= math.floor(len(datasetB) // self.args.GLOBAL_BATCH_SIZE):
                iterB = 0
                np.random.shuffle(datasetB)
            
            file = datasetB[iterB*self.args.GLOBAL_BATCH_SIZE:(iterB+1)*self.args.GLOBAL_BATCH_SIZE]
            
            # Load batch of full size images
            for idx, filename in enumerate(file):
                yield tf.convert_to_tensor(np.rot90(np.load(filename), 
                                                    np.random.choice([-1, 0, 1])), dtype=tf.float32)
            
            iterB += 1
    
    def valDatagenA(self):
        while True:
            i = random.randint(0,self.pathA['validation'].shape[0]-1)
            yield (tf.convert_to_tensor(np.load(self.pathA['validation'][i]) , dtype=tf.float32), i)
    
    def valDatagenB(self):
        while True:
            i = random.randint(0,self.pathB['validation'].shape[0]-1) 
            yield (tf.convert_to_tensor(np.load(self.pathB['validation'][i]), dtype=tf.float32), i)
            
    
    ''' Functions for data preprocessing '''
    def body(self, arr, image):
        return [tf.image.random_crop(image, size=self.segmentation_patch_shape), image]
    
    def imagingCondition(self, arr, image):
        return tf.math.less(tf.math.reduce_max(arr), self.IMAGE_THRESH)
    
    def segmentationCondition(self, arr, image):
        return tf.math.less(tf.math.reduce_max(arr), self.SEG_THRESH)
    
    def processImagingDomain(self, image):
        ''' Standardizes image data and creates subvolumes '''
        subvol = tf.image.random_crop(image, size=self.imaging_patch_shape)
        if self.otf_imaging is not None:
            subvol = self.otf_imaging(subvol)
        return subvol
    
    def processSegDomain(self, image):
        arr = tf.image.random_crop(value=image, size=self.segmentation_patch_shape)
        arr, _ = tf.while_loop(self.segmentationCondition, self.body, [arr, image], maximum_iterations=10)
        return arr
    
    def plotSampleDataset(self):
        """
        Plots a sample of the input datasets A and B along with their histograms. 
        The function saves a 3D TIFF file of the input data.

        Args:
        - self.trainDatasetA: Dataset A.
        - self.trainDatasetB: Dataset B.
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
            
        fig, axs = plt.subplots(nfig+1, 2, figsize=(10, 15))
        fig.subplots_adjust(hspace=0.5)
        for i, samples in enumerate(zip(self.trainDatasetA.take(1), self.trainDatasetB.take(1))):   
    
            dA = samples[0].numpy()
            dB = samples[1].numpy()
    
            if self.args.DIMENSIONS == 3:
                ''' Save 3D images '''
                io.imsave("./GANMonitor/Test_Input_A.tiff", 
                                      np.transpose(dA,(2,0,1,3)), 
                                      bigtiff=False, check_contrast=False)
    
                io.imsave("./GANMonitor/Test_Input_B.tiff", 
                                      np.transpose(dB,(2,0,1,3)), 
                                      bigtiff=False, check_contrast=False)
    
            if self.args.DIMENSIONS == 2:
                showA = (dA * 127.5 + 127.5).astype('uint8')
                showB = (dB * 127.5 + 127.5).astype('uint8')
                axs[0, 0].imshow(showA, cmap='gray')
                axs[0, 1].imshow(showB, cmap='gray')
            else:
                for j in range(0,nfig):
                    showA = (dA[:,:,j*int(self.args.SUBVOL_PATCH_SIZE[2]/nfig),])
                    showB = (dB[:,:,j*int(self.args.SUBVOL_PATCH_SIZE[2]/nfig),])
                    axs[j, 0].imshow(showA, cmap='gray')
                    axs[j, 1].imshow(showB, cmap='gray')
    
            ''' Include histograms '''
            axs[nfig,0].hist(dA.ravel(), bins=256, range=(np.amin(dA),np.amax(dA)), fc='k', ec='k', density=True)
            axs[nfig,1].hist(dB.ravel(), bins=256, range=(np.amin(dB),np.amax(dB)), fc='k', ec='k', density=True)
            
            # Set axis labels
            axs[0, 0].set_title('Dataset A Example (XY Slices)')
            axs[0, 1].set_title('Dataset B Example (XY Slices)')
            axs[nfig, 0].set_ylabel('Voxel Frequency')
            plt.show(block=False)
            plt.close()
        
            
            if self.args.DIMENSIONS == 3:
                _, axs = plt.subplots(nfig, 2, figsize=(10, 15))
                for j in range(0,nfig):
                    showA = dA[:,j*int(self.args.SUBVOL_PATCH_SIZE[1]/nfig),:,0]
                    showB = dB[:,j*int(self.args.SUBVOL_PATCH_SIZE[1]/nfig),:self.args.SUBVOL_PATCH_SIZE[2]-1,0] 
                    axs[j, 0].imshow(showA, cmap='gray')
                    axs[j, 1].imshow(showB, cmap='gray')
                 
            # Set axis labels
            axs[0, 0].set_title('Dataset A Example (YZ Slices)')
            axs[0, 1].set_title('Dataset B Example (YZ Slices)')
            plt.show(block=False)
            plt.close()
        
