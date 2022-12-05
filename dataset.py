import os
import random
import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

from loss_functions import rescaleTensor

def normalise(img):
    img = (img - tf.math.reduce_min(img)) / (tf.math.reduce_max(img) - tf.math.reduce_min(img))
    return rescaleTensor(img, alpha=-0.5, beta=0.5)

def getDataset(args, pathA, pathB, strategy: tf.distribute.Strategy):
    
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    
    
    ''' Functions to gather imaging subvolumes '''
    def datagenA(typ='training'):
        iterA = 0
        dataset = pathA[typ]
        np.random.shuffle(dataset)
        while True:
            if iterA >= math.floor(len(dataset) // args.GLOBAL_BATCH_SIZE):
                iterA = 0
                np.random.shuffle(dataset)
            
            file = dataset[iterA*args.GLOBAL_BATCH_SIZE:(iterA+1)*args.GLOBAL_BATCH_SIZE]
            
            # Load batch of full size images
            for idx, filename in enumerate(file):
                yield tf.convert_to_tensor(np.load(filename), dtype=tf.float32)
            iterA += 1
            
    def body(arr, image):
        return [tf.image.random_crop(image, size=args.SUBVOL_PATCH_SIZE), image]

    def imagingCondition(arr, image):
        return tf.logical_or(tf.math.less(tf.math.reduce_max(arr), 0.), 
                             tf.math.greater_equal(tf.random.uniform(shape=[], minval=0.,maxval=1.), 0.05))
    
    def segmentationCondition(arr, image):
        return tf.math.less(tf.math.reduce_max(arr), 0.8)
            
    def processImagingDomain(image):
        return tf.image.random_crop(value=image, size=args.SUBVOL_PATCH_SIZE)
    
    def data_gen_testA():
        i = random.randint(0,pathA['validation'].shape[0]-1)
        yield (tf.convert_to_tensor(np.load(pathA['validation'][i]) , dtype=tf.float32), i)

    
    ''' Functions to gather synthetic datasets '''
    def datagenB(typ='training'):
        iterB = 0
        dataset = pathB[typ]
        np.random.shuffle(dataset)
        while True:
            if iterB >= math.floor(len(dataset) // args.GLOBAL_BATCH_SIZE):
                iterB = 0
                np.random.shuffle(dataset)
            
            file = dataset[iterB*args.GLOBAL_BATCH_SIZE:(iterB+1)*args.GLOBAL_BATCH_SIZE]
            
            for idx, filename in enumerate(file):
                yield tf.convert_to_tensor(np.load(filename), 
                                                     dtype=tf.float32)
            iterB += 1

    def processSegDomain(image):
        arr = tf.image.random_crop(value=image, size=(args.SUBVOL_PATCH_SIZE))
        arr, _ = tf.while_loop(segmentationCondition, body, [arr, image])
        return arr
    
    def splitProcess(image):
        image, _ = tf.split(image, num_or_size_splits=2, axis=2)
        return image
    
    def data_gen_testB():
        i = random.randint(0,pathB['validation'].shape[0]-1) 
        yield (tf.convert_to_tensor(np.load(pathB['validation'][i]), dtype=tf.float32), i)
    

    ''' Create tensorflow training datasets '''
    trainDatasetA = tf.data.Dataset.from_generator(lambda: datagenA(typ='training'), 
                                                   output_types=tf.float32,
                                                   output_shapes=(args.TARG_IMG_SIZE[0], args.TARG_IMG_SIZE[1], args.TARG_IMG_SIZE[2], 1))
    trainDatasetA = trainDatasetA.map(map_func=processImagingDomain, num_parallel_calls=tf.data.AUTOTUNE)
    trainDatasetA = trainDatasetA.repeat()
    trainDatasetA = trainDatasetA.with_options(options)
    
    
    trainDatasetB = tf.data.Dataset.from_generator(lambda: datagenB(typ='training'), 
                                                   output_types=tf.float32,
                                                   output_shapes=(args.SYNTH_IMG_SIZE[0], args.SYNTH_IMG_SIZE[1], args.SYNTH_IMG_SIZE[2], 1))
    trainDatasetB = trainDatasetB.map(map_func=processSegDomain, num_parallel_calls=tf.data.AUTOTUNE)
    trainDatasetB = trainDatasetB.repeat()
    trainDatasetB = trainDatasetB.with_options(options)
    
    
    ''' Create validation datasets ''' 
    valDatasetA = tf.data.Dataset.from_generator(lambda: datagenA(typ='validation'), 
                                                    output_types=tf.float32,
                                                    output_shapes=(args.TARG_IMG_SIZE[0], args.TARG_IMG_SIZE[1], args.TARG_IMG_SIZE[2], 1))
    valDatasetA = trainDatasetA.map(map_func=processImagingDomain, num_parallel_calls=tf.data.AUTOTUNE)
    valDatasetA = trainDatasetA.repeat()
    valDatasetA = trainDatasetA.with_options(options)
    
    valDatasetB = tf.data.Dataset.from_generator(lambda: datagenB(typ='validation'), 
                                                   output_types=tf.float32,
                                                   output_shapes=(args.SYNTH_IMG_SIZE[0], args.SYNTH_IMG_SIZE[1], args.SYNTH_IMG_SIZE[2], 1))
    valDatasetB = valDatasetB.map(map_func=processSegDomain, num_parallel_calls=tf.data.AUTOTUNE)
    valDatasetB = valDatasetB.repeat()
    valDatasetB = valDatasetB.with_options(options)
    
    valFullDatasetA = tf.data.Dataset.from_generator(data_gen_testA, output_types=(tf.float32, tf.int8))
    valFullDatasetB = tf.data.Dataset.from_generator(data_gen_testB, output_types=(tf.float32, tf.int8))
    
    
    ''' Plot dataset output '''
    plotSampleDataset(args,
                      trainDatasetA=trainDatasetA, 
                      trainDatasetB=trainDatasetB)
    
    
    train_ds = tf.data.Dataset.zip((trainDatasetA.batch(args.GLOBAL_BATCH_SIZE, drop_remainder=True),
                                    trainDatasetB.batch(args.GLOBAL_BATCH_SIZE, drop_remainder=True))).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.zip((valDatasetA.batch(args.GLOBAL_BATCH_SIZE, drop_remainder=True),
                                  valDatasetB.batch(args.GLOBAL_BATCH_SIZE, drop_remainder=True))).prefetch(tf.data.AUTOTUNE)

    
    ''' Create distributed dataset '''
    train_ds = strategy.experimental_distribute_dataset(train_ds)
    val_ds = strategy.experimental_distribute_dataset(val_ds)
    
    return train_ds, val_ds, valFullDatasetA, valFullDatasetB


def plotSampleDataset(args, trainDatasetA, trainDatasetB):

    '''
    TEST LOADED DATASETS
    '''
    
    #  Visualise some examples
    nfig = 6
    _, ax = plt.subplots(nfig+1, 2, figsize=(10, 15))
    for i, samples in enumerate(zip(trainDatasetA.take(1), trainDatasetB.take(1))):   
        
        dA = samples[0].numpy()
        dB = samples[1].numpy()
        
        io.imsave("./GANMonitor/Test_Input_A.tiff", 
                              np.transpose(dA,(2,0,1,3)), 
                              bigtiff=False, check_contrast=False)
        
        io.imsave("./GANMonitor/Test_Input_B.tiff", 
                              np.transpose(dB,(2,0,1,3)), 
                              bigtiff=False, check_contrast=False)
        
        for j in range(0,nfig):
            showA = (dA[:,:,j*int(args.SUBVOL_PATCH_SIZE[2]/nfig),0] * 127.5 + 127.5).astype('uint8')
            showB = (dB[:,:,j*int(args.SUBVOL_PATCH_SIZE[2]/nfig),0] * 127.5 + 127.5).astype('uint8')
            # showC = (dB[:,:,args.SUBVOL_PATCH_SIZE[2] + j*int(args.SUBVOL_PATCH_SIZE[2]/nfig),0] * 127.5 + 127.5).astype('uint8')
            ax[j, 0].imshow(showA, cmap='gray')
            ax[j, 1].imshow(showB, cmap='gray')
            # ax[j, 2].imshow(showC, cmap='gray')
        ax[nfig,0].hist(dA.ravel(), bins=256, range=(-1.0,1.0), fc='k', ec='k', density=True)
        ax[nfig,1].hist(dB.ravel(), bins=256, range=(-1.0,1.0), fc='k', ec='k', density=True)
    plt.show(block=False)
    plt.close()
    
    nfig = 6
    _, ax = plt.subplots(nfig, 2, figsize=(10, 15))
    for j in range(0,nfig):
        showA = (dA[:,j*int(args.SUBVOL_PATCH_SIZE[1]/nfig),:,0] * 127.5 + 127.5).astype('uint8')
        showB = (dB[:,j*int(args.SUBVOL_PATCH_SIZE[1]/nfig),:args.SUBVOL_PATCH_SIZE[2]-1,0] * 127.5 + 127.5).astype('uint8')
        # showC = (dB[:,j*int(args.SUBVOL_PATCH_SIZE[1]/nfig),args.SUBVOL_PATCH_SIZE[2]:,0] * 127.5 + 127.5).astype('uint8')
        ax[j, 0].imshow(showA, cmap='gray')
        ax[j, 1].imshow(showB, cmap='gray')
        # ax[j, 2].imshow(showC, cmap='gray')
    plt.show(block=False)
    plt.close()