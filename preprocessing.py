import os
import shutil
import random
import pickle
import itk
import cv2
import numpy as np
import scipy.stats as sp
import skimage.io as sk
from skimage import exposure

from joblib import Parallel, delayed
import multiprocessing


def get_sub_volume(image,subvol=(64,64,512),n_samples=1):

    # Initialize features and labels with `None`
    X = np.empty([subvol[0], subvol[1], subvol[2], subvol[3]],dtype='float32')

    # randomly sample sub-volume by sampling the corner voxel
    start_x = np.random.randint(image.shape[0] - subvol[0] + 1 )
    start_y = np.random.randint(image.shape[1] - subvol[1] + 1 )
    start_z = np.random.randint(image.shape[2] - subvol[2] + 1 )

    # make copy of the sub-volume
    X = np.copy(image[start_x: start_x + subvol[0],
                      start_y: start_y + subvol[1],
                      start_z: start_z + subvol[2], :])

    return X

def load_volume(file, size=(600,600,700), ext='*.tif', datatype='uint8', normalise=True):
    vol = (sk.imread(file)).astype(datatype)
    if normalise:
        vol = norm_data(vol)
    return vol

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

def norm_data(data):
    dmin = np.min(data)
    dmax = np.max(data)
    return (data - dmin) / (dmax - dmin)

def stdnorm(data):
    dstd = np.std(data)
    if dstd > 0.:
        return (data - np.mean(data)) / dstd
    else:
        return (data - np.mean(data))
    
def histogram_equalize(img):
    img_cdf, bin_centers = exposure.cumulative_distribution(img)
    return np.interp(img, bin_centers, img_cdf)

def checkNAN(arr):
    arr_sum = np.sum(arr)
    return np.isnan(arr_sum)

def subarray(arr):
    x, y, z = np.nonzero(arr)
    # Using the smallest and largest x and y indices of nonzero elements, 
    # we can find the desired rectangular bounds.  
    # And don't forget to add 1 to the top bound to avoid the fencepost problem.
    return arr[x.min():x.max()+1, y.min():y.max()+1,]

def split_dataset(raw_path, new_dir, partition_id, partition_filename, 
                  tiff_size=(600,600,700), resize=True, target_size=(600,600,700), 
                  move_only=False, local_filter=False, global_filter=True, save_filtered=False,
                  num_cores = multiprocessing.cpu_count() - 1):
    
    num_cores = int(0.8*num_cores)
    # Load and shuffle raw data
    files = os.listdir(raw_path)
    random.shuffle(files)
    
    # train_files = os.listdir('/media/sweene01/5e3122f3-5d00-4bc0-b099-982762bf3999/VS-GAN_All_Data_plusCH/trainA')
    # validate_files = os.listdir('/media/sweene01/5e3122f3-5d00-4bc0-b099-982762bf3999/VS-GAN_All_Data_plusCH/valA')
    # test_files = os.listdir('/media/sweene01/5e3122f3-5d00-4bc0-b099-982762bf3999/VS-GAN_All_Data_plusCH/testA')
    
    # Split data into train/validate/test
    print('Splitting dataset ...')
    train_files, test_files = np.split(files, [int(len(files)*0.9)])
    train_files, validate_files = np.split(files, [int(len(files)*0.8)])
    
    # Save partitioned dataset
    # partition = load_dict('/media/sweene01/SSD/3DcycleGAN_data/dataA_partition.pkl')
    partition = {}
    partition['training'] = train_files
    partition['validation'] = validate_files
    partition['testing'] = test_files
    save_dict(partition, os.path.join(new_dir, 'paired_data.pkl'))
    
    # Move data into appropriate folders
    if move_only:
        for file in range(len(partition['training'])):
            shutil.move(os.path.join(raw_path,partition['training'][file]),
                        os.path.join(new_dir,'train'+partition_id))
        for file in range(len(partition['validation'])):
            shutil.move(os.path.join(raw_path,partition['validation'][file]),
                        os.path.join(new_dir,'val'+partition_id))
        for file in range(len(partition['testing'])):
            shutil.move(os.path.join(raw_path,partition['testing'][file]),
                        os.path.join(new_dir,'test'+partition_id))
    else:
        
        print('Processing training data ...')
        Parallel(n_jobs=num_cores, verbose=50)(delayed(
            process_tiff)(file=partition['training'][file], 
                          raw_path=raw_path, 
                          new_dir=new_dir, 
                          label='train', 
                          partition_id=partition_id, 
                          tiff_size=tiff_size, 
                          resize=resize,
                          target_size=target_size,
                          local_filter=local_filter,
                          global_filter=global_filter,
                          save_filtered=save_filtered) for file in range(len(partition['training'])))
         
        print('Processing validation data ...')
        Parallel(n_jobs=num_cores, verbose=50)(delayed(
            process_tiff)(file=partition['validation'][file], 
                          raw_path=raw_path, 
                          new_dir=new_dir, 
                          label='val', 
                          partition_id=partition_id, 
                          tiff_size=tiff_size, 
                          resize=resize,
                          target_size=target_size,
                          local_filter=local_filter,
                          global_filter=global_filter,
                          save_filtered=save_filtered) for file in range(len(partition['validation'])))
        
        print('Processing testing data ...')
        Parallel(n_jobs=num_cores, verbose=50)(delayed(
            process_tiff)(file=partition['testing'][file], 
                          raw_path=raw_path, 
                          new_dir=new_dir, 
                          label='test', 
                          partition_id=partition_id, 
                          tiff_size=tiff_size, 
                          resize=resize,
                          target_size=target_size,
                          local_filter=local_filter,
                          global_filter=global_filter,
                          save_filtered=save_filtered) for file in range(len(partition['testing'])))
                          
    # Update partition directories
    new_partition = {}
    train_arr = np.empty(len(partition['training']), dtype=object)
    val_arr = np.empty(len(partition['validation']), dtype=object)
    test_arr = np.empty(len(partition['testing']), dtype=object)
    for i in range(len(partition['training'])):
        file = partition['training'][i]
        file, _ = os.path.splitext(file)
        file = file + '.npy'
        file = os.path.join(new_dir, 'train'+partition_id, file)
        train_arr[i] = file
    for i in range(len(partition['validation'])):
        file = partition['validation'][i]
        file, _ = os.path.splitext(file)
        file = file + '.npy'
        file = os.path.join(new_dir, 'val'+partition_id, file)
        val_arr[i] = file
    for i in range(len(partition['testing'])):
        file = partition['testing'][i]
        file, _ = os.path.splitext(file)
        file = file + '.npy'
        file = os.path.join(new_dir, 'test'+partition_id, file)
        test_arr[i] = file
        
    new_partition['training'] = train_arr
    new_partition['validation'] = val_arr
    new_partition['testing'] = test_arr
    
    save_dict(new_partition, os.path.join(new_dir, partition_filename))


def process_tiff(file, raw_path, new_dir, label, partition_id, tiff_size, target_size,
                 resize=True, local_filter=False, global_filter=True, save_filtered=False):
    
    stack = (sk.imread(os.path.join(raw_path, file))).astype('float32')
    
    file, ext = os.path.splitext(file)
    # if partition_id == 'A':
    stack = np.transpose(stack, (1, 2, 0))
    
    if partition_id == 'B':
        stack = subarray(stack)
    
    if local_filter:
        #  Apply local standardisation
        for j in range(stack.shape[2]):
            stack[:,:,j] = stdnorm(stack[:,:,j])
        # for j in range(stack.shape[0]):
        #     stack[j,] = stdnorm(stack[j,])
        # for j in range(stack.shape[1]):
        #     stack[:,j,] = stdnorm(stack[:,j,])
    
    if not tiff_size == target_size and resize:
        stack = (resize_volume(stack, target_size)).astype('float32')
        if partition_id == 'B':
            stack[stack < 0.] = 0.0
            stack[stack > 255.] = 255
        
    if global_filter:
        stack = stdnorm(stack)
        
    if local_filter or global_filter:
        lp = sp.scoreatpercentile(stack,0.05)
        up = sp.scoreatpercentile(stack,99.95)
        stack[stack < lp] = lp
        stack[stack > up] = up

    stack = (norm_data(stack)).astype('float32')  
    stack = (stack - 0.5) / 0.5

    if partition_id == 'B':
        stack[stack < 0.] = -1.0
        stack[stack >= 0.] = 1.0

    if not checkNAN(stack):
    
        if save_filtered:
            arr_out = os.path.join(os.path.join(new_dir,'filtered'), 
                                   label+partition_id, file+'.tiff')
            if ext == '.npy':
                sk.imsave(arr_out, (stack* 127.5 + 127.5).astype('uint8'), bigtiff=False)
            else:
                sk.imsave(arr_out, (np.transpose(stack,(2,1,0)) * 127.5 + 127.5).astype('uint8'), bigtiff=False)
        
        if partition_id == 'B':
            np.save(os.path.join(new_dir, label+partition_id, file), np.expand_dims(stack, axis=3))
        else:
            np.save(os.path.join(new_dir, label+partition_id, file), np.expand_dims(stack, axis=3))
    else:
        print('NaN detected ...')
            
def resize_volume(img, target_size=None):
    
    arr1 = np.empty([target_size[0], target_size[1], img.shape[2]], dtype='float32')
    arr2 = np.empty([target_size[0], target_size[1], target_size[2]], dtype='float32')
    
    if not img.shape[0:2] == target_size[0:2]:
        for i in range(img.shape[2]):
            arr1[:,:,i] = cv2.resize(img[:,:,i], (target_size[0], target_size[1]),
                                     interpolation=cv2.INTER_LANCZOS4)
        
        for i in range(target_size[0]):
            arr2[i,:,:] = cv2.resize(arr1[i,], (target_size[2], target_size[1]),
                                     interpolation=cv2.INTER_LANCZOS4)
            
    else:
        for i in range(target_size[0]):
            arr2[i,:,:] = cv2.resize(img[i,], (target_size[2], target_size[1]),
                                     interpolation=cv2.INTER_LANCZOS4)
        
    return arr2
