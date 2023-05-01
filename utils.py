import pickle
import cv2
import numpy as np
import tensorflow as tf
import skimage.io as sk
from skimage import exposure

def min_max_norm(data):
    """
    Perform min-max normalisation on a N-dimensional numpy array.
    
    Args:
    - data (np.ndarray): A N-dimensional numpy array containing the data to be normalised
    
    Returns:
    - np.ndarray: A N-dimensional numpy array containing the normalised data
    """
    dmin = np.min(data)
    dmax = np.max(data)
    if (dmax - dmin) == 0:
        raise ValueError("Cannot perform min-max normalization when max and min are equal.")
    return (data - dmin) / (dmax - dmin)

def min_max_norm_tf(arr, axis = None):
    """
    Performs min-max normalization on a given array using TensorFlow library.

    Args:
    - arr: A tensor, which needs to be normalized.
    - axis: (Optional) An integer specifying the axis along which to normalize. If None, the entire array will be normalized.

    Returns:
    - tensor: A normalized tensor with the same shape as the input array.
    """
    
    if axis is None:
        # Normalize entire array
        min_val = tf.reduce_min(arr)
        max_val = tf.reduce_max(arr)
        tensor = (arr - min_val) / (max_val - min_val)
    else:
        # Normalize along a specific axis
        min_val = tf.reduce_min(arr, axis=axis, keepdims=True)
        max_val = tf.reduce_max(arr, axis=axis, keepdims=True)
        tensor = (arr - min_val) / (max_val - min_val)
    
    return tensor

def rescale_arr_tf(arr, alpha = -0.5, beta = 0.5):
    """
    Rescales the values in a tensor using the alpha and beta parameters.
    alpha = -0.5, beta = 0.5: [0,1] to [-1,1]
    alpha = 1.0, beta = 2.0: [-1,1] to [0,1]

    Args:
    - arr: A tensor to rescale.
    - alpha: (Optional) A float representing the scaling factor to apply to the tensor.
    - beta: (Optional) A float representing the shift to apply to the tensor.

    Returns:
    - A rescaled tensor with the same shape as the input tensor.
    """
    return tf.math.divide_no_nan((arr + alpha), beta)

def z_score_norm(data):
    """
    Perform z-score normalisation on a one-dimensional numpy array.
    
    Args:
    - data (np.ndarray): A one-dimensional numpy array containing the data to be normalised
    
    Returns:
    - np.ndarray: A one-dimensional numpy array containing the normalised data
    """
    dstd = np.std(data)
    if dstd > 0.:
        return (data - np.mean(data)) / dstd
    else:
        raise ValueError("Cannot perform z-score normalization when the standard deviation is zero.")
        

def check_nan(arr):
    """
    Checks if there are any NaN (Not a Number) values in the input NumPy array.
    
    Args:
        arr (np.ndarray): Input NumPy array.
    
    Returns:
        bool: True if there is at least one NaN value in the array, False otherwise.
    """
    return np.any(np.isnan(arr))

def replace_nan(arr):
    """
    Replace NaN (Not a Number) values in a NumPy array with zeros.
    
    Args:
    arr (np.ndarray): A NumPy array containing NaN values.
    
    Returns:
    (np.ndarray): A NumPy array with NaN values replaced with zeros.
    """
    return tf.where(tf.math.is_nan(arr), tf.zeros_like(arr), arr)

def load_volume(file, size=(600,600,700), datatype='uint8', normalise=True):
    """
    Load a volume from a (for example) tif file and normalise it.
    
    Args:
    - file (str): path to the tif file.
    - size (Tuple[int, int, int]): volume size.
    - datatype (str): volume data type.
    - normalise (bool): flag to normalise the volume.
    
    Returns:
    - vol (np.ndarray): the loaded volume.
    
    """
    vol = (sk.imread(file)).astype(datatype)
    if normalise:
        vol = min_max_norm(vol)
    return vol

def resize_volume(img, target_size=None):
    """
    Resize a 3D volume to a target size.
    
    Args:
    img (numpy.ndarray): A 3D volume represented as a numpy array.
    target_size (tuple): A tuple of three integers representing the target size of the volume.
    
    Returns:
    numpy.ndarray: The resized 3D volume.
    """
    
    # Create two arrays to hold intermediate and final results
    arr1 = np.empty([target_size[0], target_size[1], img.shape[2]], dtype='float32')
    arr2 = np.empty([target_size[0], target_size[1], target_size[2]], dtype='float32')
    
    # If the input volume's width and height don't match the target size, resize each slice along the z-axis
    if not img.shape[0:2] == target_size[0:2]:
        for i in range(img.shape[2]):
            arr1[:,:,i] = cv2.resize(img[:,:,i], (target_size[0], target_size[1]),
                                     interpolation=cv2.INTER_LANCZOS4)
        
        for i in range(target_size[0]):
            arr2[i,:,:] = cv2.resize(arr1[i,], (target_size[2], target_size[1]),
                                     interpolation=cv2.INTER_LANCZOS4)
          
    else: # If the input volume's width and height match the target size, resize each slice along the x-axis
        for i in range(target_size[0]):
            arr2[i,:,:] = cv2.resize(img[i,], (target_size[2], target_size[1]),
                                     interpolation=cv2.INTER_LANCZOS4)
        
    return arr2

def get_vaccuum(arr, dim):
    """
    Returns the smallest subarray containing all non-zero elements in the input array along the specified dimension(s).
    
    Args:
    arr (numpy.ndarray): Input array.
    dim (int or tuple of ints): Dimension(s) along which to extract the subarray.
    
    Returns:
    numpy.ndarray: Subarray containing all non-zero elements in the input array along the specified dimension(s).
    """
    if dim == 2:
        x, y = np.nonzero(arr)
        return arr[x.min():x.max()+1, y.min():y.max()+1]
    else:
        x, y, z = np.nonzero(arr)
        return arr[x.min():x.max()+1, y.min():y.max()+1, z.min():z.max()+1]
    
def hist_equalization(img):
    """
    Applies histogram equalization to the input image.
    
    Args:
    img (numpy.ndarray): Input image.
    
    Returns:
    numpy.ndarray: Histogram equalized image.
    """
    img_cdf, bin_centers = exposure.cumulative_distribution(img)
    return np.interp(img, bin_centers, img_cdf)
        
def save_dict(di_, filename_):
    """Saves a Python dictionary object to a file using the pickle module.

    Args:
        di_ (dict): A Python dictionary object to be saved to a file.
        filename_ (str): The name of the file to save the dictionary to.
    
    Returns:
        None
    """
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    """
    Load a dictionary from a binary file using the pickle module.
    Args:
    - filename_ (str): a string representing the filename (including path) of the binary file to load.
    
    Returns:
    - A dictionary object loaded from the binary file.
    """
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

def append_dict(dict1, dict2, replace = False) -> dict:
    """
    Append items in dict2 to dict1.
    
    Args:
    - dict1 (dict): The dictionary to which items in dict2 will be appended
    - dict2 (dict): The dictionary containing items to be appended to dict1
    - replace (bool): If True, existing values in dict1 with the same key as values in dict2 will be replaced with the values from dict2
    
    Returns:
    - dict: A dictionary containing the appended items
    
    Raises:
    - TypeError: If dict1 or dict2 is not a dictionary
    """
    # Check if dict1 is a dictionary
    if not isinstance(dict1, dict):
        raise TypeError("dict1 must be a dictionary")
    # Check if dict2 is a dictionary
    if not isinstance(dict2, dict):
        raise TypeError("dict2 must be a dictionary")
    # Loop through the items in dict2
    for key, value in dict2.items():
        if replace:
            # If replace is True, replace existing values in dict1 with the same key as values in dict2
            dict1[key] = value
        else:
            # If replace is False, append the values from dict2 to the list of values for the same key in dict1
            if key not in dict1:
                dict1[key] = []
            dict1[key].append(value)
    # Return the updated dictionary
    return dict1


def get_sub_volume(image, subvol=(64, 64, 512), n_samples=1):
    """
    Extracts a sub-volume from a 4D image tensor.

    Args:
    - image (numpy.ndarray): A 4D numpy array representing the input image tensor.
    - subvol (tuple): A tuple of integers representing the shape of the sub-volume to extract.
    - n_samples (int): An integer representing the number of sub-volumes to extract.

    Returns:
    - subvol (numpy.ndarray): A numpy array of shape (subvol[0], subvol[1], subvol[2], subvol[3]) representing the sub-volume extracted from the input image tensor.
    """
    
    # Initialize features and labels with `None`
    sample = np.empty([subvol[0], subvol[1], subvol[2], subvol[3]], dtype='float32')
    
    # randomly sample sub-volume by sampling the corner voxel
    start_x = np.random.randint(image.shape[0] - subvol[0] + 1)
    start_y = np.random.randint(image.shape[1] - subvol[1] + 1)
    start_z = np.random.randint(image.shape[2] - subvol[2] + 1)

    # make copy of the sub-volume
    sample = np.copy(image[start_x: start_x + subvol[0],
                      start_y: start_y + subvol[1],
                      start_z: start_z + subvol[2], :])
    
    return sample

def get_shape(arr):
    """
    Get the shape of a nested list.
    
    Args:
        arr (list): The nested list for which to determine the shape.
    
    Returns:
        list: A list containing the size of each dimension of the nested list.
    
    Example:
        >>> arr = [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]
        >>> get_shape(arr)
        [3, 2, 2]
    """
    res = []  # create an empty list to store the shape
    while isinstance((arr), list):  # loop until the elements in arr are no longer lists
        res.append(len(arr))  # add the length of arr to the shape list
        arr = arr[0]  # set arr to the first element of arr
    return res  # return the shape list

# import tf_clahe 

# @tf.function(experimental_compile=True)  # Enable XLA
# def fast_clahe(img):
#     return tf_clahe.clahe(img, tile_grid_size=(4, 4), gpu_optimized=True)