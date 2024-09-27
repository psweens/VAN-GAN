import pickle
import cv2
import numpy as np
import tensorflow as tf
import skimage.io as sk
from skimage import exposure
from scipy import stats
import tf_clahe
import mclahe as mc
import tensorflow_addons as tfa

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
    if (dmax - dmin) == 0.:
        raise ValueError("Cannot perform min-max normalization when max and min are equal.")
    return (data - dmin) / (dmax - dmin)
#

def min_max_norm_tf(arr, axis=None):
    """
    Performs min-max normalisation on a given array using TensorFlow library.

    Args:
    - arr: A tensor, which needs to be normalized.
    - axis: (Optional) An integer specifying the axis along which to normalise.

    Returns:
    - tensor: A normalized tensor with the same shape as the input array.
    """

    if axis is None:
        # Normalize entire array
        min_val = tf.reduce_min(arr)
        max_val = tf.reduce_max(arr)
    else:
        # Normalize along a specific axis
        min_val = tf.reduce_min(arr, axis=axis, keepdims=True)
        max_val = tf.reduce_max(arr, axis=axis, keepdims=True)

    return (arr - min_val) / (max_val - min_val)


def rescale_arr_tf(arr, alpha=-0.5, beta=0.5):
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
        return data - np.mean(data)
        # raise ValueError("Cannot perform z-score normalization when the standard deviation is zero.")


import tensorflow as tf


def z_score_norm_tf(data, epsilon=1e-8):
    """
    Perform z-score normalization on a TensorFlow tensor.

    Args:
    - data (tf.Tensor): A TensorFlow tensor containing the data to be normalized.
                        Shape should be (batch, depth, width, length, channel).
    - epsilon (float): A small value to avoid division by zero when std_data is close to zero.

    Returns:
    - tf.Tensor: A TensorFlow tensor containing the normalized data.
                 Shape will be the same as the input.
    """
    mean_data = tf.math.reduce_mean(data, axis=(1, 2, 3, 4), keepdims=True)
    std_data = tf.math.reduce_std(data, axis=(1, 2, 3, 4), keepdims=True)

    return (data - mean_data) / tf.where(std_data > epsilon, std_data, epsilon)


@tf.function
def matched_crop(self, stack, axis=None, rescale=False):
    """
    Randomly crop the input tensor `stack` (which is compose of two image stacks) along a specified axis and return the resulting cropped tensors.

    Args:
    - stack: A tensor to be cropped.
    - axis: The axis along which to crop the input tensor. If axis=1, the input tensor will be cropped horizontally; if axis=3, it will be cropped vertically. Defaults to None.
    - rescale: A Boolean value indicating whether to rescale the resulting tensor values between 0 and 1. Defaults to False.

    Returns:
    - A tuple containing two cropped tensors of the same shape as the input tensor.
    """
    if axis == 1:
        shape = (self.batch_size, 2 * self.img_size[1], self.img_size[2], 1, self.channels)
        raxis = 3
    elif axis == 3:
        shape = (self.batch_size, 1, self.img_size[2], 2 * self.img_size[3], self.channels)
        raxis = 1
        axis -= 1

    arr = tf.squeeze(tf.image.random_crop(stack, size=shape),
                     axis=raxis)
    if rescale:
        arr = min_max_norm_tf(arr)
    return tf.split(arr, num_or_size_splits=2, axis=axis)


def threshold_outliers(image_volume, threshold=6):
    """
    Thresholds outlier voxels in the input 3D image volume.

    Args:
    image_volume (np.ndarray): The input 3D image volume as a NumPy array.
    threshold (float): The z-score threshold for outlier detection.

    Returns:
    (np.ndarray): The thresholded image volume after removing outliers.
    """
    # Calculate the mean and standard deviation of the image volume
    mean_intensity = np.mean(image_volume)
    std_intensity = np.std(image_volume)

    # Calculate the z-scores for the whole image volume
    z_scores = np.abs((image_volume - mean_intensity) / std_intensity)

    # Determine the largest and smallest voxel intensities not deemed outliers
    upper_limit = np.max(image_volume[z_scores <= threshold])
    lower_limit = np.min(image_volume[z_scores <= threshold])

    # Threshold the image volume based on the upper and lower limits
    thresholded_image = np.clip(image_volume, a_min=lower_limit, a_max=upper_limit)

    return thresholded_image


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


def binarise_tensor(arr):
    """
     Binarise a TensorFlow tensor by replacing positive values with ones and non-positive values with negative ones.

     Args:
     arr (tf.Tensor): Input TensorFlow tensor to be binarised.

     Returns:
     (tf.Tensor): Binarized TensorFlow tensor with ones for positive values and negative ones for non-positive values.
     """
    return tf.where(tf.math.greater_equal(arr, tf.zeros(tf.shape(arr))),
                    tf.ones(tf.shape(arr)),
                    tf.math.negative(tf.ones(tf.shape(arr))))


def add_gauss_noise(self, img, rate):
    """
    Add Gaussian noise to a TensorFlow image tensor.

    Args:
    img (tf.Tensor): Input TensorFlow image tensor to which noise will be added.
    rate (float): Standard deviation of the Gaussian noise.

    Returns:
    (tf.Tensor): TensorFlow image tensor with added Gaussian noise and values clipped between -1.0 and 1.0.
    """
    return tf.clip_by_value(img + tf.random.normal(tf.shape(img), 0.0, rate), -1., 1.)


def clip_images(images):
    """
    Clips input images to the range of [-1, 1].

    Args:
        images: Input image batch tensor.

    Returns:
        Clipped image batch tensor.
    """
    return tf.clip_by_value(images, clip_value_min=-1.0, clip_value_max=1.0)


def load_volume(file, datatype='uint8', normalise=True):
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
            arr1[:, :, i] = cv2.resize(img[:, :, i], (target_size[0], target_size[1]),
                                       interpolation=cv2.INTER_LANCZOS4)

        for i in range(target_size[0]):
            arr2[i, :, :] = cv2.resize(arr1[i,], (target_size[2], target_size[1]),
                                       interpolation=cv2.INTER_LANCZOS4)

    else:  # If the input volume's width and height match the target size, resize each slice along the x-axis
        for i in range(target_size[0]):
            arr2[i, :, :] = cv2.resize(img[i,], (target_size[2], target_size[1]),
                                       interpolation=cv2.INTER_LANCZOS4)

    return arr2


# def get_vacuum(arr, dim=3):
#     """
#     Returns the smallest subarray containing all non-zero elements in the input array along the specified dimension(s).
#
#     Args:
#     arr (numpy.ndarray): Input array.
#     dim (int or tuple of ints): Dimension(s) along which to extract the subarray.
#
#     Returns:
#     numpy.ndarray: Subarray containing all non-zero elements in the input array along the specified dimension(s).
#     """
#     if dim == 2:
#         x, y, _ = np.nonzero(arr)
#         return arr[x.min():x.max() + 1, y.min():y.max() + 1]
#     else:
#         x, y, z, _ = np.nonzero(arr)
#         return arr[x.min():x.max() + 1, y.min():y.max() + 1, z.min():z.max() + 1]

@tf.function
def get_vacuum(arr):
    """
    Returns the smallest subarray containing all non-zero elements in the input array along the specified dimension(s).

    Args:
    arr (tensorflow.Tensor): Input array.
    dim (int or tuple of ints): Dimension(s) along which to extract the subarray.

    Returns:
    tensorflow.Tensor: Subarray containing all non-zero elements in the input array along the specified dimension(s).
    """
    non_zero_indices = tf.where(tf.math.not_equal(arr, 0))
    min_indices = tf.reduce_min(non_zero_indices, axis=0)
    max_indices = tf.reduce_max(non_zero_indices, axis=0)
    return tf.slice(arr, min_indices, max_indices - min_indices + 1)

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


def append_dict(dict1, dict2, replace=False) -> dict:
    """
    Append items in dict2 to dict1.
    
    Args: - dict1 (dict): The dictionary to which items in dict2 will be appended - dict2 (dict): The dictionary
    containing items to be appended to dict1 - replace (bool): If True, existing values in dict1 with the same key as
    values in dict2 will be replaced with the values from dict2
    
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

    Returns: - subvol (numpy.ndarray): A numpy array of shape (subvol[0], subvol[1], subvol[2], subvol[3])
    representing the sub-volume extracted from the input image tensor.
    """

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
    """
    res = []  # create an empty list to store the shape
    while isinstance(arr, list):  # loop until the elements in arr are no longer lists
        res.append(len(arr))  # add the length of arr to the shape list
        arr = arr[0]  # set arr to the first element of arr
    return res  # return the shape list


@tf.function
def fast_clahe(img, gpu_optimized=True):
    return tf_clahe.clahe(img, clip_limit=1.5, gpu_optimized=gpu_optimized)

@tf.function
def clahe_3d(image):
    """
    Applies 3D Contrast Limited Adaptive Histogram Equalization (CLAHE) to a 3D image.

    Args:
        image (tf.Tensor): Input 3D image of shape (batch_size, width, length, depth, channels).
        clip_limit (float): Clip limit for CLAHE.
        grid_size (tuple): Size of the grid for histogram equalization (depth, width, length).
        num_bins (int): Number of bins in the histogram.

    Returns:
        tf.Tensor: Processed 3D image.
    """
    # Extract dimensions
    batch_size, width, length, depth, channels = image.shape

    # Initialize a list to hold the processed slices
    processed_slices = []

    # Create a CLAHE op for each depth slice and append it to the list
    for d in range(depth):
        slice_image = image[:, :, :, d, :]

        # Apply CLAHE to the slice using fast_clahe function
        # clahe = tfa.image.median_filter2d(
        #     fast_clahe(slice_image),
        #     filter_shape=(2, 2)
        # )
        clahe = fast_clahe(slice_image)

        # Append the processed slice to the list
        processed_slices.append(clahe)

    # Stack the processed slices to form the final 3D image
    processed_image = tf.stack(processed_slices, axis=3)

    return processed_image


def save_args(args, filename):
    def format_value(value):
        if isinstance(value, tuple):
            return f"({', '.join(map(str, value))})"
        return str(value)

    # Filter out attributes that are not argparse arguments
    arg_dict = {arg: value for arg, value in vars(args).items() if not arg.startswith('_')}

    with open(filename, "w") as f:
        f.write("Command line arguments:\n")
        for arg, value in arg_dict.items():
            formatted_value = format_value(value)
            f.write(f"{arg}: {formatted_value}\n")


