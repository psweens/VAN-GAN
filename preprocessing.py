import tensorflow as tf
import numpy as np

def normalize_img(img):
    # img = tf.cast(img, dtype=tf.float32)
    # Map values in the range [-1, 1]
    # return (img / 127.5) - 1.0
    return img + 1.0


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