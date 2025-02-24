import cv2
import numpy as np
import scipy.stats as sp
import SimpleITK as sitk
from scipy.ndimage import zoom, median_filter, gaussian_filter
from utils import min_max_norm, z_score_norm
from skimage import exposure
from scipy import ndimage

# Function used for preprocessing imaging domain images
# The following is used for preprocessing raster-scanning optoacoustic mesoscopic (RSOM) image volumes
def preprocess_rsom(img, lower_thresh=0.05, upper_thresh=99.95):
    """
    Preprocesses a 3D image array using slice-wise Z-score normalization and clipping of upper and lower percentiles.

    Args:
    - img (np.ndarray): A 3D numpy array representing the image to be preprocessed.
    - lower_thresh (float): The lower percentile value to clip the image at (default: 0.05).
    - upper_thresh (float): The upper percentile value to clip the image at (default: 99.95).

    Returns:
    - np.ndarray: The preprocessed 3D numpy array.
    """

    # if img.shape[2] > 140 and img.shape[2] > 180:
    #     img = img[:, :, 20:(img.shape[2] - 20)]

    # Slice-wise Z-Score Normalisation
    for z in range(img.shape[2]):
        img[..., z] = z_score_norm(img[..., z])

    # Clipping of upper and lower percentiles
    lp = sp.scoreatpercentile(img, lower_thresh)
    up = sp.scoreatpercentile(img, upper_thresh)
    img[img < lp] = lp
    img[img > up] = up

    # img = z_score_norm(img)
    return img


# Two-photon laser scanning microscopy
def preprocess_tplsm(img):
    img = img.astype(np.float32)

    for z in range(img.shape[2]):
        img[..., z] = img[..., z] - np.median(img[..., z])

    img = min_max_norm(img)
    img = np.sqrt(img)  # TPLSM
    img = median_filter(img, size=(3, 3, 3))

    return img


# High-resolution episcopic microscopy
def gaussian_blur_3d(image, sigma_x=2, sigma_y=2, sigma_z=2):
    """
    Apply a 3D Gaussian blur to a 3D image.

    Args:
    image (np.ndarray): The 3D image array.
    sigma_x (int): Standard deviation for the Gaussian kernel along the X axis.
    sigma_y (int): Standard deviation for the Gaussian kernel along the Y axis.
    sigma_z (int): Standard deviation for the Gaussian kernel along the Z axis.

    Returns:
    np.ndarray: The blurred 3D image.
    """
    # Apply Gaussian blur considering all three dimensions simultaneously
    if sigma_z > 0:
        blurred_image = gaussian_filter(image, sigma=(sigma_x, sigma_y, sigma_z))
    else:
        blurred_image = gaussian_filter(image, sigma=(sigma_x, sigma_y))

    return blurred_image


def preprocess_hrem(img):
    img = img.astype(np.float32)
    up = sp.scoreatpercentile(img, 98.)
    img[img > up] = up
    img = gaussian_blur_3d(img)

    return img

from skimage.measure import block_reduce
# Light-sheet microscopy
def preprocess_lsm(img):
    """
    Preprocesses a 3D image array using slice-wise Z-score normalization and clipping of upper and lower percentiles.

    Args:
    - img (np.ndarray): A 3D numpy array representing the image to be preprocessed.
    - lower_thresh (float): The lower percentile value to clip the image at (default: 0.05).
    - upper_thresh (float): The upper percentile value to clip the image at (default: 99.95).

    Returns:
    - np.ndarray: The preprocessed 3D numpy array.
    """

    # 3D median filter
    img = min_max_norm(img)
    img = median_filter(img, size=3)
    img = np.sqrt(img)
    # img = z_score_norm(img)
    # up = sp.scoreatpercentile(img, 99.95)
    # img[img > up] = up

    # Clipping of upper and lower percentiles
    # Iterate over each slice along the z-axis
    # for z in range(img.shape[2]):
    #     # Select the slice at position z
    #     image = img[:, :, z]
    #
    #     # Calculate the lower and upper percentiles for the current slice
    #     up = sp.scoreatpercentile(image, 99)
    #
    #     # Apply clipping to the current slice
    #     image[image > up] = up
    #
    #     # Update the slice in the original image
    #     img[:, :, z] = image

    # img = zoom(img, zoom=[1, 1, 5], order=3)
    # img = block_reduce(img, (1, 1, 2))
    img = min_max_norm(img)

    return img


def create_circular_roi(image_shape, radius_factor=0.95):
    """
    Creates a circular region of interest (ROI) in the center of the image.

    Parameters:
        image_shape (tuple): Shape of the input image (height, width).
        radius_factor (float): Factor to determine the size of the circular region.

    Returns:
        numpy.ndarray: Binary mask with a circular ROI in the center.
    """
    height, width = image_shape[:2]
    center = (int(width / 2), int(height / 2))
    radius = int(min(center[0], center[1]) * radius_factor)  # Radius is a factor of width/height
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, center, radius, (1,), thickness=-1)
    return mask, center, radius  # Return the center and radius for cropping


def preprocess_retinal_image(image, lower_thresh=0.05, upper_thresh=99.95):
    """
    Preprocess a retinal image for vascular segmentation with N4 bias field correction and CLAHE.

    Parameters:
        image (numpy.ndarray): Input retinal image with color channels in the second dimension.
        lower_thresh (int): Lower percentile for intensity clipping.
        upper_thresh (int): Upper percentile for intensity clipping.

    Returns:
        numpy.ndarray: Preprocessed image ready for U-Net input.
    """

    # Step 1: Convert to Grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Step 2: N4 Bias Field Correction
    # Convert grayscale image to SimpleITK image
    sitk_image = sitk.GetImageFromArray(grayscale_image)

    # Perform N4 Bias Field Correction
    mask_image = sitk.OtsuThreshold(sitk_image, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_image = corrector.Execute(sitk_image, mask_image)

    # Convert back to NumPy array
    corrected_image = sitk.GetArrayFromImage(corrected_image)

    # Normalize corrected image
    corrected_image = min_max_norm(corrected_image)

    # Step 3: Apply OpenCV's CLAHE
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    enhanced_image = clahe.apply((corrected_image * 255).astype(np.uint8))

    # Step 6: Create Circular ROI Mask for Z-score Normalization
    mask, center, radius = create_circular_roi(enhanced_image.shape, radius_factor=0.92)

    # Step 5: Clipping of upper and lower percentiles
    lp = np.percentile(enhanced_image[mask == 1], lower_thresh)
    up = np.percentile(enhanced_image[mask == 1], upper_thresh)
    clipped_image = np.clip(enhanced_image, lp, up)

    # Step 4: Z-score Normalization
    clipped_image = (clipped_image - np.mean(clipped_image[mask == 1])) / np.std(clipped_image[mask == 1])
    clipped_image[mask == 0] = np.amax(clipped_image)

    # Step 7: Crop the image to the bounding box of the FOV
    x_start = max(0, center[0] - radius)
    x_end = min(clipped_image.shape[1], center[0] + radius)
    y_start = max(0, center[1] - radius)
    y_end = min(clipped_image.shape[0], center[1] + radius)

    cropped_image = clipped_image[y_start:y_end, x_start:x_end]

    return cropped_image * -1.

