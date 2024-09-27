import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage import io
# from joblib import Parallel, delayed
from joblib import Parallel, delayed
from tensorflow.keras import layers
from utils import min_max_norm, fast_clahe
from custom_learning_rate import polynomial_decay, cosine_decay, CyclicalCosineLR
from scipy.ndimage import gaussian_filter


class GanMonitor:
    """A callback to generate and save images after each epoch"""

    def __init__(self,
                 args,
                 dataset=None,
                 imaging_val_data=None,
                 segmentation_val_data=None,
                 process_imaging_domain=None,
                 surface_illumination=False):

        self.imgSize = args.INPUT_IMG_SIZE
        self.imaging_val_full_vol_data = dataset.imaging_val_full_vol_data
        self.segmentation_val_full_vol_data = dataset.segmentation_val_full_vol_data
        self.imaging_val_data = imaging_val_data
        self.segmentation_val_data = segmentation_val_data
        self.process_imaging_domain = process_imaging_domain
        self.period = args.PERIOD_2D_CALLBACK
        self.period3D = args.PERIOD_3D_CALLBACK
        self.model_path = args.output_dir
        self.dims = args.DIMENSIONS
        self.surface_illumination = surface_illumination

    def save_model(self, model, epoch):
        """Save the trained model at the given epoch.

        Args:
            model (object): The VANGAN model object.
            epoch (int): The epoch number.
        """

        # if epoch > 100:
        model.gen_AB.save(os.path.join(self.model_path, "checkpoints/e{epoch}_genAB".format(epoch=epoch + 1)))
        model.gen_BA.save(os.path.join(self.model_path, "checkpoints/e{epoch}_genBA".format(epoch=epoch + 1)))
        model.disc_A.save(os.path.join(self.model_path, "checkpoints/e{epoch}_discA".format(epoch=epoch + 1)))
        model.disc_B.save(os.path.join(self.model_path, "checkpoints/e{epoch}_discB".format(epoch=epoch + 1)))

    def gaussian_weight(self, shape, sigma=1):
        """Create a Gaussian kernel."""
        if len(shape) == 2:
            x = np.linspace(-1, 1, shape[0])
            y = np.linspace(-1, 1, shape[1])
            xv, yv = np.meshgrid(x, y, indexing='ij')
            kernel = np.exp(-(xv ** 2 + yv ** 2) / (2 * sigma ** 2))
        else:
            x = np.linspace(-1, 1, shape[0])
            y = np.linspace(-1, 1, shape[1])
            z = np.linspace(-1, 1, shape[2])
            xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
            kernel = np.exp(-(xv ** 2 + yv ** 2 + zv ** 2) / (2 * sigma ** 2))
        kernel /= kernel.max()  # Normalize to have max value of 1
        kernel = kernel[..., np.newaxis]  # Add a new axis for the channel dimension
        return kernel

    def stitch_subvolumes(self, gen, img, subvol_size, epoch=-1, stride=None, name=None, output_path=None,
                          complete=False):
        """
        Stitch together subvolumes to create a full volume prediction.
        """
        # Calculate stride
        if stride is None:
            stride = [max(1, x // 2) for x in subvol_size[1:4]]

        # Adjust for 2D case
        if self.dims == 2:
            subvol_size = list(subvol_size)
            subvol_size[3] = 1  # Set depth to 1 in the 2D case
            subvol_size = tuple(subvol_size)
            stride[2] = 1

        # Padding if complete is True
        if complete:
            pad_factor = [self.imgSize[i] / img.shape[i - 1] for i in range(1, 4)]

            if subvol_size[3] == img.shape[2]:
                pad_factor[2] = 0
                stride[2] = 0

            pad = [int(0.5 * pad_factor[i] * img.shape[i]) for i in range(3)]
            if self.dims == 2:
                padding = [(p, p) for p in pad[:2]] + [(0, 0)]  # Padding for the color channel correctly
            else:
                padding = [(p, p) for p in pad] + [(0, 0)]  # Padding for the color channel correctly

            img = np.pad(img, padding, mode='reflect')

        if self.dims == 2:
            H, W, C = img.shape
            D = 1
        else:
            H, W, D, C = img.shape

        # Initialize the prediction and weight map
        pred = np.zeros(img.shape, dtype='float32')
        weight_map = np.zeros(img.shape, dtype='float32')

        # Precompute Gaussian weight
        if self.dims == 2:
            gauss_weight = self.gaussian_weight((subvol_size[1], subvol_size[2], 1),
                                                sigma=1)  # For 2D, remove extra dimension
            gauss_weight = np.squeeze(gauss_weight, axis=-1)  # Ensure gauss_weight is (height, width, channels)
        else:
            gauss_weight_shape = (
            subvol_size[1], subvol_size[2], subvol_size[3] if subvol_size[3] != img.shape[2] else 1)
            gauss_weight = self.gaussian_weight(gauss_weight_shape, sigma=1)

        if self.dims == 3 and subvol_size[3] == img.shape[2]:  # No striding in the z-dimension for 3D case
            gauss_weight = np.repeat(gauss_weight[:, :, np.newaxis, :], D, axis=2)

        # Calculate the steps for sliding window
        row_steps = range(0, H - subvol_size[1] + 1, stride[0])
        col_steps = range(0, W - subvol_size[2] + 1, stride[1])
        dep_steps = [0] if subvol_size[3] == img.shape[2] else range(0, D - subvol_size[3] + 1, stride[2])

        if complete:
            print(f"Padding applied: {pad}")
            print(f"Stride length: {stride}")
            print(f"Subvolume size: {subvol_size}")

        # Stitching process
        for start_row in row_steps:
            for start_col in col_steps:
                for start_dep in dep_steps:
                    end_row = min(start_row + subvol_size[1], H)
                    end_col = min(start_col + subvol_size[2], W)
                    end_dep = min(start_dep + subvol_size[3], D)

                    row_slice = slice(start_row, end_row)
                    col_slice = slice(start_col, end_col)
                    dep_slice = slice(start_dep, end_dep)

                    subvol = img[row_slice, col_slice, dep_slice]

                    # Determine padding needed for subvolume
                    pad_height = subvol_size[1] - subvol.shape[0]
                    pad_width = subvol_size[2] - subvol.shape[1]
                    pad_depth = subvol_size[3] - subvol.shape[2]

                    # Apply padding if necessary
                    if pad_height > 0 or pad_width > 0 or pad_depth > 0:
                        pad_dims = [(0, pad_height), (0, pad_width), (0, pad_depth)]
                        if subvol.ndim == 4:  # Add padding for the channel dimension
                            pad_dims.append((0, 0))
                        subvol = np.pad(subvol, pad_dims, mode='reflect')

                    # Expand dimensions and predict
                    subvol = np.expand_dims(subvol, axis=0)

                    # Ensure `pred_subvol` is a NumPy array
                    pred_subvol = gen(subvol, training=False)[0].numpy()  # Convert TensorFlow tensor to NumPy

                    # Accumulate predictions and weights
                    if self.dims == 2:
                        # Explicitly reshape pred_subvol and gauss_weight to avoid broadcasting issues
                        pred_subvol = pred_subvol[:end_row - row_slice.start,
                                               :end_col - col_slice.start, :]

                        gauss_weight_reshaped = gauss_weight[:end_row - row_slice.start, :end_col - col_slice.start, :]

                        np.add(pred[row_slice, col_slice, 0],
                               pred_subvol[:, :, 0] * gauss_weight_reshaped[:, :, 0],
                               out=pred[row_slice, col_slice, 0])

                        np.add(weight_map[row_slice, col_slice, 0], gauss_weight_reshaped[:, :, 0],
                               out=weight_map[row_slice, col_slice, 0])
                    else:
                        np.add.at(pred, (row_slice, col_slice, dep_slice),
                                  pred_subvol[:end_row - row_slice.start,
                                  :end_col - col_slice.start,
                                  :end_dep - dep_slice.start] * gauss_weight[:end_row - row_slice.start,
                                                                :end_col - col_slice.start,
                                                                :end_dep - dep_slice.start])

                        np.add.at(weight_map, (row_slice, col_slice, dep_slice),
                                  gauss_weight[:end_row - row_slice.start,
                                  :end_col - col_slice.start,
                                  :end_dep - dep_slice.start])

        # Normalize prediction by weight map
        np.divide(pred, weight_map, out=pred, where=weight_map != 0)

        # Remove padding from the final prediction if complete
        if complete:
            if self.dims == 2:
                pred = pred[pad[0]:H - pad[0], pad[1]:W - pad[1], :]
            else:
                pred = pred[pad[0]:H - pad[0], pad[1]:W - pad[1], pad[2]:D - pad[2], :]

        # Normalize the prediction to [0, 255]
        pred = 255 * min_max_norm(pred)
        pred = pred.astype('uint8')

        # Save the prediction as a TIFF image
        if self.dims == 2:
            pred = np.squeeze(pred)
            io.imsave(os.path.join(output_path, f"{name}.tiff"), pred)
        else:
            io.imsave(os.path.join(output_path, f"{name}.tiff"), np.transpose(pred, (2, 0, 1, 3)), bigtiff=True,
                      check_contrast=False)

    def imagePlotter(self, epoch, filename, setlist, dataset, genX, genY, nfig=6, outputFull=True, process_img=False):
        """
        Plot and save 2D sample images during training.

        Parameters:
        epoch (int): The current epoch number.
        filename (str): The filename to save the plot as.
        setlist (list): A list of filenames for samples to be plotted.
        dataset (tf.data.Dataset): The dataset containing the samples.
        genX (tf.keras.Model): The generator model.
        genY (tf.keras.Model): The inverse generator model.
        nfig (int): The number of sample images to plot.
        outputFull (bool): If True, generate and save 3D predictions.
        process_img (bool): If True and self.process_imaging_domain is not None, process the images before plotting.

        Returns:
        None
        """

        # Extract test array and filename
        sample = list(dataset.take(1))
        idx = sample[0][1]
        sample = sample[0][0]
        storeSample = tf.identity(sample)
        sampleName = setlist[idx]
        sampleName = os.path.splitext(os.path.split(sampleName)[1])[0]

        # Generate random crop of sample
        if self.dims == 2:
            sample = tf.expand_dims(
                tf.image.random_crop(sample, size=(self.imgSize[1], self.imgSize[2], self.imgSize[3])),
                axis=0)
        else:
            sample = tf.expand_dims(
                tf.image.random_crop(sample, size=(self.imgSize[1], self.imgSize[2], self.imgSize[3], self.imgSize[4])),
                axis=0)

        if process_img and self.process_imaging_domain is not None:
            sample = self.process_imaging_domain(sample)

        prediction = genX(sample, training=False)
        cycled = genY(prediction, training=False)
        identity = genY(sample, training=False)

        sample = sample[0].numpy()
        prediction = prediction[0].numpy()
        cycled = cycled[0].numpy()
        identity = identity[0].numpy()

        if self.dims == 2:
            nfig = 1
            _, ax = plt.subplots(nfig + 1, 4, figsize=(12, 12))
            ax[0, 0].imshow(sample, cmap='gray')
            ax[0, 1].imshow(prediction, cmap='gray')
            ax[0, 2].imshow(cycled, cmap='gray')
            ax[0, 3].imshow(identity, cmap='gray')
            ax[0, 0].set_title("Input image")
            ax[0, 1].set_title("Translated image")
            ax[0, 2].set_title("Cycled image")
            ax[0, 3].set_title("Identity image")
            ax[0, 0].axis("off")
            ax[0, 1].axis("off")
            ax[0, 2].axis("off")
            ax[0, 3].axis("off")
        else:
            _, ax = plt.subplots(nfig + 1, 4, figsize=(12, 12))
            for j in range(nfig):
                ax[j, 0].imshow(sample[:, :, j * int(sample.shape[2] / nfig), 0], cmap='gray')
                ax[j, 1].imshow(prediction[:, :, j * int(sample.shape[2] / nfig), 0], cmap='gray')
                ax[j, 2].imshow(cycled[:, :, j * int(sample.shape[2] / nfig), 0], cmap='gray')
                ax[j, 3].imshow(identity[:, :, j * int(sample.shape[2] / nfig), 0], cmap='gray')
                ax[j, 0].set_title("Input image")
                ax[j, 1].set_title("Translated image")
                ax[j, 2].set_title("Cycled image")
                ax[j, 3].set_title("Identity image")
                ax[j, 0].axis("off")
                ax[j, 1].axis("off")
                ax[j, 2].axis("off")
                ax[j, 3].axis("off")
        ax[nfig, 0].hist(sample.ravel(), bins=256, range=(np.amin(sample), np.amax(sample)), fc='k', ec='k',
                         density=True)
        ax[nfig, 1].hist(prediction.ravel(), bins=256, range=(np.amin(prediction), np.amax(prediction)), fc='k', ec='k',
                         density=True)
        ax[nfig, 2].hist(cycled.ravel(), bins=256, range=(np.amin(sample), np.amax(sample)), fc='k', ec='k',
                         density=True)
        ax[nfig, 3].hist(identity.ravel(), bins=256, range=(np.amin(sample), np.amax(sample)), fc='k', ec='k',
                         density=True)

        plt.savefig("./GANMonitor/{epoch}_{genID}.png".format(epoch=epoch + 1,
                                                              genID=filename),
                    dpi=300)

        plt.tight_layout()
        plt.show(block=False)
        plt.close()

        # Generate 3D predictions, stitch and save
        # if epoch % self.period3D == 1 and outputFull and epoch > 160:
        #     self.stitch_subvolumes(genX, storeSample.numpy(),
        #                            self.imgSize, epoch=epoch, name=sampleName, process_img=process_img)

    def set_learning_rate(self, model, epoch, args):
        """
        Sets the learning rate for each optimizer based on the current epoch.

        Parameters:
            model: VANGAN object
                An instance of the VANGAN class.
            epoch: int
                The current epoch number.
            args: argparse.Namespace
                An argparse namespace containing the command line arguments.

        Returns:
            None
        """

        if epoch == args.INITIATE_LR_DECAY:
            model.gen_I_optimizer.lr = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=args.INITIAL_LR,
                decay_steps=(args.EPOCHS - args.INITIATE_LR_DECAY - 10) * args.train_steps,
                end_learning_rate=2e-6,
                power=1)

            model.gen_S_optimizer.lr = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=args.INITIAL_LR,
                decay_steps=(args.EPOCHS - args.INITIATE_LR_DECAY - 10) * args.train_steps,
                end_learning_rate=2e-6,
                power=1)

            model.disc_I_optimizer.lr = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=args.INITIAL_LR,
                decay_steps=(args.EPOCHS - args.INITIATE_LR_DECAY - 10) * args.train_steps,
                end_learning_rate=2e-6,
                power=1)

            model.disc_S_optimizer.lr = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=args.INITIAL_LR,
                decay_steps=(args.EPOCHS - args.INITIATE_LR_DECAY - 10) * args.train_steps,
                end_learning_rate=2e-6,
                power=1)

    def updateDiscriminatorNoise(self, model, init_noise, epoch, args):
        """
        Update the standard deviation of the Gaussian noise layer in a VANGAN discriminator.

        Args:
            model (tf.keras.model): The Keras model to update the noise layer for.
            init_noise (float): The initial standard deviation of the noise layer.
            epoch (int): The current epoch number.
            args (argparse.Namespace): The command-line arguments containing the noise decay rate.

        Returns:
            None

        """
        if args.NO_NOISE == 0:
            decay_rate = 1.
        else:
            decay_rate = epoch / args.NO_NOISE
        noise = init_noise * (1. - decay_rate)
        if noise < 0.0:
            noise = 0.0
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.GaussianNoise):
                layer.stddev = noise
        print('Noise std: %0.5f' % noise)

    def on_epoch_start(self, model, epoch, args, logs=None):
        """
        Callback function that is called at the start of each training epoch.

        Args:
            model (tf.keras.model): The Keras model being trained.
            epoch (int): The current epoch number.
            args (argparse.Namespace): The command-line arguments containing the learning rate and noise decay rate.
            logs (Optional[Dict[str, float]]): Dictionary of logs to update during training. Defaults to None.

        Returns:
            None

        """

        self.set_learning_rate(model, epoch, args)

        self.updateDiscriminatorNoise(model.disc_I, model.layer_noise, epoch, args)
        self.updateDiscriminatorNoise(model.disc_S, model.layer_noise, epoch, args)

    def on_epoch_end(self, model, epoch, logs=None):
        """
        Callback function that is called at the end of each training epoch.

        Args:
            model (tf.keras.model): The Keras model being trained.
            epoch (int): The current epoch number.
            logs (Optional[Dict[str, float]]): Dictionary of logs to update during training. Defaults to None.

        Returns:
            None

        """

        # Generate 2D plots
        self.imagePlotter(epoch, "genIS", self.imaging_val_data, self.imaging_val_full_vol_data, model.gen_IS,
                          model.gen_SI, process_img=True)
        self.imagePlotter(epoch, "genSI", self.segmentation_val_data, self.segmentation_val_full_vol_data, model.gen_SI,
                          model.gen_IS, outputFull=True)

    def run_mapping(self, model, test_set, sub_img_size=(64, 64, 512, 1), segmentation=True, stride=None,
                    padFactor=0.25, filetext=None, filepath=''):
        """
        Runs mapping on a set of test images using the specified generator model and sub-volume size.

        Args:
            model (tf.keras.model): The generator model to use for mapping.
            test_set (List[str]): A list of file paths to the test images.
            sub_img_size (Tuple[int, int, int, int]): The size of the sub-volumes to use for mapping. Defaults to (64,64,512,1).
            segmentation (bool): A flag indicating whether to perform segmentation. Defaults to True.
            stride (Tuple[int, int, int]): The stride to use when mapping sub-volumes. Defaults to (25,25,1).
            padFactor (float): The padding factor to use when mapping sub-volumes. Defaults to 0.25.
            filetext (Optional[str]): A string to append to the output file names. Defaults to None.
            filepath (str): The output file path. Defaults to ''.

        Returns:
            None

        """

        # num_cores = int(0.8*(multiprocessing.cpu_count() - 1))
        # print('Processing training data ...')
        # Parallel(n_jobs=num_cores, verbose=50)(delayed(
        #     self.stitch_subvolumes)(gen=model.gen_IS,
        #                               img=np.load(test_set[imgdir]),
        #                               subvol_size=sub_img_size,
        #                               name=filetext+os.path.splitext(os.path.split(os.path.basename(test_set[imgdir]))[1])[0],
        #                               complete=True) for imgdir in range(len(test_set)))

        for imgdir in range(len(test_set)):
            # Extract test array and filename
            img = np.load(test_set[imgdir])
            filename = os.path.basename(test_set[imgdir])
            filename = os.path.splitext(os.path.split(filename)[1])[0]
            if segmentation:
                print('Segmenting %s ... (%i / %i)' % (filename, imgdir + 1, len(test_set)))
                # Generate segmentations, stitch and save
                self.stitch_subvolumes(model.gen_IS, img, sub_img_size, name=filetext + filename, output_path=filepath,
                                       complete=True, stride=stride)
            else:
                print('Mapping %s ... (%i / %i)' % (filename, imgdir + 1, len(test_set)))
                # Generate segmentations, stitch and save
                self.stitch_subvolumes(model.gen_SI, img, sub_img_size, name=filetext + filename, output_path=filepath,
                                       complete=True, process_img=True, stride=stride)
