import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage import io
# from joblib import Parallel, delayed
from joblib import Parallel, delayed
from tensorflow.keras import layers
from utils import min_max_norm
from scipy.ndimage import gaussian_filter
import scipy.signal
from typing import Tuple, Optional, Dict, List


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

    def gaussian_window(self, shape, sigma=1):
        """Create a Gaussian window for 2D or 3D data."""
        if len(shape) == 2:
            x = np.linspace(-1, 1, shape[0])
            y = np.linspace(-1, 1, shape[1])
            xv, yv = np.meshgrid(x, y, indexing='ij')
            kernel = np.exp(-(xv ** 2 + yv ** 2) / (2 * sigma ** 2))
        elif len(shape) == 3:
            x = np.linspace(-1, 1, shape[0])
            y = np.linspace(-1, 1, shape[1])
            z = np.linspace(-1, 1, shape[2])
            xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
            kernel = np.exp(-(xv ** 2 + yv ** 2 + zv ** 2) / (2 * sigma ** 2))
        else:
            raise ValueError("Unsupported shape length: expected 2 or 3 dimensions")
        kernel /= kernel.max()  # Normalize so the max value is 1
        kernel = kernel[..., np.newaxis]  # Add a new axis for the channel dimension
        return kernel

    def hann_window(self, shape):
        """Create a Hann window for 2D or 3D data."""
        if len(shape) == 2:
            w1 = np.hanning(shape[0])
            w2 = np.hanning(shape[1])
            kernel = np.outer(w1, w2)
        elif len(shape) == 3:
            w1 = np.hanning(shape[0])
            w2 = np.hanning(shape[1])
            w3 = np.hanning(shape[2])
            # Outer product to form a 3D window:
            kernel = np.outer(w1, w2).reshape(shape[0], shape[1], 1) * w3.reshape(1, 1, shape[2])
        else:
            raise ValueError("Unsupported shape length: expected 2 or 3 dimensions")
        kernel /= kernel.max()
        kernel = kernel[..., np.newaxis]
        return kernel

    def tukey_window(self, shape, alpha=0.5):
        """Create a Tukey window for 2D or 3D data."""
        if len(shape) == 2:
            w1 = scipy.signal.tukey(shape[0], alpha=alpha)
            w2 = scipy.signal.tukey(shape[1], alpha=alpha)
            kernel = np.outer(w1, w2)
        elif len(shape) == 3:
            w1 = scipy.signal.tukey(shape[0], alpha=alpha)
            w2 = scipy.signal.tukey(shape[1], alpha=alpha)
            w3 = scipy.signal.tukey(shape[2], alpha=alpha)
            kernel = np.outer(w1, w2).reshape(shape[0], shape[1], 1) * w3.reshape(1, 1, shape[2])
        else:
            raise ValueError("Unsupported shape length: expected 2 or 3 dimensions")
        kernel /= kernel.max()
        kernel = kernel[..., np.newaxis]
        return kernel

    def get_window(self, window_type, shape, **kwargs):
        """
        Returns a window (weighting function) for the given shape and type.

        window_type: one of 'gaussian', 'hann', or 'tukey'
        shape: tuple of dimensions (for 2D, e.g., (height, width); for 3D, (height, width, depth))
        kwargs: extra parameters (e.g., sigma for Gaussian, alpha for Tukey)
        """
        window_type = window_type.lower()
        if window_type == 'gaussian':
            sigma = kwargs.get('sigma', 1)
            return self.gaussian_window(shape, sigma=sigma)
        elif window_type == 'hann':
            return self.hann_window(shape)
        elif window_type == 'tukey':
            alpha = kwargs.get('alpha', 0.5)
            return self.tukey_window(shape, alpha=alpha)
        else:
            raise ValueError("Unsupported window type. Choose 'gaussian', 'hann', or 'tukey'.")

    def stitch_subvolumes(
            self,
            gen,
            img: np.ndarray,
            subvol_size: Tuple[int, int, int, int],
            *,
            epoch: int = -1,
            stride: Optional[Tuple[int, int, int]] = None,
            name: Optional[str] = None,
            output_path: Optional[str] = None,
            complete: bool = False,
            window_type: str = 'tukey',
            window_params: Optional[Dict] = None,
            batch_size: int = 16,
    ):
        """Stitch together sub‑volumes to create a full‑volume prediction using an
        arbitrary apodisation window **with batched inference**.

        Parameters
        ----------
        gen : tf.keras.Model | Callable
            The neural network (or generator function) used for inference.
        img : np.ndarray
            Input image/volume. Shape **(H, W, D, C)** for 3‑D or **(H, W, C)** for 2‑D.
        subvol_size : tuple
            Nominal sub‑volume size **(N, h, w, d)** where **N** is the batch axis –
            retained for backwards compatibility but ignored internally.
        stride : tuple | None, optional
            Sliding‑window step. Defaults to **½** of each spatial dim.
        name : str, optional
            Base name of the output TIFF.
        output_path : str | Path, optional
            Directory for the TIFF.
        complete : bool, default = ``False``
            If *True*, reflect‑pads *img* so every pixel/voxel is covered.
        window_type : {'tukey', 'hann', 'gaussian'}, default = 'tukey'
            Type of apodisation window.
        window_params : dict, optional
            Extra kwargs for the chosen window.
        batch_size : int, default = 8
            Maximum number of patches processed in one forward pass.
        """

        # ---------------------------------------------------------------------
        # 0. Input sanitisation & defaults
        # ---------------------------------------------------------------------
        if window_params is None:
            window_params = {}

        if stride is None:
            stride = [max(1, x // 2) for x in subvol_size[1:4]]

        # 2‑D convenience adjustments ------------------------------------------------
        if self.dims == 2:
            subvol_size = list(subvol_size)
            subvol_size[3] = 1  # depth = 1
            subvol_size = tuple(subvol_size)
            stride[2] = 1

        # ---------------------------------------------------------------------
        # 1. Optional padding to guarantee full coverage
        # ---------------------------------------------------------------------
        if complete:
            pad_factor = [self.imgSize[i] / img.shape[i - 1] for i in range(1, 4)]
            if subvol_size[3] == img.shape[2]:
                pad_factor[2] = 0
                stride[2] = 0
            pad = [int(0.5 * pad_factor[i] * img.shape[i]) for i in range(3)]
            if self.dims == 2:
                padding = [(pad[0], pad[0]), (pad[1], pad[1]), (0, 0)]
            else:
                padding = [(pad[0], pad[0]), (pad[1], pad[1]), (pad[2], pad[2]), (0, 0)]
            img = np.pad(img, padding, mode='reflect')
        else:
            pad = [0, 0, 0]

        # ---------------------------------------------------------------------
        # 2. Shapes & buffers
        # ---------------------------------------------------------------------
        if self.dims == 2:
            H, W, C = img.shape
            D = 1
        else:
            H, W, D, C = img.shape

        pred = np.zeros_like(img, dtype=np.float32)
        weight_map = np.zeros_like(img, dtype=np.float32)

        # ---------------------------------------------------------------------
        # 3. Pre‑compute the apodisation window (NumPy + TensorFlow)
        # ---------------------------------------------------------------------
        if self.dims == 2:
            win_shape = (subvol_size[1], subvol_size[2], 1)
        else:
            win_shape = (
                subvol_size[1],
                subvol_size[2],
                subvol_size[3] if subvol_size[3] != img.shape[2] else 1,
            )

        window_np = self.get_window(window_type, win_shape, **window_params).astype(np.float32)
        window_tf = tf.convert_to_tensor(window_np, dtype=tf.float32)  # on‑device copy

        # If the sub‑volume spans the full depth, repeat window along z
        if self.dims == 3 and subvol_size[3] == img.shape[2]:
            window_np = np.repeat(window_np[:, :, np.newaxis, :], D, axis=2)
            window_tf = tf.repeat(window_tf[:, :, tf.newaxis, :], D, axis=2)

        # ---------------------------------------------------------------------
        # 4. Generate sliding‑window grid
        # ---------------------------------------------------------------------
        row_steps = list(range(0, H - subvol_size[1] + 1, stride[0]))
        col_steps = list(range(0, W - subvol_size[2] + 1, stride[1]))
        if subvol_size[3] == img.shape[2]:
            dep_steps = [0]
        else:
            dep_steps = list(range(0, D - subvol_size[3] + 1, stride[2]))

        # ---------------------------------------------------------------------
        # 5. Batched inference helpers
        # ---------------------------------------------------------------------
        batch_subvols: List[np.ndarray] = []
        batch_meta: List[Tuple[slice, slice, slice, Tuple[int, int, int]]] = []

        def flush_batch():
            """Run accumulated batch through *gen* and scatter weighted predictions."""
            nonlocal batch_subvols, batch_meta, pred, weight_map
            if not batch_subvols:
                return

            # -- Stack and optional preprocessing ---------------------------------
            batch_arr = np.stack(batch_subvols, axis=0).astype(np.float32)  # (B, h, w, d, C)
            if self.process_imaging_domain is not None:
                batch_arr = self.process_imaging_domain(batch_arr)

            batch_tf = tf.convert_to_tensor(batch_arr, dtype=tf.float32)
            preds_tf = gen(batch_tf, training=False)  # forward pass
            preds_tf *= window_tf  # weight **after** inference (GPU)
            preds_np = preds_tf.numpy()  # back to CPU once

            # -- Scatter weighted predictions and window --------------------------
            for p, (r_slice, c_slice, d_slice, orig_shape) in zip(preds_np, batch_meta):
                hr, wr, dr = orig_shape
                if self.dims == 2:
                    pred[r_slice, c_slice, 0] += p[:hr, :wr, 0]
                    weight_map[r_slice, c_slice, 0] += window_np[:hr, :wr, 0]
                else:
                    pred[r_slice, c_slice, d_slice] += p[:hr, :wr, :dr]
                    weight_map[r_slice, c_slice, d_slice] += window_np[:hr, :wr, :dr]

            batch_subvols.clear()
            batch_meta.clear()

        # ------------------------------------------------------------------
        # 6. Traverse grid & accumulate patches ----------------------------
        # ------------------------------------------------------------------
        for start_row in row_steps:
            for start_col in col_steps:
                for start_dep in dep_steps:
                    end_row = min(start_row + subvol_size[1], H)
                    end_col = min(start_col + subvol_size[2], W)
                    end_dep = min(start_dep + subvol_size[3], D)

                    r_slice = slice(start_row, end_row)
                    c_slice = slice(start_col, end_col)
                    d_slice = slice(start_dep, end_dep)

                    subvol = img[r_slice, c_slice, d_slice]

                    # Pad to nominal size where necessary -------------------
                    pad_h = subvol_size[1] - subvol.shape[0]
                    pad_w = subvol_size[2] - subvol.shape[1]
                    pad_d = subvol_size[3] - subvol.shape[2]
                    if pad_h or pad_w or pad_d:
                        pad_dims = [(0, pad_h), (0, pad_w), (0, pad_d)]
                        if subvol.ndim == 4:
                            pad_dims.append((0, 0))
                        subvol = np.pad(subvol, pad_dims, mode='reflect')

                    batch_subvols.append(subvol)
                    batch_meta.append((r_slice, c_slice, d_slice,
                                       (end_row - start_row,
                                        end_col - start_col,
                                        end_dep - start_dep)))

                    if len(batch_subvols) == batch_size:
                        flush_batch()

        # Flush any remainder -----------------------------------------------------
        flush_batch()

        # ------------------------------------------------------------------
        # 7. Normalise by accumulated weights ------------------------------------
        # ------------------------------------------------------------------
        np.divide(pred, weight_map, out=pred, where=weight_map != 0)

        # ------------------------------------------------------------------
        # 8. Remove padding if requested -----------------------------------------
        # ------------------------------------------------------------------
        if complete:
            if self.dims == 2:
                pred = pred[pad[0]:H - pad[0], pad[1]:W - pad[1], :]
            else:
                pred = pred[pad[0]:H - pad[0], pad[1]:W - pad[1], pad[2]:D - pad[2], :]

        # ------------------------------------------------------------------
        # 9. Normalise to 0‑255 and save TIFF -------------------------------------
        # ------------------------------------------------------------------
        pred = 255 * min_max_norm(pred)
        pred = pred.astype(np.uint8)

        if self.dims == 2:
            io.imsave(os.path.join(output_path, f"{name}.tiff"), np.squeeze(pred))
        else:
            io.imsave(
                os.path.join(output_path, f"{name}.tiff"),
                np.transpose(pred, (2, 0, 1, 3)),  # (depth, H, W, C)
                bigtiff=True,
                check_contrast=False,
            )

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
        # if epoch % self.period3D == 1 and outputFull and epoch > 180:
        #     self.stitch_subvolumes(genX, storeSample.numpy(), self.imgSize, epoch=epoch, name=sampleName)

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
                decay_steps=(args.EPOCHS - args.INITIATE_LR_DECAY) * args.train_steps,
                end_learning_rate=2e-8,
                power=1)

            model.gen_S_optimizer.lr = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=args.INITIAL_LR,
                decay_steps=(args.EPOCHS - args.INITIATE_LR_DECAY) * args.train_steps,
                end_learning_rate=2e-8,
                power=1)

            model.disc_I_optimizer.lr = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=args.INITIAL_LR,
                decay_steps=(args.EPOCHS - args.INITIATE_LR_DECAY) * args.train_steps,
                end_learning_rate=2e-8,
                power=1)

            model.disc_S_optimizer.lr = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=args.INITIAL_LR,
                decay_steps=(args.EPOCHS - args.INITIATE_LR_DECAY) * args.train_steps,
                end_learning_rate=2e-8,
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
        print('%s Noise Std: %0.5f' % (model.discriminator_name, noise))

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
                self.stitch_subvolumes(model.gen_IS, img, sub_img_size, name=filetext + filename,
                                       complete=True, stride=stride, output_path=filepath)
            else:
                print('Mapping %s ... (%i / %i)' % (filename, imgdir + 1, len(test_set)))
                # Generate segmentations, stitch and save
                self.stitch_subvolumes(model.gen_SI, img, sub_img_size, name=filetext + filename,
                                       complete=True, process_img=True, stride=stride, output_path=filepath)
