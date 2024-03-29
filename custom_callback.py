import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage import io
# from joblib import Parallel, delayed
# import multiprocessing
from tensorflow.keras import layers
from utils import min_max_norm


class GanMonitor:
    """A callback to generate and save images after each epoch"""

    def __init__(self,
                 args,
                 dataset=None,
                 imaging_val_data=None,
                 segmentation_val_data=None,
                 process_imaging_domain=None):

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

    def save_model(self, model, epoch):
        """Save the trained model at the given epoch.

        Args:
            model (object): The VANGAN model object.
            epoch (int): The epoch number.
        """

        # if epoch > 100:
        model.gen_IS.save(os.path.join(self.model_path, "checkpoints/e{epoch}_genAB".format(epoch=epoch + 1)))
        model.gen_SI.save(os.path.join(self.model_path, "checkpoints/e{epoch}_genBA".format(epoch=epoch + 1)))
        model.disc_I.save(os.path.join(self.model_path, "checkpoints/e{epoch}_discA".format(epoch=epoch + 1)))
        model.disc_S.save(os.path.join(self.model_path, "checkpoints/e{epoch}_discB".format(epoch=epoch + 1)))

    def stitch_subvolumes(self, gen, img, subvol_size,
                          epoch=-1, stride=(25, 25, 128),
                          name=None, output_path=None, complete=False, padFactor=0.25, border_removal=True,
                          process_img=False):
        """
        Stitch together subvolumes to create a full volume prediction.

        Args:
            gen: A VANGAN generator model used for prediction.
            img: numpy.ndarray, Image data used for prediction.
            subvol_size: tuple, Size of subvolumes used for prediction.
            epoch: int, Epoch number for saving output prediction.
            stride: tuple, Size of strides used for overlapping subvolumes.
            name: str, Name of output prediction file.
            output_path: str, Path for saving output prediction.
            complete: bool, If True padding is applied to edges to account for border effects.
            padFactor: float, Padding factor used for edge padding.
            border_removal: bool, If True remove border pixels from output prediction.
            process_img: bool, If True the input image is passed through the `process_imaging_domain` function.
                         If self.process_imaging_domain is not defined, this step is skipped.

        Returns:
            pred: numpy.ndarray, Full volume prediction stitched from subvolumes.

        Raises:
            AssertionError: If the dimensions of subvol_size and img do not match.
        """
        if self.dims == 2:
            subvol_size = list(subvol_size)
            subvol_size[3] = 1
            subvol_size = tuple(subvol_size)
            stride = list(stride)
            stride[2] = 1
            stride = list(stride)

        if complete:
            xspacing = int(padFactor * img.shape[0])
            yspacing = int(padFactor * img.shape[1])
            oimgshape = img.shape
            if stride[2] == 1:
                if self.dims == 2:
                    img = np.expand_dims(img, axis=-1)
                    oimgshape = img.shape
                    zspacing = 1
                    img = np.pad(img, ((xspacing, xspacing),
                                       (yspacing, yspacing),
                                       (0, 0)), 'symmetric')
                else:
                    img = np.pad(img, ((xspacing, xspacing),
                                       (yspacing, yspacing),
                                       (0, 0),
                                       (0, 0)), 'symmetric')
            else:
                zspacing = int(padFactor * img.shape[2])
                img = np.pad(img, ((xspacing, xspacing),
                                   (yspacing, yspacing),
                                   (zspacing, zspacing),
                                   (0, 0)), 'symmetric')

        if self.dims == 2:
            H, W, D, C = img.shape[0], img.shape[1], 1, img.shape[2]
        else:
            H, W, D, C = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
        kH, kW, kD = subvol_size[1], subvol_size[2], subvol_size[3]

        if not complete or not border_removal:
            pH, pW, pD = 0, 0, 0
        else:
            pH, pW, pD = int(0.1 * kH), int(0.1 * kW), int(0.1 * kD)
            if kD == D:
                pD = 0

        if self.dims == 2:
            pix_tracker = np.zeros([H, W, C], dtype='float32')
        else:
            pix_tracker = np.zeros([H, W, D, C], dtype='float32')
        pred = np.zeros(img.shape, dtype='float32')

        sh, sw, sd = stride

        dim_out_h = int(np.floor((H - kH) / sh + 1))
        dim_out_w = int(np.floor((W - kW) / sw + 1))
        dim_out_d = int(np.floor((D - kD) / sd + 1))

        if complete:
            print(
                '\tImage size (X,Y,Z,C): %i x %i x %i x %i' % (oimgshape[0], oimgshape[1], oimgshape[2], oimgshape[3]))
            print('\tImage size w/ padding (X,Y,Z,C): %i x %i x %i x %i' % (H, W, D, C))
            print('\tSampling patch size (X,Y,Z,C): %i x %i x %i x %i' % (kH, kW, kD, 1))
            print('\tBorder artefact removal pixel width (X,Y,Z): (%i, %i, %i)' % (pH, pW, pD))
            print('\tStride pixel length (X,Y,Z): (%i, %i, %i)' % (sh, sw, sd))
            print('\tNo. of stiches (X x Y x Z): %i x %i x %i' % (dim_out_h, dim_out_w, dim_out_d))

        start_row = 0
        end_row = H
        for i in range(dim_out_h + 1):
            start_col = 0
            end_col = W
            if start_row > H - kH:
                start_row = H - kH
            if end_row < kH:
                end_row = kH

            for j in range(dim_out_w + 1):
                start_dep = 0
                end_dep = D
                if start_col > W - kW:
                    start_col = W - kW
                if end_col < kW:
                    end_col = kW

                for k in range(dim_out_d + 1):
                    if start_dep > D - kD:
                        start_dep = D - kD
                    if end_dep < kD:
                        end_dep = kD

                    # From one corner
                    pix_tracker[start_row + pH:(start_row + kH - pH), start_col + pW:(start_col + kW - pW),
                    start_dep + pD:(start_dep + kD - pD)] += 1.
                    arr = img[start_row:(start_row + kH),
                          start_col:(start_col + kW),
                          start_dep:(start_dep + kD)]

                    if process_img and self.process_imaging_domain is not None:
                        arr = self.process_imaging_domain(arr, axis=None, keepdims=False)

                    arr = gen(np.expand_dims(arr,
                                             axis=0), training=False)[0]

                    arr = arr[pH:kH - pH,
                          pW:kW - pW,
                          pD:kD - pD]

                    pred[start_row + pH:(start_row + kH - pH),
                    start_col + pW:(start_col + kW - pW),
                    start_dep + pD:(start_dep + kD - pD)] += arr

                    start_dep += sd
                    end_dep -= sd
                start_col += sw
                end_col -= sw
            start_row += sh
            end_row -= sh

        pred = np.true_divide(pred, pix_tracker)
        # pred = np.nan_to_num(pred, nan=-1.)

        if complete:
            if stride[2] == 1:
                pred = pred[xspacing:oimgshape[0] + xspacing, yspacing:oimgshape[1] + yspacing, ]
            else:
                pred = pred[xspacing:oimgshape[0] + xspacing, yspacing:oimgshape[1] + yspacing,
                       zspacing:oimgshape[2] + zspacing, ]

        pred = 255 * min_max_norm(pred)

        if not complete:
            pred = pred.astype('uint8')

        if not complete:
            if self.dims == 2:
                pred = np.squeeze(pred)
                io.imsave(os.path.join(self.model_path, "e{epoch}_{name}.tiff".format(epoch=epoch + 1, name=name)),
                          pred)
            else:
                io.imsave(os.path.join(self.model_path, "e{epoch}_{name}.tiff".format(epoch=epoch + 1, name=name)),
                          np.transpose(pred, (2, 0, 1, 3)),
                          bigtiff=False, check_contrast=False)
        else:
            if self.dims == 2:
                pred = np.squeeze(pred)
                io.imsave(os.path.join(output_path, "{name}.tiff".format(name=name)), pred)
            else:
                io.imsave(os.path.join(output_path, "{name}.tiff".format(name=name)),
                          np.transpose(pred, (2, 0, 1, 3)),
                          bigtiff=False, check_contrast=False)

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

        _, ax = plt.subplots(nfig + 1, 4, figsize=(12, 12))
        if self.dims == 2:
            nfig = 1
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
        ax[nfig, 2].hist(cycled.ravel(), bins=256, range=(np.amin(cycled), np.amax(cycled)), fc='k', ec='k',
                         density=True)
        ax[nfig, 3].hist(identity.ravel(), bins=256, range=(np.amin(identity), np.amax(identity)), fc='k', ec='k',
                         density=True)

        plt.savefig("./GANMonitor/{epoch}_{genID}.png".format(epoch=epoch + 1,
                                                              genID=filename),
                    dpi=300)

        plt.tight_layout()
        plt.show(block=False)
        plt.close()

        # Generate 3D predictions, stitch and save
        if epoch % self.period3D == 1 and outputFull and epoch > 160:
            self.stitch_subvolumes(genX, storeSample.numpy(),
                                   self.imgSize, epoch=epoch, name=sampleName, process_img=process_img)

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
                end_learning_rate=0,
                power=1)

            model.gen_S_optimizer.lr = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=args.INITIAL_LR,
                decay_steps=(args.EPOCHS - args.INITIATE_LR_DECAY) * args.train_steps,
                end_learning_rate=0,
                power=1)

            model.disc_I_optimizer.lr = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=args.INITIAL_LR,
                decay_steps=(args.EPOCHS - args.INITIATE_LR_DECAY) * args.train_steps,
                end_learning_rate=0,
                power=1)

            model.disc_S_optimizer.lr = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=args.INITIAL_LR,
                decay_steps=(args.EPOCHS - args.INITIATE_LR_DECAY) * args.train_steps,
                end_learning_rate=0,
                power=1)

        if model.checkpoint_loaded and epoch > args.INITIATE_LR_DECAY:
            model.checkpoint_loaded = False

            learning_gradient = args.INITIAL_LR / (args.EPOCHS - args.INITIATE_LR_DECAY)
            intermediate_learning_rate = learning_gradient * (args.EPOCHS - epoch)

            print('Initial learning rate: %0.8f' % intermediate_learning_rate)

            model.gen_I_optimizer.lr = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=intermediate_learning_rate,
                decay_steps=(args.EPOCHS - args.INITIATE_LR_DECAY - epoch) * args.train_steps,
                end_learning_rate=0,
                power=1)

            model.gen_S_optimizer.lr = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=intermediate_learning_rate,
                decay_steps=(args.EPOCHS - args.INITIATE_LR_DECAY - epoch) * args.train_steps,
                end_learning_rate=0,
                power=1)

            model.disc_I_optimizer.lr = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=intermediate_learning_rate,
                decay_steps=(args.EPOCHS - args.INITIATE_LR_DECAY - epoch) * args.train_steps,
                end_learning_rate=0,
                power=1)

            model.disc_S_optimizer.lr = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=intermediate_learning_rate,
                decay_steps=(args.EPOCHS - args.INITIATE_LR_DECAY - epoch) * args.train_steps,
                end_learning_rate=0,
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
        # noise = 0.9 ** (epoch + 1)
        print('Noise std: %0.5f' % noise)
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.GaussianNoise):
                layer.stddev = noise

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
        self.imagePlotter(epoch, "geSI", self.segmentation_val_data, self.segmentation_val_full_vol_data, model.gen_SI,
                          model.gen_IS, outputFull=True)

    def run_mapping(self, model, test_set, sub_img_size=(64, 64, 512, 1), segmentation=True, stride=(25, 25, 1),
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
                                       complete=True, stride=stride, padFactor=padFactor)
            else:
                print('Mapping %s ... (%i / %i)' % (filename, imgdir + 1, len(test_set)))
                # Generate segmentations, stitch and save
                self.stitch_subvolumes(model.gen_SI, img, sub_img_size, name=filetext + filename, output_path=filepath,
                                       complete=True, process_img=True, stride=stride, padFactor=padFactor)
