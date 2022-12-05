import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage import io
from preprocessing import norm_data
from joblib import Parallel, delayed
import multiprocessing
from tensorflow.keras import layers

class GANMonitor():
    """A callback to generate and save images after each epoch"""
    def __init__(self, 
                 imgSize,
                 num_img=1, 
                 test_AB=None, 
                 test_BA=None, 
                 Alist=None, 
                 Blist=None,
                 period=5,
                 period3D=10,
                 model_path=None):
        self.imgSize = imgSize
        self.batches = num_img
        self.test_AB = test_AB
        self.test_BA = test_BA
        self.Alist = Alist
        self.Blist = Blist
        self.period = period
        self.period3D = period3D
        self.model_path = model_path
        
    def saveModel(self, model, epoch):
        
        model.gen_AB.save(os.path.join(self.model_path, "e{epoch}_genAB".format(epoch=epoch+1)))
        model.gen_BA.save(os.path.join(self.model_path, "e{epoch}_genBA".format(epoch=epoch+1)))
        model.disc_A.save(os.path.join(self.model_path, "e{epoch}_discA".format(epoch=epoch+1)))
        model.disc_B.save(os.path.join(self.model_path, "e{epoch}_discB".format(epoch=epoch+1)))
        
    def stitch_subvolumes(self, gen, img, subvol_size, 
                          filepath='/media/sweene01/5e3122f3-5d00-4bc0-b099-982762bf3999/cycleGAN_output', 
                          epoch=-1, stride=(25,25,128),
                         name=None, complete=False, padFactor=0.25):

        if complete:
            oimgshape = img.shape
            xspacing = int(padFactor*img.shape[0])
            yspacing = int(padFactor*img.shape[1])
            zspacing = int(padFactor*img.shape[2])
            if stride[2] == 1:
                img = np.pad(img, ((xspacing, xspacing),
                                   (yspacing, yspacing),
                                   (0, 0), 
                                   (0, 0)), 'symmetric')
            else:
                img = np.pad(img, ((xspacing, xspacing),
                                   (yspacing, yspacing),
                                   (zspacing, zspacing), 
                                   (0, 0)), 'symmetric')
                
        H, W, D, C = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
        kH, kW, kD = subvol_size[1], subvol_size[2], subvol_size[3]
        pH, pW, pD = int(0.1*kH), int(0.1*kW), int(0.1*kD)
        if kD == img.shape[2]:
            pD = 0
            
        if not complete:
            pH = 0
            pW = 0
            pD = 0
        
        pix_tracker = np.zeros([H, W, D, C], dtype='float32')
        pred = np.zeros(img.shape, dtype='float32')
        
        sh, sw, sd = stride
    
        dim_out_h = int(np.floor( (H - kH) / sh + 1 ))
        dim_out_w = int(np.floor( (W - kW) / sw + 1 ))
        dim_out_d = int(np.floor( (D - kD) / sd + 1 ))
        
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
                    pix_tracker[start_row+pH:(start_row+kH-pH), start_col+pW:(start_col+kW-pW), start_dep+pD:(start_dep+kD-pD)] += 1.
                    arr = gen(np.expand_dims(img[start_row:(start_row+kH), 
                                        start_col:(start_col+kW), 
                                        start_dep:(start_dep+kD)], 
                                        axis=0), training=False)[0]
                    arr = arr[pH:kH-pH,
                              pW:kW-pW,
                              pD:kD-pD]
                    
                    pred[start_row+pH:(start_row+kH-pH), 
                         start_col+pW:(start_col+kW-pW), 
                         start_dep+pD:(start_dep+kD-pD)] += arr    
                                                                                                        
                                                                            
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
                pred = pred[xspacing:oimgshape[0]+xspacing,yspacing:oimgshape[1]+yspacing,]
            else:
                pred = pred[xspacing:oimgshape[0]+xspacing,yspacing:oimgshape[1]+yspacing,zspacing:oimgshape[2]+zspacing,]
        
        pred = 255 * norm_data(pred)
        if not complete:
            pred = pred.astype('uint8')

        if not complete:
            io.imsave(os.path.join(filepath,"e{epoch}_{name}.tiff".format(epoch=epoch+1, name=name)), 
                                  np.transpose(pred,(2,0,1,3)), 
                                  bigtiff=False, check_contrast=False)
        else:
            io.imsave(os.path.join(filepath,"{name}.tiff".format(name=name)), 
                                  np.transpose(pred,(2,0,1,3)), 
                                  bigtiff=False, check_contrast=False)
        
    def imagePlotter(self, epoch, filename, setlist, dataset, genX, genY, nfig=6):
        
        for i, sample in enumerate(dataset.take(self.batches)):
            
            # Extract test array and filename
            idx = sample[1]
            sample = sample[0]
            storeSample = sample
            sampleName = setlist[idx]
            sampleName = os.path.splitext(os.path.split(sampleName)[1])[0]
            
            # Generate random crop of sample
            sample = tf.expand_dims(tf.image.random_crop(sample, size=(self.imgSize[1],self.imgSize[2],self.imgSize[3],self.imgSize[4])),
                                    axis=0)
            
            prediction = genX(sample, training=False)
            cycled = genY(prediction, training=False)
            identity = genY(sample, training=False)
        
            sample = sample[0].numpy()
            prediction = prediction[0].numpy()
            cycled = cycled[0].numpy()
            identity = identity[0].numpy()
            
            _, ax = plt.subplots(nfig+1, 4, figsize=(12, 12))
            for j in range(nfig):
                ax[j, 0].imshow(sample[:,:,j*int(sample.shape[2]/nfig),0], cmap='gray')
                ax[j, 1].imshow(prediction[:,:,j*int(sample.shape[2]/nfig),0], cmap='gray')
                ax[j, 2].imshow(cycled[:,:,j*int(sample.shape[2]/nfig),0], cmap='gray')
                ax[j, 3].imshow(identity[:,:,j*int(sample.shape[2]/nfig),0], cmap='gray')
                ax[j, 0].set_title("Input image")
                ax[j, 1].set_title("Translated image")
                ax[j, 2].set_title("Cycled image")
                ax[j, 3].set_title("Identity image")
                ax[j, 0].axis("off")
                ax[j, 1].axis("off")
                ax[j, 2].axis("off")
                ax[j, 3].axis("off")
            ax[nfig,0].hist(sample.ravel(), bins=256, range=(-1.0,1.0), fc='k', ec='k', density=True)
            ax[nfig,1].hist(prediction.ravel(), bins=256, range=(-1.0,1.0), fc='k', ec='k', density=True)
            ax[nfig,2].hist(cycled.ravel(), bins=256, range=(-1.0,1.0), fc='k', ec='k', density=True)
            ax[nfig,3].hist(identity.ravel(), bins=256, range=(-1.0,1.0), fc='k', ec='k', density=True)
    
            plt.savefig("./GANMonitor/{epoch}_{i}_{genID}.png".format(epoch=epoch+1, 
                                                                     i=i,
                                                                     genID=filename),
                        dpi=300)
            
            plt.tight_layout()
            plt.show(block=False)
            plt.close()
            
            # Generate 3D predictions, stitch and save
            if (epoch) % self.period3D == 1:
                self.stitch_subvolumes(genX, storeSample.numpy(), 
                                      self.imgSize, epoch=epoch, name=sampleName)
                
    def set_learning_rate(self, model, epoch, args):
        
        if epoch == args.INITIATE_LR_DECAY:
            
            model.gen_A_optimizer.lr = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=args.INITIAL_LR,
                                                                                      decay_steps=(args.EPOCHS-args.INITIATE_LR_DECAY)*args.train_steps,
                                                                                      end_learning_rate=0,
                                                                                      power=1)
            
            model.gen_B_optimizer.lr = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=args.INITIAL_LR,
                                                                                      decay_steps=(args.EPOCHS-args.INITIATE_LR_DECAY)*args.train_steps,
                                                                                      end_learning_rate=0,
                                                                                      power=1)
            
            model.disc_A_optimizer.lr = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=args.INITIAL_LR,
                                                                                      decay_steps=(args.EPOCHS-args.INITIATE_LR_DECAY)*args.train_steps,
                                                                                      end_learning_rate=0,
                                                                                      power=1)
            
            model.disc_B_optimizer.lr = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=args.INITIAL_LR,
                                                                                      decay_steps=(args.EPOCHS-args.INITIATE_LR_DECAY)*args.train_steps,
                                                                                      end_learning_rate=0,
                                                                                      power=1)
            
        if model.checkpoint_loaded and epoch > args.INITIATE_LR_DECAY:
            
            model.checkpoint_loaded = False
            
            learning_gradient = args.INITIAL_LR / (args.EPOCHS - args.INITIATE_LR_DECAY)
            intermediate_learning_rate = learning_gradient * (args.EPOCHS - epoch)
            
            print('Initial learning rate: %0.8f' %intermediate_learning_rate)
            
            model.gen_A_optimizer.lr = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=intermediate_learning_rate,
                                                                                      decay_steps=(args.EPOCHS-args.INITIATE_LR_DECAY-epoch)*args.train_steps,
                                                                                      end_learning_rate=0,
                                                                                      power=1)
            
            model.gen_B_optimizer.lr = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=intermediate_learning_rate,
                                                                                      decay_steps=(args.EPOCHS-args.INITIATE_LR_DECAY-epoch)*args.train_steps,
                                                                                      end_learning_rate=0,
                                                                                      power=1)
            
            model.disc_A_optimizer.lr = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=intermediate_learning_rate,
                                                                                      decay_steps=(args.EPOCHS-args.INITIATE_LR_DECAY-epoch)*args.train_steps,
                                                                                      end_learning_rate=0,
                                                                                      power=1)
            
            model.disc_B_optimizer.lr = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=intermediate_learning_rate,
                                                                                      decay_steps=(args.EPOCHS-args.INITIATE_LR_DECAY-epoch)*args.train_steps,
                                                                                      end_learning_rate=0,
                                                                                      power=1)
            
        
    def updateDiscriminatorNoise(self, model, init_noise, epoch, args):
        if args.NO_NOISE == 0:
            decay_rate = 1.
        else:
            decay_rate = epoch / args.NO_NOISE
        noise = init_noise * (1. - decay_rate)
        # noise = 0.9 ** (epoch + 1)
        print('Noise std: %0.5f' %noise)
        for layer in model.layers:
            if type(layer) == layers.GaussianNoise:
                if noise > 0.:
                    layer.stddev = noise
                else:
                    layer.stddev = 0.0                
                
    def on_epoch_start(self, model, epoch, args, logs=None):
        
        self.set_learning_rate(model, epoch, args)
        
        self.updateDiscriminatorNoise(model.disc_A, model.layer_noise, epoch, args)
        self.updateDiscriminatorNoise(model.disc_B, model.layer_noise, epoch, args)
        

    def on_epoch_end(self, model, epoch, args, logs=None):

        # Generate 2D plots
        self.imagePlotter(epoch, "genAB", self.Alist, self.test_AB, model.gen_AB, model.gen_BA)
        self.imagePlotter(epoch, "genBA", self.Blist, self.test_BA, model.gen_BA, model.gen_AB)
        
    def run_mapping(self, model, test_set, sub_img_size=(64,64,512,1), segmentation=True, stride=(25,25,1), padFactor=0.25, filetext=None, filepath='/media/sweene01/5e3122f3-5d00-4bc0-b099-982762bf3999/cycleGAN_output'):
        
        # num_cores = int(0.8*(multiprocessing.cpu_count() - 1))
        # print('Processing training data ...')
        # Parallel(n_jobs=num_cores, verbose=50)(delayed(
        #     self.stitch_subvolumes)(gen=model.gen_AB, 
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
                print('Segmenting '+filename+' ...')
                # Generate segmentations, stitch and save
                self.stitch_subvolumes(model.gen_AB, img, sub_img_size, filepath=filepath, name=filetext+filename,
                                  complete=True, padFactor=padFactor)
            else:
                print('Mapping '+filename+' ...')
                # Generate segmentations, stitch and save
                self.stitch_subvolumes(model.gen_BA, img, sub_img_size, filepath=filepath, name=filetext+filename,
                                  complete=True, stride=stride, padFactor=padFactor)
            
            
                