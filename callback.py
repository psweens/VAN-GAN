import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from skimage import io

class GANMonitor(keras.callbacks.Callback):
    """A callback to generate and save images after each epoch"""

    def __init__(self, num_img=1, test_AB=None, test_BA=None, sub_img_size=None):
        self.num_img = num_img
        self.test_AB = test_AB
        self.test_BA = test_BA
        self.sub_img_size = sub_img_size

    def on_epoch_end(self, epoch, logs=None):
        
        if (epoch + 1) % 50 == 1:
            
            nfig = 4
            
            for i, img in enumerate(self.test_AB.take(self.num_img)):
                
                stich_subvolumes(self.model.gen_AB, img.numpy(), 
                                 self.sub_img_size, epoch, name='AB')
                
                img = tf.expand_dims(tf.image.random_crop(img, size=self.sub_img_size),
                                     axis=0)
                
                _, ax = plt.subplots(nfig, 2, figsize=(12, 12))
                prediction = self.model.gen_AB(img)[i].numpy()
                prediction = prediction - 1.0
                img = img - 1.0
                img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)
                prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
    
                for j in range(nfig):
                    ax[j, 0].imshow(img[:,:,j*int(img.shape[2]/nfig),0], cmap='gray')
                    ax[j, 1].imshow(prediction[:,:,j*int(img.shape[2]/nfig),0], cmap='gray')
                    ax[j, 0].set_title("Input image")
                    ax[j, 1].set_title("Translated image")
                    ax[j, 0].axis("off")
                    ax[j, 1].axis("off")
    
                
                plt.savefig("./GANMonitor/{epoch}_{i}_genAB_.png".format(epoch=epoch+1, i=i),
                                        dpi=300)
    
                # io.imsave("./GANMonitor/{epoch}_{i}_AB_input.tiff".format(epoch=epoch+1, i=i), 
                #           np.transpose(img,(2,0,1,3)), bigtiff=False, check_contrast=False)
                # io.imsave("./GANMonitor/{epoch}_{i}_AB_pred.tiff".format(epoch=epoch+1, i=i), 
                #           np.transpose(prediction,(2,0,1,3)), bigtiff=False, check_contrast=False)
                
                plt.tight_layout()
                plt.show()
                plt.close()
    
            
            for i, img in enumerate(self.test_BA.take(self.num_img)):
                
                stich_subvolumes(self.model.gen_BA, img.numpy(), 
                                 self.sub_img_size, epoch, name='BA')
                
                img = tf.expand_dims(tf.image.random_crop(img, size=self.sub_img_size),
                                     axis=0)
                
                _, ax = plt.subplots(nfig, 2, figsize=(12, 12))
                prediction = self.model.gen_BA(img)[i].numpy()
                prediction = prediction - 1.0
                img = img - 1.0
    
                img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)
                prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
                
                for j in range(nfig):
                    ax[j, 0].imshow(img[:,:,j*int(img.shape[2]/nfig),0], cmap='gray')
                    ax[j, 1].imshow(prediction[:,:,j*int(img.shape[2]/nfig),0], cmap='gray')
                    ax[j, 0].set_title("Input image")
                    ax[j, 1].set_title("Translated image")
                    ax[j, 0].axis("off")
                    ax[j, 1].axis("off")
    
    
                plt.savefig("./GANMonitor/{epoch}_{i}_genBA_.png".format(epoch=epoch+1, i=i),
                            dpi=300)
                # io.imsave("./GANMonitor/{epoch}_{i}_BA_input.tiff".format(epoch=epoch+1, i=i), 
                #           np.transpose(img,(2,0,1,3)), bigtiff=False, check_contrast=False)
                # io.imsave("./GANMonitor/{epoch}_{i}_BA_pred.tiff".format(epoch=epoch+1, i=i), 
                #           np.transpose(prediction,(2,0,1,3)), bigtiff=False, check_contrast=False)
                
                plt.tight_layout()
                plt.show()
                plt.close()
    
    
def stich_subvolumes(gen, img, subvol_size, epoch, stride=(50,50,1),
                     name=None):
    
    H, W, D, C = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    sH, sW, sD = subvol_size[0], subvol_size[1], subvol_size[2]
    step_x, step_y, step_z = stride[0], stride[1], stride[2]
    
    pix_tracker = np.zeros([H, W, D, C], dtype='uint8')
    pred = np.empty(img.shape, dtype='float32')
    # BA_pred = np.empty(img.shape, dtype='float32')
    
    for k in range(0,D-sD+1,step_z):
        for j in range(0,W-sW+1,step_y):
            for i in range(0,H-sH+1,step_x):
                
                pix_tracker[i:i+sH, j:j+sW, k:k+sD, :] += 1
                
                pred[i:i+sH, j:j+sW, k:k+sD, :] += gen(np.expand_dims(
                    img[i:i+sH, j:j+sW, k:k+sD, :], axis=0))[0]
                
                # BA_pred[i:i+sH, j:j+sW, k:k+sD, :] += gen_BA(np.expand_dims(
                #     img[i:i+sH, j:j+sW, k:k+sD, :], axis=0))[0]
    
    for i in range(0,H-sH+1,step_x):
        pix_tracker[i:i+sH,-sW:,:,:] += 1
        pred[i:i+sH,-sW:,:,:] += gen(np.expand_dims(img[i:i+sH,-sW:,:,:], axis=0))[0]         
        # BA_pred[i:i+sH,-sW:,:,:] += gen_BA(np.expand_dims(img[i:i+sH,-sW:,:,:], axis=0))[0]
    
    for j in range(0,W-sW+1,step_y):
        pix_tracker[-sH:,j:j+sW,:,:] += 1
        pred[-sH:,j:j+sW,:,:] += gen(np.expand_dims(img[-sH:,j:j+sW,:,:], axis=0))[0]         
        # BA_pred[-sH:,j:j+sW,:,:] += gen_BA(np.expand_dims(img[-sH:,j:j+sW,:,:], axis=0))[0]
        
    pix_tracker[-sH:,-sW:,:,:] += 1
    pred[-sH:,-sW:,:,:] += gen(np.expand_dims(img[-sH:,-sW:,:,:], axis=0))[0]         
    # BA_pred[-sH:,-sW:,:,:] += gen_BA(np.expand_dims(img[-sH:,-sW:,:,:], axis=0))[0]
    
    pred = np.divide(255*pred, pix_tracker).astype(np.uint8)
    io.imsave("/media/sweene01/SSD/cycleGAN_output/{epoch}_{name}_prediction.tiff".format(epoch=epoch+1, name=name), 
                          np.transpose(pred,(2,0,1,3)), 
                          bigtiff=False, check_contrast=False)
    io.imsave("/media/sweene01/SSD/cycleGAN_output/{epoch}_{name}_input.tiff".format(epoch=epoch+1, name=name), 
                          np.transpose((255*img).astype(np.uint8),(2,0,1,3)),
                          bigtiff=False, check_contrast=False)
    
    # BA_pred = np.divide(255*BA_pred, pix_tracker).astype(np.uint8)
    # io.imsave("./GANMonitor/{epoch}_BA_pred.tiff".format(epoch=epoch+1), 
    #                       np.transpose(BA_pred,(2,0,1,3)), 
    #                       bigtiff=False, check_contrast=False)
     
    # pd.DataFrame(np.squeeze(pix_tracker[:,:,300], axis=2)).to_csv("./GANMonitor/pix_tracker.csv")
    
# def get_sub_volume(image,subvol=(64,64,512),n_samples=1):

#     # Initialize features and labels with `None`
#     X = np.empty([subvol[0], subvol[1], subvol[2], subvol[3]],dtype='float32')

#     # randomly sample sub-volume by sampling the corner voxel
#     start_x = np.random.randint(image.shape[0] - subvol[0] + 1 )
#     start_y = np.random.randint(image.shape[1] - subvol[1] + 1 )
#     start_z = np.random.randint(image.shape[2] - subvol[2] + 1 )

#     # make copy of the sub-volume
#     X = np.copy(image[start_x: start_x + subvol[0],
#                       start_y: start_y + subvol[1],
#                       start_z: start_z + subvol[2], :])

#     return X