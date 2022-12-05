import os
import numpy as np

from callback import stich_subvolumes

def performance(valA, valB, sub_img_size=(64,64,512,1), gen_AB=None, gen_BA=None, Bonly=None):
    
    if Bonly is None:
        for i in range(len(valA)):
                    
                    # Extract test array and filename
                    img = np.load(valA[i])
                    filename = os.path.basename(valA[i])
                    filename = os.path.splitext(os.path.split(filename)[1])[0]
                    
                    # Generate A->B predictions, stitch and save
                    stich_subvolumes(gen_AB, img, sub_img_size, name='AB_'+filename,
                                     complete=True)

                
    for i in range(len(valB)):
                
                # Extract test array and filename
                img = np.load(valB[i])
                filename = os.path.basename(valB[i])
                filename = os.path.splitext(os.path.split(filename)[1])[0]
                
                # Generate B->A predictions, stitch and save
                stich_subvolumes(gen_BA, img, sub_img_size, name='BA_'+filename,
                                 complete=True)
                
                