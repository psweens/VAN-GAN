import os

def epoch_sweep(args, vangan_model, plotter, test_path='', start=100, end=200, step=2, segmentation=True):
    """
    Perform a sweep of epochs for the given VANGAN model and save the resulting images using the given plotter.
    
    Args:
    - args: command-line arguments
    - vangan_model: a VANGAN object
    - plotter: a GANMonitor object
    - test_path (str): path to the directory containing the test images
    - start (int): the starting epoch number (inclusive)
    - end (int): the ending epoch number (inclusive)
    - step (int): the number of epochs to skip between each saved image
    - segmentation (bool): if True, generate segmentation images; otherwise, generate fake imaging domain images
    
    Returns:
    - None
    """

    test_path = '/mnt/sda/VS-GAN_deepVess/testA/'
    
    for i in range(start,end+1,step):
        print(i)
        vangan_model.load_checkpoint(epoch=i,
                                        newpath='/mnt/sdb/TPLSM/Boas_DeepVess_Image_Standardisation/VG_Output/checkpoints')
    
        # Make epoch folders
        filepath = '/mnt/sdb/TPLSM/Boas_DeepVess_Image_Standardisation/Epoch_Sampling/'
        folder = os.path.join(filepath, 'e{idx}'.format(idx=i))
        if not os.path.isdir(folder):
            os.makedirs(folder)
        
        testfiles = os.listdir(test_path)
        filename = 'e{idx}_VG_'.format(idx=i)
        for file in testfiles:
            testfiles[file] = os.path.join(test_path,file)
            
        plotter.run_mapping(vangan_model, testfiles, args.INPUT_IMG_SIZE, filetext=filename, segmentation=segmentation, stride=(50,50,50), filepath=folder, padFactor=0.1)
