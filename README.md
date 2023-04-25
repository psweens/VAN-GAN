# VAN-GAN: Vessel Segmentation Generative Adversarial Network

## Introduction
Innovations in imaging hardware have led to a revolution in our ability to visualise vascular networks in 3D at high resolution. The segmentation of microvascular networks from these 3D image volumes and interpretation of their meaning in the context of physiological and pathological processes unfortunately remains a time consuming and error-prone task. Deep learning has the potential to solve this problem, but current supervised analysis frameworks require human-annotated ground truth labels. To overcome these limitations, we present an unsupervised image-to-image translation deep learning model called the *vessel segmentation generative adversarial network* (VAN-GAN). VAN-GAN integrates synthetic blood vessel networks that closely resemble real-life anatomy into its training process and learns to replicate the underlying physics of an imaging system in order to learn how to segment vasculature from 3D biomedical images. By leveraging synthetic data to reduce the reliance on manual labelling, VAN-GAN lower the barriers to entry for high-quality blood vessel segmentation to benefit imaging studies of vascular structure and function.

## Methodology
This Python package utilises image-to-image translation to segment 3D biomedical image volumes of vascular networks. Our unsupervised deep learning framework builds upon [CycleGAN](https://arxiv.org/abs/1703.10593) in several ways:
* Extend the design to 3D for image volume generation using 3D convolutions.
* Utilise a deep residual U-Net architecture for generators.
* Apply random Gaussian noise to discriminator inputs and convolution layers for improved training stability and regularisation.
* Use a modified objective loss function:
  * Introduce a structure similarity reconstruction loss between real and cycled biomedical image volumes.
  * Introduce a [spatial and topological constraint](https://arxiv.org/abs/2003.07311) between real and cycled segmentation labels.
  * Exclude identity loss.

## Installation
To install the package from source, download the latest release on the VAN-GAN repository or run the following in a terminal window:
```bash
git clone https://github.com/psweens/VAN-GAN.git
```

The required packages can be install using _pip_ in a terminal window:
```bash
pip install 
```
This command can be run in a [_conda_](https://www.anaconda.com/download/) environment.

VAN-GAN has been tested on Ubuntu 22.04.2 LTS with the following package versions:
* Tensorflow

## Code Contributors
VAN-GAN code was originally developed by [Paul W. Sweeney](www.psweeney.co.uk) who continues to actively develop the framework. VAN-GAN is an open-source tool and so would benefit from suggestions and edits by all and so community development and involvement is welcomed.

## References
If you use this code or data, we kindly ask that you please cite XX. Please check fthe following reference for more details:

## Licence
The project is licenced under the MIT Licence.

Code framework built upon the 2D cycleGAN implemention in Tensorflow by A.K. Nain (https://github.com/keras-team/keras-io/blob/master/examples/generative/cyclegan.py).
