# 🧠 VAN-GAN: Unsupervised Vascular Network Segmentation from 3D Images

**VAN-GAN** provides an accessible and efficient deep learning framework for the **segmentation of vascular networks in 3D images — without requiring annotated ground truth labels**.

---

## 🌱 Introduction

As 3D biomedical imaging improves in resolution and accessibility, segmenting vascular networks remains a major bottleneck due to manual annotation requirements. VAN-GAN addresses this using a **fully unsupervised deep learning approach** based on image-to-image translation.

It adapts and extends the CycleGAN framework to translate between real photoacoustic images and synthetic vessel labels using domain-consistent constraints and 3D residual networks.

![VAN-GAN Overview](VANGAN_Overview.jpg)

---

## 🧰 Key Features

- **3D Deep Residual U-Net** for realistic vascular structure segmentation
- **CycleGAN-style architecture** for unpaired domain translation
- **No identity loss** for simplified training
- **Synthetic training images** eliminate dependence on manual labels
- **Topological & structural constraints** for accurate domain alignment
- **Sliding window inference** for high-resolution image volumes

![Generator Architecture](Generator_Architecture.jpg)

---

## 🛠 Installation

Clone the repository:

```bash
git clone https://github.com/psweens/VAN-GAN.git
```

Install dependencies:

```bash
pip install opencv-python scikit-image tqdm tensorflow_addons joblib matplotlib
```

Ensure you have:
- Python 3.9+
- TensorFlow 2.10.1
- CUDA 11.2.2 and cuDNN 8.1.0.77

We recommend using a `conda` environment for clean setup. Follow TensorFlow [GPU install guide](https://www.tensorflow.org/install/pip) for compatibility.

---

## 📦 Environment Versions (Tested)

| Tool              | Version         |
|-------------------|-----------------|
| Ubuntu            | 22.04.2 LTS     |
| Python            | 3.9.16          |
| TensorFlow        | 2.10.1          |
| Cuda Toolkit      | 11.2.2          |
| cuDNN             | 8.1.0.77        |
| OpenCV            | 4.7.0.72        |
| scikit-image      | 0.20.0          |
| tqdm              | 4.65.0          |
| tf-addons         | 0.20.0          |
| joblib            | 1.2.0           |
| matplotlib        | 3.7.1           |

---

## 🧪 Example Dataset

A paired dataset of **simulated photoacoustic images** and **synthetic vascular segmentations** is available:

📦 [Download via University of Cambridge Repository](https://doi.org/10.17863/CAM.96379)

This dataset is ideal for training VAN-GAN in an unsupervised manner and validating predictions against known vascular structures.

---

## 🧑‍💻 Contributors

Developed by [Paul W. Sweeney](https://www.psweeney.co.uk).  
Community contributions are welcome! If you're using VAN-GAN in your work, feedback and pull requests are encouraged.

---

## 📖 Citation

If you use VAN-GAN in your research, please cite:

> [Unsupervised Segmentation of 3D Microvascular Photoacoustic Images Using Deep Generative Learning](https://doi.org/10.1002/advs.202402195)  
> Paul W. Sweeney et al., *Advanced Science*, 2024.

---

## 🧾 License

Licensed under the MIT License.

Original CycleGAN implementation adapted from [A.K. Nain's TensorFlow example](https://github.com/keras-team/keras-io/blob/master/examples/generative/cyclegan.py).

---
