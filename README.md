# WIA-LD2ND: Wavelet-based Image Alignment for Self-supervised Low-Dose CT Denoising **[MICCAI 2024]**

## Introduction
In clinical examinations and diagnoses, low-dose computed tomography (LDCT) is crucial for minimizing health risks compared with normal-dose computed tomography (NDCT). However, reducing the radiation dose compromises the signal-to-noise ratio, leading to degraded quality of CT images. To address this, we analyze LDCT denoising task based on experimental results from the frequency perspective, and then introduce a novel self-supervised CT image denoising method called WIA-LD2ND, only using NDCT data. The proposed WIA-LD2ND comprises two modules: Wavelet-based Image Alignment (WIA) and Frequency-Aware Multi-scale Loss (FAM). First, WIA is introduced to align NDCT with LDCT by mainly adding noise to the high-frequency components, which is the main difference between LDCT and NDCT. Second, to better capture high-frequency components and detailed information, Frequency-Aware Multi-scale Loss (FAM) is proposed by effectively utilizing multi-scale feature space. Extensive experiments on two public LDCT denoising datasets demonstrate that our WIA-LD2ND, only uses NDCT, outperforms existing several state-of-the-art weakly-supervised and self-supervised methods. 
## Prerequisites
- Linux
- Python 3.7
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/zhaohaoyu376/morestyle
cd wia
```

- Install [PyTorch](http://pytorch.org) and other dependencies (e.g., torchvision, [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)).

## Datasets
The 2016 AAPM-Mayo dataset can be downloaded from: [CT Clinical Innovation Center](https://ctcicblog.mayo.edu/2016-low-dose-ct-grand-challenge/)

The 2020 AAPM-Mayo dataset can be downloaded from: [cancer imaging archive](https://www.cancerimagingarchive.net/collection/ldct-and-projection-data/)
### WIA-LD2ND train/test
- train the model:
```bash
python train_wavelet.py 
```

- test the model:
```bash
python test.py
```

## Citation
If you use this code for your research, please cite our papers.
```
@article{zhao2024wia,
  title={WIA-LD2ND: Wavelet-based Image Alignment for Self-supervised Low-Dose CT Denoising},
  author={Zhao, Haoyu and Liang, Guyu and Zhao, Zhou and Du, Bo and Xu, Yongchao and Yu, Rui},
  journal={arXiv preprint arXiv:2403.11672},
  year={2024}
}
```

## Acknowledgments
Our code is inspired by [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
