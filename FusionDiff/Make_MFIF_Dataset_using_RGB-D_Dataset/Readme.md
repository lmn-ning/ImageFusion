﻿### This folder makes multi-focus image datasets using the Depth Image Dataset (NYU-D v2)

References: 

> Zhang Y ,  Liu Y ,  Sun P , et al. IFCNN: A general image fusion framework based on convolutional neural network[J]. Information Fusion, 2020(54-):54.

Steps are as follows.

1. Download the NYU-D dataset at [NYU Depth Dataset V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html), as nyu_depth_v2_labeled.mat

2. Get RGB image and depth image according to nyu_depth_v2_label.mat, please run 

```bash
python Extract_Imgs_from_MatFile/extract_RGB_imgs.py
python Extract_Imgs_from_MatFile/extract_depth_imgs.py
```
3. A multi-focus image dataset is obtained from RGB images and depth images, where the number of generated images can be set by yourself. You can set the number of images to be generated by changing the 'dataset_num' parameter in get_MFI_dataset.py. Please run

```bash
python get_MFI_dataset.py
```
