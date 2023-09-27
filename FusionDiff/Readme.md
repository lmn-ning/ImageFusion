## FusionDiff

This is an implementation of the following paper.

```bash
@article{Li2023FusionDiff,
title = {FusionDiff: Multi-focus image fusion using denoising diffusion probabilistic models},
author = {Mining Li and Ronghao Pei and Tianyou Zheng and Yang Zhang and Weiwei Fu},
journal = {Expert Systems with Applications},
pages = {121664},
year = {2023},
issn = {0957-4174}
}
```
### Data preparation
If you want to make a multi-focus image dataset using RGB-D depth dataset, refer to folder 'Make_MFIF_Dataset_using_RGB-D_Dataset'.

Follow the correct folder naming format to set up your training dataset.
### Set the training and inference parameters
Set the training and inference parameters in config.json, it is very important.

 - start_epoch: Change this parameter when the program breaks to retrain the model.
 - train- - drop_last: please set drop_last=true.
 - valid - - batch_size: batch_size=1 must be set when the test set image sizes are not consistent, such as for the MFFW dataset.
 - valid - - imgSize: This parameter has no effect, so it can be set to any number.
 - valid - - generat_imgs_num: This parameter controls the number of fused images generated for a test image pair, set generat_imgs_num>1 if you want to verify the certainty of FusionDiff.
 - Condition_Noise_Predictor - - use_preTrain_model: If you want to train with a pre-trained model, set use_preTrain_model=true
 - Condition_Noise_Predictor - - preTrain_Model_path: If you want to train with a pre-trained model, you need to give the relative path of the model.

### Training

```bash
python train.py
```

### Testing with pre-trained model
You need to provide the path of the pretrained model in inference.py (line 78).

```bash
python inference.py
```

### Citing FusionDiff

If you find this work useful for your research, please cite our [paper](https://www.sciencedirect.com/science/article/pii/S0957417423021668)
