# Function: generate fusion images using FusionDiff

import json
import os
import time
import torch
from torch.utils.data import DataLoader

from dataset import MFI_Dataset
from Diffusion import GaussianDiffusion
from Condition_Noise_Predictor.UNet import NoisePred

device = "cuda:0" if torch.cuda.is_available() else "cpu"


# Gets the filename without the extension
def get_model_name(model_path):
    model_name_expend = os.path.basename(model_path)
    return model_name_expend.split(".")[0]


# Inference use pretrain_model
def valid(config_path, model_path, timestr):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # DDPM
    T = config["diffusion_model"]["T"]
    beta_schedule_type = config["diffusion_model"]["beta_schedule_type"]
    add_noise = config["diffusion_model"]["add_noise"]
    diffusion = GaussianDiffusion(T, beta_schedule_type)

    # valid dataset
    valid_datasePath = config["dataset"]["valid"]["path"]
    valid_phase = config["dataset"]["valid"]["phase"]
    valid_batch_size = config["dataset"]["valid"]["batch_size"]
    valid_use_dataTransform = config["dataset"]["valid"]["use_dataTransform"]
    valid_dataset = config["dataset"]["valid"]["resize"]
    valid_imgSize = config["dataset"]["valid"]["imgSize"]
    valid_shuffle = config["dataset"]["valid"]["shuffle"]
    valid_drop_last = config["dataset"]["valid"]["drop_last"]
    valid_dataset = MFI_Dataset(valid_datasePath, phase=valid_phase, use_dataTransform=valid_use_dataTransform,
                                resize=valid_dataset, imgSzie=valid_imgSize)
    valid_dataloader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=valid_shuffle,
                                  drop_last=valid_drop_last)
    assert len(valid_dataset) % valid_batch_size == 0, "please reset valid_batch_size"
    valid_step_sum = len(valid_dataloader)

    # Load noise_pred_model
    print(f"device = {device}")
    in_channels = config["Condition_Noise_Predictor"]["UNet"]["in_channels"]
    out_channels = config["Condition_Noise_Predictor"]["UNet"]["out_channels"]
    model_channels = config["Condition_Noise_Predictor"]["UNet"]["model_channels"]
    num_res_blocks = config["Condition_Noise_Predictor"]["UNet"]["num_res_blocks"]
    dropout = config["Condition_Noise_Predictor"]["UNet"]["dropout"]
    time_embed_dim_mult = config["Condition_Noise_Predictor"]["UNet"]["time_embed_dim_mult"]
    down_sample_mult = config["Condition_Noise_Predictor"]["UNet"]["down_sample_mult"]
    model = NoisePred(in_channels, out_channels, model_channels, num_res_blocks, dropout, time_embed_dim_mult,
                      down_sample_mult)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model_name = get_model_name(model_path)
    concat_type = config["Condition_Noise_Predictor"]["concat_type"]

    # valid
    generat_imgs_num = config["dataset"]["valid"]["generat_imgs_num"]
    dataset_name = config["dataset"]["valid"]["dataset_name"]
    model.eval()
    with torch.no_grad():
        for valid_step, valid_images in enumerate(valid_dataloader):
            valid_sourceImg1 = valid_images[0].to(device)
            valid_sourceImg2 = valid_images[1].to(device)
            diffusion.sample(model, valid_sourceImg1, valid_sourceImg2, add_noise, concat_type, model_name, model_path,
                             generat_imgs_num, valid_step * valid_batch_size + 1, timestr, valid_step_sum, dataset_name)


if __name__ == '__main__':
    model_path = "weight/20230925_192400/epoch_0.pt"
    timestr = time.strftime('%Y%m%d_%H%M%S')
    print(f"time: {timestr}")
    config_path = "config.json"
    valid(config_path, model_path, timestr)
    print("End of valid")
