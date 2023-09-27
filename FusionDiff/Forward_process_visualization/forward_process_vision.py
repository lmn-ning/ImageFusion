# Function: Visualize the forward process

import os
import json
import torch
from torchvision import transforms
import cv2
from tqdm import tqdm

from Diffusion import GaussianDiffusion
from utils import tensor2img

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def forwardProcess(filePath, saveDir, config_path, stepSize):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    T = config["diffusion_model"]["T"]
    beta_schedule_type = config["diffusion_model"]["beta_schedule_type"]
    diffusion = GaussianDiffusion(T, beta_schedule_type)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda t: (t * 2) - 1)])
    img = cv2.imread(filePath)
    img = transform(img).to(device)
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    for i in tqdm(range(0, T, stepSize), desc="forward diffusion"):
        t = torch.Tensor([i]).long().to(device)

        image, noise = diffusion.q_sample(img, t, return_noise=True)
        image = tensor2img(image)
        noise = tensor2img(noise)

        img_file_path = os.path.join(saveDir, "image_" + str(i) + ".jpg")
        cv2.imwrite(img_file_path, image)
        noise_file_path = os.path.join(saveDir, "noise_" + str(i) + ".jpg")
        cv2.imwrite(noise_file_path, noise)


if __name__ == '__main__':
    filePath = "test.jpg"
    saveDir = "froward_process_images"
    config_path = "../config.json"
    stepSize = 20
    forwardProcess(filePath, saveDir, config_path, stepSize)
