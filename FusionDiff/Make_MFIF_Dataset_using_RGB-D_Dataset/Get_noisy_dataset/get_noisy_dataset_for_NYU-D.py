# Function: Gaussian noise, Impulse noise, Poisson noise, Rayleigh noise, and uniform noise were added to the NYU-D dataset


import cv2
import numpy as np
import os
from tqdm import tqdm
from numpy import random


def add_GaussNoise(img):
    kernal_size = 5
    sigma = 3
    gauss_img = cv2.GaussianBlur(src=img, ksize=(kernal_size, kernal_size), sigmaX=sigma, sigmaY=sigma, borderType=1)
    return gauss_img


def add_ImpulseNoise(img):
    prob = 0.001
    noisy_img = np.zeros(img.shape, np.uint8)
    thres = 1 - prob
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rdn = np.random.rand()
            if rdn < prob:
                noisy_img[i][j] = 0
            elif rdn > thres:
                noisy_img[i][j] = 255
            else:
                noisy_img[i][j] = img[i][j]
    return noisy_img


def add_PoissonNoise(img):
    # 计算图像像素的分布范围
    vals = len(np.unique(img))
    vals = 2 ** np.ceil(np.log2(vals))
    # 给图片添加泊松噪声
    noisy_img = np.random.poisson(img * vals) / float(vals)
    return noisy_img


def add_RayleighNoise(img):
    noise_std = 5
    noise = np.random.rayleigh(noise_std, img.shape)
    rayleigh_img = img + noise
    noisy_img = np.uint8(cv2.normalize(rayleigh_img, None, 0, 255, cv2.NORM_MINMAX))
    return noisy_img


def add_UniformNoise(img):
    mean, sigma = 1, 5
    a = 2 * mean - np.sqrt(12 * sigma)
    b = 2 * mean + np.sqrt(12 * sigma)
    noiseUniform = np.random.uniform(a, b, img.shape)
    imgUniformNoise = img + noiseUniform
    noisy_img = np.uint8(cv2.normalize(imgUniformNoise, None, 0, 255, cv2.NORM_MINMAX))
    return noisy_img


if __name__ == '__main__':
    dataset_path = "../MFI_dataset/NYU-D_100"
    source_1_path = os.path.join(dataset_path, "source_1")
    source_2_path = os.path.join(dataset_path, "source_2")
    fusion_path = os.path.join(dataset_path, "full_clear")
    source_1_names = os.listdir(source_1_path)
    source_2_names = os.listdir(source_2_path)
    fusion_names = os.listdir(fusion_path)
    assert len(source_1_names) == len(source_2_names) == len(fusion_names)

    noisy_dataset_path = "Noisy_dataset/NYU-D_100"
    noisy_source1_path = os.path.join(noisy_dataset_path, "source_1")
    noisy_source2_path = os.path.join(noisy_dataset_path, "source_2")
    noisy_fusion_path = os.path.join(noisy_dataset_path, "full_clear")
    if not os.path.isdir(noisy_source1_path):
        os.makedirs(noisy_source1_path)
    if not os.path.isdir(noisy_source2_path):
        os.makedirs(noisy_source2_path)
    if not os.path.isdir(noisy_fusion_path):
        os.makedirs(noisy_fusion_path)

    number = 1
    for i in tqdm(range(len(source_1_names)), desc="process"):
        img1 = cv2.imread(os.path.join(source_1_path, source_1_names[i]))
        img2 = cv2.imread(os.path.join(source_2_path, source_2_names[i]))
        fusion_img = cv2.imread(os.path.join(fusion_path, fusion_names[i]))

        gauss_img1 = add_GaussNoise(img1)
        gauss_img2 = add_GaussNoise(img2)
        gauss_img_name = str(number) + "_gauss.jpg"
        cv2.imwrite(os.path.join(noisy_source1_path, gauss_img_name), gauss_img1)
        cv2.imwrite(os.path.join(noisy_source2_path, gauss_img_name), gauss_img2)
        cv2.imwrite(os.path.join(noisy_fusion_path, gauss_img_name), fusion_img)

        impulse_img1 = add_ImpulseNoise(img1)
        impulse_img2 = add_ImpulseNoise(img2)
        impulse_img_name = str(number) + "_impulse.jpg"
        cv2.imwrite(os.path.join(noisy_source1_path, impulse_img_name), impulse_img1)
        cv2.imwrite(os.path.join(noisy_source2_path, impulse_img_name), impulse_img2)
        cv2.imwrite(os.path.join(noisy_fusion_path, impulse_img_name), fusion_img)

        poisson_img1 = add_PoissonNoise(img1)
        poisson_img2 = add_PoissonNoise(img2)
        poisson_img_name = str(number) + "_poisson.jpg"
        cv2.imwrite(os.path.join(noisy_source1_path, poisson_img_name), poisson_img1)
        cv2.imwrite(os.path.join(noisy_source2_path, poisson_img_name), poisson_img2)
        cv2.imwrite(os.path.join(noisy_fusion_path, poisson_img_name), fusion_img)

        rayleigh_img1 = add_RayleighNoise(img1)
        rayleigh_img2 = add_RayleighNoise(img2)
        rayleigh_img_name = str(number) + "_rayleigh.jpg"
        cv2.imwrite(os.path.join(noisy_source1_path, rayleigh_img_name), rayleigh_img1)
        cv2.imwrite(os.path.join(noisy_source2_path, rayleigh_img_name), rayleigh_img2)
        cv2.imwrite(os.path.join(noisy_fusion_path, rayleigh_img_name), fusion_img)

        uniform_img1 = add_UniformNoise(img1)
        uniform_img2 = add_UniformNoise(img2)
        uniform_img_name = str(number) + "_uniform.jpg"
        cv2.imwrite(os.path.join(noisy_source1_path, uniform_img_name), uniform_img1)
        cv2.imwrite(os.path.join(noisy_source2_path, uniform_img_name), uniform_img2)
        cv2.imwrite(os.path.join(noisy_fusion_path, uniform_img_name), fusion_img)

        number += 1
