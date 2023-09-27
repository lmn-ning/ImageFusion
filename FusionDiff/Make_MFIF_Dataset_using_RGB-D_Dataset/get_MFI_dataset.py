# Function: Multi-focus image dataset is generated using NYU-D-v2 dataset


import cv2
import numpy as np
import random
import os
from tqdm import tqdm
from numpy import random


# 获得高斯模糊图像 blurry image Ib
def get_Ib(img):
    kernal_size_list = [3, 5, 7, 9, 11, 13, 15]
    kernal_size = kernal_size_list[random.randint(0, len(kernal_size_list) - 1)]
    sigma_list = [3, 5, 7]
    sigma = sigma_list[random.randint(0, len(sigma_list) - 1)]
    Ib = cv2.GaussianBlur(src=img, ksize=(kernal_size, kernal_size), sigmaX=sigma, sigmaY=sigma, borderType=1)
    return Ib


# 获得焦点图
def get_Im(depth_img):
    Im = np.zeros_like(depth_img)
    max_depth = np.max(depth_img)

    # 随机深度阈值
    thres_depth = random.uniform(0.3, 0.7) * max_depth

    for i in range(depth_img.shape[0]):
        for j in range(depth_img.shape[1]):
            if depth_img[i][j][0] < thres_depth:
                for k in range(3):
                    Im[i][j][k] = 1

    return Im


def get_I1_I2(img, depth_img):
    Ib = get_Ib(img)
    Im = get_Im(depth_img)
    I1 = img * Im + Ib * (1 - Im)
    I2 = img * (1 - Im) + Ib * Im
    return I1, I2


if __name__ == '__main__':
    dataset_num = 100
    print(f"dataset_num = {dataset_num}")

    imags_dir = "NYU_Dataset_Imgs/NYU_RGB_imgs"
    depth_img_dir = "NYU_Dataset_Imgs/NYU_depth_imgs"
    image_names = os.listdir(imags_dir)
    depth_image_names = os.listdir(depth_img_dir)

    # 保存图像的文件夹
    dataset_dir = "MFI_dataset/NYU-D_" + str(dataset_num)
    fusion_dir = os.path.join(dataset_dir, "full_clear")
    I1_dir = os.path.join(dataset_dir, "source_1")
    I2_dir = os.path.join(dataset_dir, "source_2")
    if not os.path.isdir(fusion_dir):
        os.makedirs(fusion_dir)
    if not os.path.isdir(I1_dir):
        os.makedirs(I1_dir)
    if not os.path.isdir(I2_dir):
        os.makedirs(I2_dir)

    if dataset_num < len(image_names):
        indexs = random.randint(len(image_names), size=(dataset_num))
        num = 1
        for i in tqdm(indexs, desc="get MFI"):
            img_path = os.path.join(imags_dir, image_names[i])
            depth_img_path = os.path.join(depth_img_dir, depth_image_names[i])
            img_prime = cv2.imread(img_path)
            depth_img_prime = cv2.imread(depth_img_path)

            # 裁剪掉周围的白边, [row0 : row1, col0 : col1]
            img = img_prime[20:460, 20:620]
            depth_img = depth_img_prime[20:460, 20:620]

            I1, I2 = get_I1_I2(img, depth_img)

            # 保存图像
            fusion_path = os.path.join(fusion_dir, str(num) + ".jpg")
            I1_path = os.path.join(I1_dir, str(num) + ".jpg")
            I2_path = os.path.join(I2_dir, str(num) + ".jpg")
            cv2.imwrite(fusion_path, img)
            cv2.imwrite(I1_path, I1)
            cv2.imwrite(I2_path, I2)

            num += 1

    else:
        # 每张图像生成的图像数量
        img_nums = int(dataset_num / len(image_names))

        num = 1
        for i in tqdm(range(len(image_names)), desc="get MFI"):
            img_path = os.path.join(imags_dir, image_names[i])
            depth_img_path = os.path.join(depth_img_dir, depth_image_names[i])
            img_prime = cv2.imread(img_path)
            depth_img_prime = cv2.imread(depth_img_path)

            # 裁剪掉周围的白边
            img = img_prime[20:460, 20:620]  # [row0 : row1, col0 : col1]
            depth_img = depth_img_prime[20:460, 20:620]

            for j in range(img_nums):
                I1, I2 = get_I1_I2(img, depth_img)

                # 保存图像
                fusion_path = os.path.join(fusion_dir, str(num) + ".jpg")
                I1_path = os.path.join(I1_dir, str(num) + ".jpg")
                I2_path = os.path.join(I2_dir, str(num) + ".jpg")
                cv2.imwrite(fusion_path, img)
                cv2.imwrite(I1_path, I1)
                cv2.imwrite(I2_path, I2)

                num += 1

        # 随机选取 res_num 张图像生成新图像
        res_num = dataset_num-img_nums*len(image_names)
        indexs = random.randint(res_num, size=(res_num))
        for i in tqdm(indexs, desc="get MFI"):
            img_path = os.path.join(imags_dir, image_names[i])
            depth_img_path = os.path.join(depth_img_dir, depth_image_names[i])
            img_prime = cv2.imread(img_path)
            depth_img_prime = cv2.imread(depth_img_path)

            # 裁剪掉周围的白边
            img = img_prime[20:460, 20:620]  # [row0 : row1, col0 : col1]
            depth_img = depth_img_prime[20:460, 20:620]

            I1, I2 = get_I1_I2(img, depth_img)

            # 保存图像
            fusion_path = os.path.join(fusion_dir, str(num) + ".jpg")
            I1_path = os.path.join(I1_dir, str(num) + ".jpg")
            I2_path = os.path.join(I2_dir, str(num) + ".jpg")
            cv2.imwrite(fusion_path, img)
            cv2.imwrite(I1_path, I1)
            cv2.imwrite(I2_path, I2)

            num += 1