# Function: Extract label images from the NYU-D-v2 dataset


import numpy as np
import h5py
import os
from PIL import Image


if __name__ == '__main__':
    f = h5py.File("../nyu_depth_v2_labeled.mat")
    labels = f["labels"]
    labels = np.array(labels)

    path_converted = '../NYU_Dataset_Imgs/NYU_label_imgs/'
    if not os.path.isdir(path_converted):
        os.makedirs(path_converted)

    labels_number = []
    for i in range(len(labels)):
        print(f"process: {i}/{len(labels)}")
        labels_number.append(labels[i])
        labels_0 = np.array(labels_number[i])
        label_img = Image.fromarray(np.uint8(labels_number[i]))
        label_img = label_img.transpose(Image.ROTATE_270)

        iconpath = path_converted + str(i) + '.png'
        label_img.save(iconpath, 'png', optimize=True)