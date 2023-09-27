# Function: Extract depth RGB from the NYU-D-v2 dataset


import numpy as np
import h5py
import os
from PIL import Image


if __name__ == '__main__':
    f = h5py.File("../nyu_depth_v2_labeled.mat")
    images = f["images"]
    images = np.array(images)

    path_converted = '../NYU_Dataset_Imgs/NYU_RGB_imgs/'
    if not os.path.isdir(path_converted):
        os.makedirs(path_converted)

    images_number = []
    for i in range(len(images)):
        print(f"process: {i}/{len(images)}")
        images_number.append(images[i])
        a = np.array(images_number[i])

        r = Image.fromarray(a[0]).convert('L')
        g = Image.fromarray(a[1]).convert('L')
        b = Image.fromarray(a[2]).convert('L')
        img = Image.merge("RGB", (r, g, b))
        img = img.transpose(Image.ROTATE_270)

        iconpath = path_converted + str(i) + '.jpg'
        img.save(iconpath, optimize=True)