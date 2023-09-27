import os
from torch.utils.data import Dataset
from torchvision import transforms
import cv2


class MFI_Dataset(Dataset):
    def __init__(self, datasetPath, phase, use_dataTransform, resize, imgSzie):
        super(MFI_Dataset, self).__init__()
        self.datasetPath = datasetPath
        self.phase = phase
        self.use_dataTransform = use_dataTransform
        self.resize = resize
        self.imgSzie = imgSzie

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)])

    def __len__(self):
        dirsName = os.listdir(self.datasetPath)
        assert len(dirsName) >= 2, "Please check that the dataset is formatted correctly."
        dirsPath = os.path.join(self.datasetPath, dirsName[0])
        return len(os.listdir(dirsPath))

    def __getitem__(self, index):
        if self.phase == "train":
            # source image1
            sourceImg1_dirPath = os.path.join(self.datasetPath, "source_1")
            sourceImg1_names = os.listdir(sourceImg1_dirPath)
            sourceImg1_path = os.path.join(sourceImg1_dirPath, sourceImg1_names[index])
            sourceImg1 = cv2.imread(sourceImg1_path)

            # source image2
            sourceImg2_dirPath = os.path.join(self.datasetPath, "source_2")
            sourceImg2_names = os.listdir(sourceImg2_dirPath)
            sourceImg2_path = os.path.join(sourceImg2_dirPath, sourceImg2_names[index])
            sourceImg2 = cv2.imread(sourceImg2_path)

            # full_clear image
            clearImg_dirPath = os.path.join(self.datasetPath, "full_clear")
            clearImg_names = os.listdir(clearImg_dirPath)
            clearImg_path = os.path.join(clearImg_dirPath, clearImg_names[index])
            clearImg = cv2.imread(clearImg_path)

            if self.resize:
                sourceImg1 = cv2.resize(sourceImg1, (self.imgSzie, self.imgSzie))
                sourceImg2 = cv2.resize(sourceImg2, (self.imgSzie, self.imgSzie))
                clearImg = cv2.resize(clearImg, (self.imgSzie, self.imgSzie))
            if self.use_dataTransform:
                sourceImg1 = self.transform(sourceImg1)
                sourceImg2 = self.transform(sourceImg2)
                clearImg = self.transform(clearImg)

            return [sourceImg1, sourceImg2, clearImg]

        if self.phase == "valid":
            # source image1
            sourceImg1_dirPath = os.path.join(self.datasetPath, "source_1")
            sourceImg1_names = os.listdir(sourceImg1_dirPath)
            sourceImg1_path = os.path.join(sourceImg1_dirPath, sourceImg1_names[index])
            sourceImg1 = cv2.imread(sourceImg1_path)

            # source image2
            sourceImg2_dirPath = os.path.join(self.datasetPath, "source_2")
            sourceImg2_names = os.listdir(sourceImg2_dirPath)
            sourceImg2_path = os.path.join(sourceImg2_dirPath, sourceImg2_names[index])
            sourceImg2 = cv2.imread(sourceImg2_path)

            if self.resize:
                sourceImg1 = cv2.resize(sourceImg1, (self.imgSzie, self.imgSzie))
                sourceImg2 = cv2.resize(sourceImg2, (self.imgSzie, self.imgSzie))
            if self.use_dataTransform:
                sourceImg1 = self.transform(sourceImg1)
                sourceImg2 = self.transform(sourceImg2)

            return [sourceImg1, sourceImg2]