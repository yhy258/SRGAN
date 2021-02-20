
from torch.utils.data import Dataset
from skimage import io
import os
import torch.nn as nn
"""
    lr_path와 hr_path를 받아와서 실행 시켜볼까 한다. 기준 x4
"""

class SR_dataset(Dataset):
    def __init__(self, lr_path, hr_path, transform):
        self.lr_path = lr_path
        self.hr_path = hr_path
        self.transform = transform
        self.lr_image_names = os.listdir(lr_path)
        self.hr_image_names = os.listdir(hr_path)

    def __len__(self):
        return len(self.lr_image_names)

    def __getitem__(self, i):
        image_name = self.lr_image_names[i]
        lr_path = self.lr_path + "/" + image_name
        hr_path = self.hr_path + "/" + image_name
        lr_image = io.imread(lr_path)
        hr_image = io.imread(hr_path)

        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)

        return lr_image, hr_image
