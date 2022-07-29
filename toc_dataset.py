import os

import PIL.Image
import pandas as pd
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt


class TocDataset(Dataset):
    def __init__(self, is_train=True):
        self.file = "data/train/carbon.csv"
        self.img_dir = "data/train/images"
        if is_train is False:
            self.file = "data/test/carbon.csv"
            self.img_dir = "data/test/images"
        self.img_labels = pd.read_csv(self.file)
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.CenterCrop((2304,2304)),
            # transforms.Resize(128)
        ])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1])
        image = PIL.Image.open(img_path)
        image = self.transforms(image)
        label = self.img_labels.iloc[idx, 0]
        return image, torch.tensor(label, dtype=torch.float32)

if __name__ == "__main__":
    cid = TocDataset()
    dataloader = DataLoader(cid, batch_size=1, shuffle=True)
    
    for image, label in dataloader:
        print(image.shape)
        print(label)
        for i in image:
            plt.imshow(i[0].numpy())
            plt.show()
            exit(0)

