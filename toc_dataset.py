import os
import PIL.Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn import model_selection


class TocDataset(Dataset):
    def __init__(self, is_train=True):
        self.file = "data/csv.csv"
        self.img_dir = "data/gen"
        csv = pd.read_csv(self.file)
        train, test = model_selection.train_test_split(csv, test_size=0.2, random_state=2)
        self.data = test
        if is_train:
            self.data = train
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            # transforms.CenterCrop((2304,2304)),
            # transforms.Resize(128)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = PIL.Image.open(img_path)
        image = self.transforms(image)
        oc = self.data.iloc[idx, 1]
        return image, torch.tensor(oc, dtype=torch.float32)


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

