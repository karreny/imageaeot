import torch
from torch.utils.data import Dataset
from torchvision import transforms

import skimage.io as io
import pandas as pd
import numpy as np

import os

class CellImageDataset(Dataset):

    def __init__(self, datadir, metafile, mode='train'):
        self.datadir = datadir
        self.metadata = pd.read_csv(metafile)
        self.mode = mode

        if mode == 'train':
            self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()
            ])

        elif mode == 'val':
            self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
            ])

        else:
            raise KeyError("dataset mode must be one of ['train', 'val'], not %s" % mode)

    def __len__(self):
        return len(self.metadata)

    def load_image(self, fname):
        img = io.imread(fname, plugin='tifffile')
        img = img.astype(np.uint8)
        img, _ = torch.max(torch.from_numpy(img).view(1, 32, 64, 64),1)
        return img

    def __getitem__(self, idx):
        sample = self.metadata.iloc[idx]

        img = self.load_image(os.path.join(self.datadir, sample['filename']))
        img = self.transform(img)

        return {'image': img, 'label': sample['label'], 'label_name': sample['label_name'], 'key': sample['filename']}

