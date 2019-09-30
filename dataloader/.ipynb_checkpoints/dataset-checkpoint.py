import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import trange
import os
from pycocotools import mask
from torchvision import transforms
import custom_transforms as tr
from PIL import Image, ImageFile
from glob import glob
from sklearn.utils import shuffle
import pandas as pd

class Dataset():
    def __init__(self, base_dir="../sample_data/", split="train", split_rate=0.7):
        # read label.csv
        label_path = base_dir + "label.csv"
        y = pd.read_csv(label_path)["label"].values
        ids = pd.read_csv(label_path)["id"].values
        
        # get image path
        img_path = []
        for id in ids:
            img_path += [base_dir+"{}.png".format(id)]
        img_path = np.array(img_path)
        
        # shuffle data
        img_path, y = shuffle(img_path, y, random_state=0)
        
        # train_len -> split data to train and validation
        train_len = int(img_path.shape[0] * split_rate)

        if split=="train":
            self.img_path, self.y = img_path[:train_len], y[:train_len]
        else:
            self.img_path, self.y = img_path[train_len:], y[train_len:]
        self.split = split

    def __getitem__(self, index):
        _img = Image.open(self.img_path[index]).convert('RGB')
        _target = self.y[index]
        sample = {'image': _img, 'label': _target}

        if self.split == "train":
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()
        ])
        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()
        ])
        return composed_transforms(sample)

    def __len__(self):
        return len(self.img_path)