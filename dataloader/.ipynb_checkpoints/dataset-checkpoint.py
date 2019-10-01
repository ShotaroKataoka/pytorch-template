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

import sys
sys.path.append('..')
from config import Config

conf = Config()

class Dataset():
    NUM_CLASSES = conf.num_class
    
    def __init__(self, split="train"):
        # Read label.csv
        label_path = conf.dataset_dir + "label.csv"
        y = pd.read_csv(label_path)["label"].values
        ids = pd.read_csv(label_path)["id"].values
        
        # Get image path
        img_path = []
        for id in ids:
            img_path += [conf.dataset_dir+"{}.png".format(id)]
        img_path = np.array(img_path)
        
        # Shuffle data
        img_path, y = shuffle(img_path, y, random_state=0)
        
        # train_len -> split data to train and validation
        train_len = int(img_path.shape[0] * conf.split_rate)

        if split=="train":
            self.img_path, self.y = img_path[:train_len], y[:train_len]
        elif split=="val":
            self.img_path, self.y = img_path[train_len:], y[train_len:]
        elif split=="test":
            self.img_path, self.y = None, None
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