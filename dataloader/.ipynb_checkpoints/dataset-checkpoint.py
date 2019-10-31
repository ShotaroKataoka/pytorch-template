from glob import glob
import sys

import torch
from torchvision import transforms
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from PIL import Image

import dataloader.custom_transforms as tr
sys.path.append('..')
from my_setting import MyPath

# setting of your directory path.
mypath = MyPath()

class Dataset():
    """
    This module is used to get and transform input data.
    You MUST change this module.
    
    __init__(): initializer. Prepare input data and split into train, val, test here.
    __getitem__(): This is implementation of "dataset[index]."
    transform_tr(): transform method for train data.
    transform_val(): transform method for validation data.
    __len__(): This is implementation of "len(dataset)."
    """
    NUM_CLASSES = mypath.num_class
    def __init__(self, split="train"):
        # Data Getter.
        ## ***Read label.csv (train_y)***
        label_path = mypath.dataset_dir + "label.csv"
        y = pd.read_csv(label_path)["label"].values
        ids = pd.read_csv(label_path)["id"].values
        
        ## ***Get Image data. (train_x)***
        img_path = ["{0}{1}.png".format(mypath.dataset_dir, i) for i in ids]
        x = [Image.open(p).convert('RGB') for p in img_path]
        
        # Arrange data
        ## ***Shuffle data***
        x, y = shuffle(x, y, random_state=0)
        
        ## ***Define split length of train and validation.***
        split_rate = 0.7
        train_len = int(y.shape[0] * split_rate)

        ## ***Split data***
        if split=="train":
            self.x, self.y = x[:train_len], y[:train_len]
        elif split=="val":
            self.x, self.y = x[train_len:], y[train_len:]
        elif split=="test":
            self.x, self.y = None, None
        self.split = split

    def __getitem__(self, index):
        # Define "sample" which is dictionaly of {"input": x, "label": y}
        _input = self.x[index]
        _target = self.y[index]
        sample = {'input': _input, 'label': _target}
        
        # Call transform for each "train", "val" and "test".
        if self.split == "train":
            return self.transform_tr(sample)
        elif self.split == "val":
            return self.transform_val(sample)
        elif self.split == "test":
            return self.transform_val(sample)

    def transform_tr(self, sample):
        """
        You can change transforms with <dataloader.custom_transforms>.
        """
        composed_transforms = transforms.Compose([
            tr.Resize(size=(64, 64)),
            tr.Normalize(mean=(0.30273438, 0.30273438, 0.30273438), std=(0.44050565, 0.44050565, 0.44050565)),
            tr.ToTensor()
        ])
        return composed_transforms(sample)

    def transform_val(self, sample):
        """
        You can change transforms with <dataloader.custom_transforms>.
        """
        composed_transforms = transforms.Compose([
            tr.Resize(size=(64, 64)),
            tr.Normalize(mean=(0.30273438, 0.30273438, 0.30273438), std=(0.44050565, 0.44050565, 0.44050565)),
            tr.ToTensor()
        ])
        return composed_transforms(sample)

    def __len__(self):
        return self.y.shape[0]