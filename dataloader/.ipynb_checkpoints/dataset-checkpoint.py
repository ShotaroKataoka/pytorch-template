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
from config import Config

# instance of config
conf = Config()

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
    NUM_CLASSES = conf.num_class
    def __init__(self, split="train"):
        # Data Getter.
        ## ***Read label.csv (train_y)***
        label_path = conf.dataset_dir + "label.csv"
        y = pd.read_csv(label_path)["label"].values
        ids = pd.read_csv(label_path)["id"].values
        
        ## ***Get Image data path. (train_x)***
        img_path = []
        for id in ids:
            img_path += [conf.dataset_dir+"{}.png".format(id)]
        img_path = np.array(img_path)
        
        # Arrange data
        ## ***Shuffle data***
        img_path, y = shuffle(img_path, y, random_state=0)
        
        ## ***Define split length of train and validation.***
        train_len = int(img_path.shape[0] * conf.split_rate)

        ## ***Split data***
        if split=="train":
            self.img_path, self.y = img_path[:train_len], y[:train_len]
        elif split=="val":
            self.img_path, self.y = img_path[train_len:], y[train_len:]
        elif split=="test":
            self.img_path, self.y = None, None
        self.split = split

    def __getitem__(self, index):
        # Define "sample" which is dictionaly of {"input": x, "label": y}
        _img = Image.open(self.img_path[index]).convert('RGB')
        _target = self.y[index]
        sample = {'input': _img, 'label': _target}

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
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()
        ])
        return composed_transforms(sample)

    def transform_val(self, sample):
        """
        You can change transforms with <dataloader.custom_transforms>.
        """
        composed_transforms = transforms.Compose([
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()
        ])
        return composed_transforms(sample)

    def __len__(self):
        return len(self.img_path)