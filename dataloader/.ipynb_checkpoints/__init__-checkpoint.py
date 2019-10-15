import sys

from torch.utils.data import DataLoader

from dataloader.dataset import Dataset
sys.path.append('..')
from config import Config

# instance of config
conf = Config()

def make_data_loader(batch_size=16):
    train_set = Dataset(split="train")
    val_set = Dataset(split="val")
    test_set = Dataset(split="test")
    num_class = train_set.NUM_CLASSES
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, num_class