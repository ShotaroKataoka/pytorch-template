from torch.utils.data import DataLoader

from dataloaders.dataset import Dataset
import sys
sys.path.append('..')
from config import Config

conf = Config()

def make_data_loader():
    train_set = Dataset(split="train")
    val_set = Dataset(split="val")
    test_set = Dataset(split="test")
    num_class = train_set.NUM_CLASSES
    
    train_loader = DataLoader(train_set, batch_size=conf.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=conf.batch_size, shuffle=False)
    test_loader = None
    
    return train_loader, val_loader, test_loader, num_class