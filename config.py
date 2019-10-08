class Config(object):
    # general setting
    dataset_dir = "./sample_data/" # dataloader.dataset.Dataset.__init__()
    output_dir = "./run/"
    split_rate = 0.7 # dataloader.Dataset.__init__()
    
    # model setting
    input_channel = 3
    num_class = 2
    hidden_channel = 128
    hidden_layer = 4
    

class pycolor:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    PURPLE = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    END = '\033[0m'
    BOLD = '\038[1m'
    UNDERLINE = '\033[4m'
    INVISIBLE = '\033[08m'
    REVERCE = '\033[07m'
