class Config(object):
    # general setting
    dataset_dir = "./sample_data/"
    output_dir = "./run/"
    split_rate = 0.7
    
    # model setting
    optimizer_name = ["Adam", "SGD"][0]
    num_classes = 2


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
