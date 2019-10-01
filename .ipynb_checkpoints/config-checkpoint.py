class Config(object):
    # general setting
    dataset_dir = "./sample_data/" # dataloader.dataset.Dataset.__init__()
    output_dir = "./run/"
    split_rate = 0.7 # dataloader.Dataset.__init__()
    
    # model setting
    input_channel = 3
    num_class = 5
    hidden_channel = 128
    hidden_layer = 3